"""vLLM backend for DGX Spark (via spark-vllm-docker)."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import Backend, BackendStatus


class VLLMSparkBackend(Backend):
    name = "vllm-spark"

    def __init__(
        self,
        spark_vllm_path: str | None = None,
        nodes: list[str] | None = None,
    ):
        self._spark_vllm_path = Path(
            spark_vllm_path
            or os.environ.get(
                "SPARK_VLLM_DOCKER_PATH",
                os.path.expanduser("~/spark-vllm-docker"),
            )
        )
        self._nodes = nodes or []

    def is_installed(self) -> bool:
        run_recipe = self._spark_vllm_path / "run-recipe.py"
        return run_recipe.exists() and shutil.which("docker") is not None

    def install(self) -> None:
        if self.is_installed():
            return
        print(f"spark-vllm-docker not found at {self._spark_vllm_path}")
        print("Clone it:")
        print(
            f"  git clone https://github.com/eugr/spark-vllm-docker {self._spark_vllm_path}"
        )
        raise RuntimeError("spark-vllm-docker must be installed manually")

    def download_model(self, model: dict[str, Any]) -> None:
        recipe = model.get("backends", {}).get("vllm-spark", {}).get("recipe")
        if not recipe:
            raise ValueError(f"No vllm-spark recipe for model {model['id']}")
        print(f"Downloading model via recipe: {recipe}")
        subprocess.run(
            ["python3", "run-recipe.py", recipe, "--download-only"],
            cwd=self._spark_vllm_path,
            check=True,
        )

    def start(self, model: dict[str, Any], port: int = 8000) -> None:
        recipe = model.get("backends", {}).get("vllm-spark", {}).get("recipe")
        if not recipe:
            raise ValueError(f"No vllm-spark recipe for model {model['id']}")

        # Stop existing containers
        subprocess.run(
            ["bash", "launch-cluster.sh", "stop"],
            cwd=self._spark_vllm_path,
            capture_output=True,
        )

        # Start local node
        print(f"Starting vLLM with recipe: {recipe}")
        subprocess.run(
            ["python3", "run-recipe.py", recipe, "--solo", "-d"],
            cwd=self._spark_vllm_path,
            check=True,
        )

        # Start remote nodes
        for node in self._nodes[1:]:
            print(f"Starting vLLM on {node}...")
            subprocess.run(
                [
                    "ssh",
                    node,
                    f"cd {self._spark_vllm_path} && python3 run-recipe.py {recipe} --solo -d",
                ],
                check=True,
            )

        # Wait for readiness (up to 15 min)
        endpoints = self._get_endpoints(port)
        self._wait_for_ready(endpoints, timeout=900)

    def stop(self) -> None:
        subprocess.run(
            ["bash", "launch-cluster.sh", "stop"],
            cwd=self._spark_vllm_path,
            capture_output=True,
        )
        for node in self._nodes[1:]:
            subprocess.run(
                [
                    "ssh",
                    node,
                    f"cd {self._spark_vllm_path} && bash launch-cluster.sh stop",
                ],
                capture_output=True,
            )

    def status(self) -> BackendStatus:
        endpoints = self._get_endpoints(8000)
        for ep in endpoints:
            if self.health_check(ep):
                return BackendStatus(running=True, endpoint=ep)
        return BackendStatus(running=False)

    def _get_endpoints(self, port: int) -> list[str]:
        if self._nodes:
            return [f"http://{n}:{port}" for n in self._nodes]
        return [f"http://127.0.0.1:{port}"]

    def _wait_for_ready(self, endpoints: list[str], timeout: int = 900) -> None:
        start = time.time()
        remaining = set(endpoints)
        while remaining and (time.time() - start) < timeout:
            for ep in list(remaining):
                if self.health_check(ep):
                    print(f"  {ep} is READY")
                    remaining.discard(ep)
            if remaining:
                time.sleep(10)
        if remaining:
            raise RuntimeError(f"Nodes not ready after {timeout}s: {remaining}")
