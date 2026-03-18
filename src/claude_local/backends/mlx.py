"""MLX backend for claude-local (macOS Apple Silicon only)."""
from __future__ import annotations

import subprocess
import sys
import time
from typing import Any

from .base import Backend, BackendStatus


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self):
        self._process: subprocess.Popen | None = None

    def is_installed(self) -> bool:
        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False

    def install(self) -> None:
        if self.is_installed():
            return
        if sys.platform != "darwin":
            raise RuntimeError(
                "MLX backend is only supported on macOS with Apple Silicon"
            )
        print("Installing mlx-lm...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "mlx-lm"], check=True
        )

    def download_model(self, model: dict[str, Any]) -> None:
        mlx_repo = model.get("backends", {}).get("mlx", {}).get("repo")
        if not mlx_repo:
            raise ValueError(f"No MLX repo defined for model {model['id']}")
        print(f"Downloading {mlx_repo}...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(mlx_repo)
        except ImportError:
            print(f"Model will be downloaded on first start from {mlx_repo}")

    def start(self, model: dict[str, Any], port: int = 8000) -> None:
        mlx_config = model.get("backends", {}).get("mlx", {})
        mlx_repo = mlx_config.get("repo", model["repo"])

        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlx_lm.server",
                "--model",
                mlx_repo,
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        endpoint = f"http://127.0.0.1:{port}"
        for _ in range(120):
            if self.health_check(endpoint):
                break
            if self._process.poll() is not None:
                stderr = (
                    self._process.stderr.read().decode()
                    if self._process.stderr
                    else ""
                )
                raise RuntimeError(
                    f"MLX server exited unexpectedly: {stderr}"
                )
            time.sleep(1)
        else:
            raise RuntimeError("MLX server did not start within 120s")

    def stop(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def status(self) -> BackendStatus:
        if self._process and self._process.poll() is None:
            return BackendStatus(running=True, pid=self._process.pid)
        return BackendStatus(running=False)
