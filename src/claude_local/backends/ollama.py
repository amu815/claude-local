"""Ollama serving backend."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from typing import Any

from .base import Backend, BackendStatus


class OllamaBackend(Backend):
    """Backend that manages an Ollama server process."""

    name: str = "ollama"

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._port: int | None = None
        self._model: str | None = None

    # ------------------------------------------------------------------ #
    # Installation
    # ------------------------------------------------------------------ #

    def is_installed(self) -> bool:
        """Check whether the ``ollama`` binary is on PATH."""
        return shutil.which("ollama") is not None

    def install(self) -> None:
        """Install Ollama (or print instructions for the current OS)."""
        system = platform.system().lower()
        if system == "darwin":
            print(
                "Install Ollama for macOS:\n"
                "  brew install ollama\n"
                "  — or download from https://ollama.com/download/mac"
            )
        elif system == "windows":
            print(
                "Install Ollama for Windows:\n"
                "  Download from https://ollama.com/download/windows"
            )
        elif system == "linux":
            print("Running Ollama install script …")
            subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                check=True,
            )
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #

    def download_model(self, model: dict[str, Any]) -> None:
        """Pull a model via ``ollama pull <tag>``."""
        tag: str = model["backends"]["ollama"]["tag"]
        print(f"Pulling model {tag} …")
        subprocess.run(["ollama", "pull", tag], check=True)

    # ------------------------------------------------------------------ #
    # Server lifecycle
    # ------------------------------------------------------------------ #

    def start(self, model: dict[str, Any], port: int = 8000) -> None:
        """Start ``ollama serve`` and load the requested model."""
        if self._process is not None and self._process.poll() is None:
            print("Ollama is already running.")
            return

        tag: str = model["backends"]["ollama"]["tag"]
        endpoint = f"http://127.0.0.1:{port}"

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"

        self._process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._port = port
        self._model = tag

        # Wait for the server to become healthy.
        for _ in range(30):
            if self.health_check(endpoint):
                break
            time.sleep(1)
        else:
            self.stop()
            raise RuntimeError(
                f"Ollama failed to start within 30 s on port {port}"
            )

        # Preload / warm the model so it is ready for inference.
        try:
            import urllib.request
            import json

            body = json.dumps({"model": tag}).encode()
            req = urllib.request.Request(
                f"{endpoint}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300):
                pass
        except Exception:
            # Non-fatal — the model will load on first real request.
            pass

        print(f"Ollama serving {tag} on {endpoint}")

    def stop(self) -> None:
        """Terminate the managed Ollama server process."""
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None
        self._port = None
        self._model = None

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def status(self) -> BackendStatus:
        """Return the current status of the Ollama backend."""
        if self._process is None or self._process.poll() is not None:
            return BackendStatus(running=False)

        endpoint = (
            f"http://127.0.0.1:{self._port}" if self._port else None
        )
        return BackendStatus(
            running=True,
            pid=self._process.pid,
            endpoint=endpoint,
            model=self._model,
        )
