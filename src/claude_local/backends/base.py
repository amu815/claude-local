"""Abstract base class for serving backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class BackendStatus:
    """Status information for a backend process."""

    running: bool
    pid: int | None = None
    endpoint: str | None = None
    model: str | None = None
    error: str | None = None


class Backend(abc.ABC):
    """Base class that all serving backends must implement."""

    name: str = "base"

    @abc.abstractmethod
    def install(self) -> None:
        """Install the backend runtime."""
        ...

    @abc.abstractmethod
    def is_installed(self) -> bool:
        """Return True if the backend binary/runtime is available."""
        ...

    @abc.abstractmethod
    def download_model(self, model: dict[str, Any]) -> None:
        """Download / pull the model weights."""
        ...

    @abc.abstractmethod
    def start(self, model: dict[str, Any], port: int = 8000) -> None:
        """Start the backend server for the given model."""
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the running backend server."""
        ...

    @abc.abstractmethod
    def status(self) -> BackendStatus:
        """Return the current status of the backend."""
        ...

    def health_check(self, endpoint: str) -> bool:
        """Probe the /v1/models endpoint; return True if reachable."""
        import urllib.request

        try:
            req = urllib.request.Request(f"{endpoint}/v1/models")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False
