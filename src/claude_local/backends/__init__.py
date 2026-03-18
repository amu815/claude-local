"""Serving-backend registry and factory."""

from __future__ import annotations

from .base import Backend, BackendStatus

__all__ = ["Backend", "BackendStatus", "get_backend"]


def get_backend(name: str) -> Backend:
    """Return a backend instance by name.

    Imports are lazy so that missing optional dependencies (e.g. MLX on
    Linux) don't break the whole package.
    """
    if name == "ollama":
        from .ollama import OllamaBackend

        return OllamaBackend()
    elif name == "mlx":
        from .mlx import MLXBackend  # type: ignore[import-not-found]

        return MLXBackend()
    elif name in ("vllm-spark", "vllm"):
        from .vllm_spark import VLLMSparkBackend  # type: ignore[import-not-found]

        return VLLMSparkBackend()
    raise ValueError(f"Unknown backend: {name}")
