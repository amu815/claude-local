"""Model registry — catalog of supported models and memory-based recommendation."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from claude_local.detect import PlatformInfo

# Default catalog location: <project_root>/configs/models.yaml
_DEFAULT_CATALOG = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


class ModelRegistry:
    """Load the model catalog and recommend models based on platform info."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        path = catalog_path or _DEFAULT_CATALOG
        with open(path) as f:
            data = yaml.safe_load(f)
        self._models: list[dict] = data.get("models", [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return all models in the catalog."""
        return list(self._models)

    def get_model(self, model_id: str) -> dict | None:
        """Find a model by its id string."""
        for m in self._models:
            if m["id"] == model_id:
                return dict(m)
        return None

    def recommend(self, info: PlatformInfo, backend: str) -> dict:
        """Pick the best model for *backend* given the platform's memory.

        Selection rules:
        1. Compute available memory via ``available_memory_for_model(info)``.
        2. Keep models whose *backend* is listed **and** ``memory_gb <= available``.
        3. Among those, pick the largest by parameter count (descending).
        4. Resolve ``max_context`` using total system memory (``info.memory_gb``).
        5. Fallback: if no backend-matched model fits, pick the smallest model overall.
        """
        from claude_local.detect import available_memory_for_model

        available = available_memory_for_model(info)

        # Candidates: backend supported AND memory requirement met
        candidates = [
            m
            for m in self._models
            if backend in m.get("backends", {}) and m["memory_gb"] <= available
        ]

        if candidates:
            # Pick largest model that fits (by parameter count, then weight size)
            candidates.sort(
                key=lambda m: (self._parse_params(m), m["weight_size_gb"]),
                reverse=True,
            )
            best = copy.deepcopy(candidates[0])
        else:
            # Fallback: smallest model regardless of backend
            fallback = sorted(
                self._models,
                key=lambda m: (self._parse_params(m), m["weight_size_gb"]),
            )
            best = copy.deepcopy(fallback[0])

        best["max_context"] = self._resolve_context(best, info.memory_gb)
        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_params(model: dict) -> float:
        """Parse the ``parameters`` field (e.g. '122B') into a numeric value."""
        raw = str(model.get("parameters", "0"))
        match = re.match(r"([0-9.]+)", raw)
        return float(match.group(1)) if match else 0.0

    def _resolve_context(self, model: dict, total_memory_gb: int) -> int:
        """Walk *context_by_memory* and return the largest context length
        whose memory threshold is <= *total_memory_gb*.

        Thresholds are compared against **total** system memory (not available
        memory) because the YAML values represent system-class tiers.

        Falls back to 4096 if no threshold is met.
        """
        mapping = model.get("context_by_memory", {})
        best_ctx = 4096
        best_threshold = -1
        for threshold, ctx in mapping.items():
            threshold = int(threshold)
            if threshold <= total_memory_gb and threshold > best_threshold:
                best_threshold = threshold
                best_ctx = ctx
        return best_ctx
