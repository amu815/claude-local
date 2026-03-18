"""Configuration management for claude-local."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "proxy": {
        "host": "127.0.0.1",
        "port": 8081,
    },
    "upstreams": [],
}


def _deep_copy(d: dict) -> dict:
    """Deep copy dicts and lists recursively."""
    return copy.deepcopy(d)


def _deep_merge(base: dict, override: dict) -> None:
    """Merge *override* into *base* in-place, recursing into nested dicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


class Config:
    """YAML-backed configuration with dotted-key access."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir: Path = config_dir or Path.home() / ".claude-local"
        self._path: Path = self.config_dir / "config.yaml"
        self._data: dict[str, Any] = _deep_copy(_DEFAULTS)

        if self._path.exists():
            with open(self._path, "r") as f:
                saved = yaml.safe_load(f)
            if isinstance(saved, dict):
                _deep_merge(self._data, saved)

    # ------------------------------------------------------------------

    def get(self, key: str) -> Any:
        """Get value by dotted key path.  E.g., ``'proxy.port'`` -> ``8081``."""
        parts = key.split(".")
        current: Any = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def set(self, key: str, value: Any) -> None:
        """Set value by dotted key path.  Creates intermediate dicts."""
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def save(self) -> None:
        """Save config to *config_dir*/config.yaml.  Creates dir if needed."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            yaml.safe_dump(self._data, f, default_flow_style=False)

    def as_dict(self) -> dict:
        """Return a deep copy of the config data."""
        return _deep_copy(self._data)
