"""Tests for the backends package."""

from __future__ import annotations

import pytest

from claude_local.backends import Backend, get_backend


def test_get_ollama_backend():
    backend = get_backend("ollama")
    assert isinstance(backend, Backend)
    assert backend.name == "ollama"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nonexistent")


def test_ollama_is_installed():
    backend = get_backend("ollama")
    result = backend.is_installed()
    assert isinstance(result, bool)


def test_backend_status_dataclass():
    from claude_local.backends.base import BackendStatus

    s = BackendStatus(running=False)
    assert s.running is False
    assert s.pid is None
    assert s.endpoint is None
    assert s.model is None
    assert s.error is None


def test_ollama_status_when_not_started():
    backend = get_backend("ollama")
    st = backend.status()
    assert st.running is False
    assert st.pid is None


def test_ollama_stop_when_not_started():
    """Calling stop() without start() should be a no-op, not an error."""
    backend = get_backend("ollama")
    backend.stop()  # should not raise
