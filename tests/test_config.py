import pytest
from pathlib import Path
from claude_local.config import Config


@pytest.fixture
def tmp_config(tmp_path):
    return Config(config_dir=tmp_path)


def test_config_save_and_load(tmp_config):
    tmp_config.set("backend", "mlx")
    tmp_config.set("model.repo", "Intel/Qwen3.5-122B-A10B-int4-AutoRound")
    tmp_config.save()
    loaded = Config(config_dir=tmp_config.config_dir)
    assert loaded.get("backend") == "mlx"
    assert loaded.get("model.repo") == "Intel/Qwen3.5-122B-A10B-int4-AutoRound"


def test_config_defaults(tmp_config):
    assert tmp_config.get("proxy.host") == "127.0.0.1"
    assert tmp_config.get("proxy.port") == 8081


def test_config_upstreams(tmp_config):
    tmp_config.set("upstreams", ["http://127.0.0.1:8000"])
    tmp_config.save()
    loaded = Config(config_dir=tmp_config.config_dir)
    assert loaded.get("upstreams") == ["http://127.0.0.1:8000"]


def test_config_nonexistent_key(tmp_config):
    assert tmp_config.get("nonexistent") is None
    assert tmp_config.get("deeply.nested.key") is None


def test_config_as_dict(tmp_config):
    tmp_config.set("backend", "ollama")
    d = tmp_config.as_dict()
    assert d["backend"] == "ollama"
    # Verify it's a copy
    d["backend"] = "changed"
    assert tmp_config.get("backend") == "ollama"
