"""Tests for the model registry."""

import pytest

from claude_local.detect import PlatformInfo
from claude_local.models.registry import ModelRegistry


@pytest.fixture()
def registry() -> ModelRegistry:
    """Return a ModelRegistry loaded from the project's models.yaml."""
    return ModelRegistry()


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def test_registry_loads_models(registry: ModelRegistry):
    models = registry.list_models()
    assert len(models) >= 3
    assert any(m["id"] == "qwen3.5-122b-int4" for m in models)


def test_get_model_found(registry: ModelRegistry):
    m = registry.get_model("qwen3-32b-int4")
    assert m is not None
    assert m["family"] == "qwen3"


def test_get_model_not_found(registry: ModelRegistry):
    assert registry.get_model("nonexistent") is None


# ---------------------------------------------------------------------------
# Recommendation — 128 GB DGX Spark (vllm-spark backend)
# ---------------------------------------------------------------------------


def test_recommend_for_128gb_dgx_spark(registry: ModelRegistry):
    info = PlatformInfo(
        os="linux",
        arch="arm64",
        memory_gb=128,
        gpu_type="nvidia",
        gpu_vram_gb=0,
        is_dgx_spark=True,
        is_uma=True,
    )
    # available = int(128 * 0.85) = 108 GB
    # qwen3.5-122b-int4 needs 80 GB → fits, has vllm-spark backend
    rec = registry.recommend(info, backend="vllm-spark")
    assert rec["id"] == "qwen3.5-122b-int4"
    # Context resolved with total=128: threshold 128 met → 262144
    assert rec["max_context"] == 262144


# ---------------------------------------------------------------------------
# Recommendation — 64 GB macOS (mlx backend)
# ---------------------------------------------------------------------------


def test_recommend_for_64gb_macos(registry: ModelRegistry):
    info = PlatformInfo(
        os="darwin",
        arch="arm64",
        memory_gb=64,
        gpu_type="apple_silicon",
        gpu_vram_gb=0,
        is_dgx_spark=False,
        is_uma=True,
    )
    # available = int(64 * 0.85) = 54 GB
    # 122B needs 80 GB → skip, 32B needs 80 GB → skip, 32B-INT4 needs 24 GB → fits
    rec = registry.recommend(info, backend="mlx")
    assert rec["id"] == "qwen3-32b-int4"
    # Context: total=64, threshold 64 met → 131072
    assert rec["max_context"] == 131072


# ---------------------------------------------------------------------------
# Recommendation — 192 GB macOS (mlx backend)
# ---------------------------------------------------------------------------


def test_recommend_for_192gb_macos(registry: ModelRegistry):
    info = PlatformInfo(
        os="darwin",
        arch="arm64",
        memory_gb=192,
        gpu_type="apple_silicon",
        gpu_vram_gb=0,
        is_dgx_spark=False,
        is_uma=True,
    )
    # available = int(192 * 0.85) = 163 GB
    # 122B needs 80 GB, has mlx → fits (weight_size_gb=63, largest that fits)
    rec = registry.recommend(info, backend="mlx")
    assert rec["id"] == "qwen3.5-122b-int4"
    # Context: total=192, threshold 128 met → 262144
    assert rec["max_context"] == 262144


# ---------------------------------------------------------------------------
# Fallback — no backend match
# ---------------------------------------------------------------------------


def test_recommend_fallback_no_backend_match(registry: ModelRegistry):
    info = PlatformInfo(
        os="linux",
        arch="x86_64",
        memory_gb=16,
        gpu_type=None,
        gpu_vram_gb=0,
        is_dgx_spark=False,
        is_uma=False,
    )
    # available = int(16 * 0.70) = 11 GB — nothing fits any backend
    # Fallback: smallest model by weight_size_gb → qwen3-32b-int4 (18 GB weights)
    rec = registry.recommend(info, backend="vllm-spark")
    assert rec["id"] == "qwen3-32b-int4"
    # Context: total=16, no threshold met (min is 24) → fallback 4096
    assert rec["max_context"] == 4096


# ---------------------------------------------------------------------------
# Context resolution edge cases
# ---------------------------------------------------------------------------


def test_resolve_context_exact_threshold(registry: ModelRegistry):
    model = registry.get_model("qwen3-32b-int4")
    assert model is not None
    ctx = registry._resolve_context(model, 32)
    assert ctx == 32768


def test_resolve_context_below_all_thresholds(registry: ModelRegistry):
    model = registry.get_model("qwen3.5-122b-int4")
    assert model is not None
    ctx = registry._resolve_context(model, 64)
    assert ctx == 4096
