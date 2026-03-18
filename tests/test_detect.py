"""Tests for platform and memory detection."""

import pytest

from claude_local.detect import (
    PlatformInfo,
    available_memory_for_model,
    detect_platform,
    recommend_backend,
)


# ---------------------------------------------------------------------------
# Live detection tests (run on any platform)
# ---------------------------------------------------------------------------


def test_detect_platform_returns_platform_info():
    info = detect_platform()
    assert isinstance(info, PlatformInfo)
    assert info.os in ("linux", "darwin", "windows")
    assert info.memory_gb > 0
    assert info.arch in ("arm64", "x86_64")


def test_detect_platform_has_gpu_info():
    info = detect_platform()
    assert info.gpu_type is None or isinstance(info.gpu_type, str)


def test_detect_platform_is_dgx_spark():
    info = detect_platform()
    assert isinstance(info.is_dgx_spark, bool)


# ---------------------------------------------------------------------------
# recommend_backend unit tests (mocked PlatformInfo)
# ---------------------------------------------------------------------------


def test_recommend_backend_dgx_spark():
    info = PlatformInfo(
        os="linux",
        arch="arm64",
        memory_gb=128,
        gpu_type="nvidia",
        gpu_vram_gb=0,
        is_dgx_spark=True,
        is_uma=True,
    )
    assert recommend_backend(info) == "vllm-spark"


def test_recommend_backend_macos():
    info = PlatformInfo(
        os="darwin",
        arch="arm64",
        memory_gb=192,
        gpu_type="apple_silicon",
        gpu_vram_gb=0,
        is_dgx_spark=False,
        is_uma=True,
    )
    assert recommend_backend(info) == "mlx"


def test_recommend_backend_nvidia_linux():
    info = PlatformInfo(
        os="linux",
        arch="x86_64",
        memory_gb=64,
        gpu_type="nvidia",
        gpu_vram_gb=24,
        is_dgx_spark=False,
        is_uma=False,
    )
    assert recommend_backend(info) == "vllm"


def test_recommend_backend_no_gpu():
    info = PlatformInfo(
        os="linux",
        arch="x86_64",
        memory_gb=32,
        gpu_type=None,
        gpu_vram_gb=0,
        is_dgx_spark=False,
        is_uma=False,
    )
    assert recommend_backend(info) == "ollama"


# ---------------------------------------------------------------------------
# available_memory_for_model unit tests
# ---------------------------------------------------------------------------


def test_available_memory_uma():
    info = PlatformInfo(
        os="darwin",
        arch="arm64",
        memory_gb=128,
        gpu_type="apple_silicon",
        gpu_vram_gb=0,
        is_dgx_spark=False,
        is_uma=True,
    )
    assert available_memory_for_model(info) == 108  # int(128 * 0.85)


def test_available_memory_discrete_gpu():
    info = PlatformInfo(
        os="linux",
        arch="x86_64",
        memory_gb=64,
        gpu_type="nvidia",
        gpu_vram_gb=24,
        is_dgx_spark=False,
        is_uma=False,
    )
    assert available_memory_for_model(info) == 24


def test_available_memory_cpu_only():
    info = PlatformInfo(
        os="linux",
        arch="x86_64",
        memory_gb=32,
        gpu_type=None,
        gpu_vram_gb=0,
        is_dgx_spark=False,
        is_uma=False,
    )
    assert available_memory_for_model(info) == 22  # int(32 * 0.70)


def test_available_memory_dgx_spark():
    info = PlatformInfo(
        os="linux",
        arch="arm64",
        memory_gb=128,
        gpu_type="nvidia",
        gpu_vram_gb=0,
        is_dgx_spark=True,
        is_uma=True,
    )
    assert available_memory_for_model(info) == 108  # int(128 * 0.85)
