"""Platform and memory detection for claude-local."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class PlatformInfo:
    """Detected platform information."""

    os: str  # "linux", "darwin", "windows"
    arch: str  # "arm64", "x86_64"
    memory_gb: int  # Total system memory in GB
    gpu_type: str | None  # "nvidia", "apple_silicon", None
    gpu_vram_gb: int  # GPU VRAM in GB (0 if UMA)
    is_dgx_spark: bool  # True if NVIDIA DGX Spark (GX10)
    is_uma: bool  # True if unified memory architecture


def _detect_os() -> str:
    """Detect operating system."""
    return platform.system().lower()


def _detect_arch() -> str:
    """Detect CPU architecture, normalised to arm64/x86_64."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "x86_64"


def _detect_memory_gb(os_name: str) -> int:
    """Detect total system memory in GB."""
    try:
        if os_name == "darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
            return int(out) // (1024 ** 3)

        if os_name == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Value is in kB
                        kb = int(line.split()[1])
                        return kb // (1024 ** 2)

        if os_name == "windows":
            out = subprocess.check_output(
                [
                    "powershell",
                    "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                ],
                text=True,
            ).strip()
            return int(out) // (1024 ** 3)
    except (OSError, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass

    return 0


def _detect_dgx_spark() -> bool:
    """Check if running on an NVIDIA DGX Spark (GX10)."""
    # Method 1: check /etc/dgx-release
    try:
        if os.path.exists("/etc/dgx-release"):
            with open("/etc/dgx-release") as f:
                content = f.read().lower()
            if "spark" in content or "gx10" in content:
                return True
    except OSError:
        pass

    # Method 2: check dpkg for dgx-ota package
    try:
        result = subprocess.run(
            ["dpkg", "-l", "dgx-ota"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "dgx-ota" in result.stdout:
            # On DGX Spark, the platform is GX10 (ARM-based)
            # Distinguish from other DGX models by checking architecture
            if platform.machine().lower() in ("arm64", "aarch64"):
                return True
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Method 3: check /sys/firmware/devicetree for GX10 model
    try:
        dt_model = "/sys/firmware/devicetree/base/model"
        if os.path.exists(dt_model):
            with open(dt_model, "rb") as f:
                model = f.read().decode("utf-8", errors="ignore").lower()
            if "gx10" in model or "dgx spark" in model:
                return True
    except OSError:
        pass

    return False


def _detect_gpu() -> tuple[str | None, int]:
    """Detect GPU type and VRAM.

    Returns:
        (gpu_type, gpu_vram_gb) where gpu_type is "nvidia", "apple_silicon",
        or None, and gpu_vram_gb is total VRAM in GB (0 for UMA).
    """
    os_name = platform.system().lower()
    arch = platform.machine().lower()

    # macOS + ARM → Apple Silicon (UMA, no discrete VRAM)
    if os_name == "darwin" and arch in ("arm64", "aarch64"):
        return "apple_silicon", 0

    # Check for NVIDIA GPU via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            ).strip()
            # Sum VRAM across all GPUs (each line is one GPU's VRAM in MiB)
            total_mb = 0
            for line in out.splitlines():
                line = line.strip()
                if line:
                    total_mb += int(line)
            vram_gb = total_mb // 1024
            return "nvidia", vram_gb
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError,
        ):
            # nvidia-smi exists but failed; still report nvidia with 0 VRAM
            return "nvidia", 0

    return None, 0


def detect_platform() -> PlatformInfo:
    """Detect OS, arch, memory, GPU, DGX Spark status."""
    os_name = _detect_os()
    arch = _detect_arch()
    memory_gb = _detect_memory_gb(os_name)
    is_dgx_spark = _detect_dgx_spark()
    gpu_type, gpu_vram_gb = _detect_gpu()

    # UMA: Apple Silicon (darwin) or DGX Spark (unified memory)
    is_uma = os_name == "darwin" or is_dgx_spark

    # On UMA systems the GPU shares system memory, so report 0 discrete VRAM
    if is_uma:
        gpu_vram_gb = 0

    return PlatformInfo(
        os=os_name,
        arch=arch,
        memory_gb=memory_gb,
        gpu_type=gpu_type,
        gpu_vram_gb=gpu_vram_gb,
        is_dgx_spark=is_dgx_spark,
        is_uma=is_uma,
    )


def recommend_backend(info: PlatformInfo) -> str:
    """Return best backend name for the platform."""
    if info.is_dgx_spark:
        return "vllm-spark"
    if info.os == "darwin" and info.gpu_type == "apple_silicon":
        return "mlx"
    if info.gpu_type == "nvidia":
        return "vllm"
    return "ollama"


def available_memory_for_model(info: PlatformInfo) -> int:
    """Estimate usable memory in GB for model loading."""
    if info.is_uma:
        return int(info.memory_gb * 0.85)
    if info.gpu_vram_gb > 0:
        return info.gpu_vram_gb
    # CPU-only fallback
    return int(info.memory_gb * 0.70)
