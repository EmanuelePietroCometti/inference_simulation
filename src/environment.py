"""
Captures the run environment for reproducibility.

A benchmark number without the machine it ran on is not reproducible. This module
records the GPU (name, VRAM, driver), the CUDA / TensorRT / ONNX Runtime / NumPy /
Python versions, the OS, the active EPs and the git commit, so every report header
states exactly where its timings came from. Everything is best-effort: a missing
``nvidia-smi`` or a detached git checkout degrades to ``None`` instead of failing.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from datetime import datetime, timezone


def _run(cmd: list) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - environment probing must never crash the run
        return None


def _gpu_info() -> dict:
    """Query the GPU via nvidia-smi (name, total VRAM, driver, clocks). Returns
    empty values when no NVIDIA GPU / driver is present."""
    q = _run(["nvidia-smi",
              "--query-gpu=name,memory.total,driver_version,clocks.max.graphics,clocks.current.graphics",
              "--format=csv,noheader,nounits"])
    info = {"name": None, "vram_total_mib": None, "driver_version": None,
            "max_graphics_clock_mhz": None, "current_graphics_clock_mhz": None,
            "clocks_locked": None}
    if not q:
        return info
    parts = [p.strip() for p in q.splitlines()[0].split(",")]
    if len(parts) >= 5:
        info["name"] = parts[0]
        info["vram_total_mib"] = _to_int(parts[1])
        info["driver_version"] = parts[2]
        info["max_graphics_clock_mhz"] = _to_int(parts[3])
        info["current_graphics_clock_mhz"] = _to_int(parts[4])
        if info["max_graphics_clock_mhz"] and info["current_graphics_clock_mhz"]:
            # Heuristic: clock at (or above) the max cap => locked/boosted steadily.
            info["clocks_locked"] = info["current_graphics_clock_mhz"] >= info["max_graphics_clock_mhz"]
    return info


def _to_int(s: str):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _versions() -> dict:
    versions = {"python": platform.python_version(), "onnxruntime": None,
                "numpy": None, "onnx": None, "tensorrt": None, "cuda_runtime": None}
    try:
        import onnxruntime as ort
        versions["onnxruntime"] = ort.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as np
        versions["numpy"] = np.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import onnx
        versions["onnx"] = onnx.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import tensorrt as trt  # only present when the TRT python package is installed
        versions["tensorrt"] = trt.__version__
    except Exception:  # noqa: BLE001
        pass
    # CUDA runtime as seen by nvidia-smi (the toolkit/runtime line).
    smi = _run(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"])
    if smi:
        versions["cuda_runtime"] = smi.splitlines()[0].strip()
    return versions


def collect_environment(active_providers: list | None = None) -> dict:
    """Assemble the full environment record for the report header."""
    git_commit = _run(["git", "rev-parse", "--short", "HEAD"])
    git_dirty = _run(["git", "status", "--porcelain"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or None,
        "gpu": _gpu_info(),
        "versions": _versions(),
        "active_providers": active_providers,
        "git_commit": git_commit,
        "git_dirty": bool(git_dirty) if git_dirty is not None else None,
    }
