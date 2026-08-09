"""Builds ONNX Runtime *session plans* for CPU / CUDA / TensorRT.

Execution provider
------------------
The GPU path uses the classic **TensorRT** execution provider
(``TensorrtExecutionProvider``) with ``trt_*`` options, listed as
``[TensorRT, CUDA, CPU]`` so a single session gets ONNX Runtime's native
**TensorRT -> CUDA -> CPU** fallback (a subgraph TensorRT cannot build/run drops
to CUDA, then CPU, inside the same session) — the same cascade as the C++ engine.

The TensorRT EP is configured like the C++ ``OrtsessionConfig.cpp``: device 0,
FP16/INT8 per ``--precision``, engine + timing caches (namespaced per precision
under ``--engine_cache_dir``), the most aggressive builder search
(``trt_builder_optimization_level=5``), and an explicit dynamic-batch profile
(``trt_profile_*_shapes``) covering the whole requested batch range so one engine
serves every batch size with no per-shape rebuild.

A "session plan" is a dict consumed by src/inference_engine.py:
  - ``{"label": str, "providers": [provider entries]}`` -> InferenceSession(providers=...)
Plans are tried in order; the first that builds a session wins.

OOM prevention
--------------
CUDA ``gpu_mem_limit`` is unset by default (ORT uses the device's VRAM); cap it
only via ``gpu_mem_limit``. The dynamic-batch profile's ``opt`` shape defaults to
the SMALLEST batch to keep the engine BUILD feasible (see _profile_shapes):
optimizing for the largest batch can need tens of GB and fail (e.g. PatchCore's
nearest-neighbour MatMul at batch 17). Workspace is ``trt_workspace_gb``.
"""

import os
from pathlib import Path

from src.utils import log, die


def _profile_shapes(input_name: str, batch_sizes: list, channels: int, height: int, width: int,
                    opt_batch: int | None):
    """Build the TensorRT profile-shape strings for the dynamic batch axis so one
    engine covers every requested batch size (min .. max).

    The optimization (``opt``) batch defaults to the SMALLEST size, not the
    largest. Optimizing for the largest batch is what makes the builder blow up:
    e.g. PatchCore's nearest-neighbour MatMul needs a ~27 GB autotuning buffer at
    batch 17, which no GPU can allocate. Optimizing for batch 1 keeps the build
    feasible; larger batches still run (dropping to CUDA/CPU or reported as OOM by
    the benchmark if they do not fit)."""
    bmin = min(batch_sizes)
    bmax = max(batch_sizes)
    bopt = opt_batch if opt_batch is not None else bmin
    bopt = max(bmin, min(bopt, bmax))  # clamp into [min, max]
    fmt = lambda b: f"{input_name}:{b}x{channels}x{height}x{width}"
    return fmt(bmin), fmt(bopt), fmt(bmax), bopt


def _cuda_options(gpu_mem_limit: int | None) -> dict:
    opts = {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",  # grow only as needed, avoids over-reserving
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "cudnn_conv_use_max_workspace": "1",  # let cuDNN pick the fastest conv algo (C++ parity)
        "do_copy_in_default_stream": True,
    }
    if gpu_mem_limit is not None:
        opts["gpu_mem_limit"] = int(gpu_mem_limit)
    return opts


def build_session_plans(device: str, precision: str, engine_cache_dir: str,
                        calibration_table: str | None, batch_sizes: list,
                        input_name: str = "image", input_channels: int = 3,
                        input_height: int = 256, input_width: int = 256,
                        gpu_mem_limit: int | None = None, trt_workspace_gb: float = 4.0,
                        trt_opt_batch: int | None = None) -> list:
    """Return an ordered list of session plans (see module docstring) for the
    requested device. inference_engine tries them in order and falls back down the
    list on failure."""
    cuda_plan = {"label": "CUDA",
                 "providers": [("CUDAExecutionProvider", _cuda_options(gpu_mem_limit)),
                               "CPUExecutionProvider"]}
    cpu_plan = {"label": "CPU", "providers": ["CPUExecutionProvider"]}

    if device == "tensorrt":
        # Namespace the engine cache per precision so fp16/int8/fp32 engines,
        # which are not interchangeable, never overwrite each other.
        cache_dir = os.path.abspath(os.path.join(engine_cache_dir, precision))
        os.makedirs(cache_dir, exist_ok=True)

        min_shapes, opt_shapes, max_shapes, bopt = _profile_shapes(
            input_name, batch_sizes, input_channels, input_height, input_width, trt_opt_batch)

        # TensorRT EP tuned like the C++ OrtsessionConfig.cpp: device 0, FP16/INT8
        # kernels, engine + timing caches, aggressive builder search, and a single
        # engine covering the whole batch range via the dynamic-batch profile.
        trt_options = {
            "device_id": 0,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": os.path.join(cache_dir, "trt_timing_cache"),
            "trt_builder_optimization_level": 5,
            "trt_max_workspace_size": int(trt_workspace_gb * 1024 * 1024 * 1024),
            "trt_fp16_enable": precision == "fp16",
            "trt_int8_enable": precision == "int8",
            "trt_profile_min_shapes": min_shapes,
            "trt_profile_opt_shapes": opt_shapes,
            "trt_profile_max_shapes": max_shapes,
        }

        if precision == "int8":
            if not calibration_table:
                die("--precision int8 requires --calibration_table (native TensorRT calibration cache).")
            table_path = Path(calibration_table).resolve()
            if not table_path.exists():
                die(f"Calibration table not found: {table_path}")
            trt_options["trt_int8_calibration_table_name"] = str(table_path)
            trt_options["trt_int8_use_native_calibration_table"] = True
            log(f"Using native TensorRT INT8 calibration table: {table_path}")

        log(f"Execution provider: TensorRT (precision={precision}, workspace={trt_workspace_gb} GB, "
            f"profile batch min/opt/max={min(batch_sizes)}/{bopt}/{max(batch_sizes)})")
        # TensorRT -> CUDA -> CPU in a single session (ORT's native EP fallback).
        trt_plan = {"label": "TensorRT",
                    "providers": [("TensorrtExecutionProvider", trt_options),
                                  ("CUDAExecutionProvider", _cuda_options(gpu_mem_limit)),
                                  "CPUExecutionProvider"]}
        # Keep a CUDA-only plan as a last resort if the whole TRT session fails to
        # build (e.g. a hard builder error rather than a per-node fallback).
        return [trt_plan, cuda_plan]

    if device == "cuda":
        log("Execution provider: CUDA"
            + (f" (gpu_mem_limit={gpu_mem_limit} bytes)" if gpu_mem_limit is not None else ""))
        return [cuda_plan]

    log("Execution provider: CPU")
    return [cpu_plan]
