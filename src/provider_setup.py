"""Builds the ONNX Runtime execution provider list for CPU / CUDA / TensorRT."""

import os
from pathlib import Path

from src.utils import log, die


def build_providers(device: str, precision: str, engine_cache_dir: str, calibration_table: str | None) -> list:
    """
    Build the ONNX Runtime provider list based on the requested device and precision.

    Precision only affects the tensorrt device: TensorRT converts the FP32 ONNX model to
    FP16/INT8 internally while building the engine, no pre-quantized ONNX model is needed.
    For INT8, a native TensorRT calibration cache must already exist (see the dedicated
    calibration script) and is passed in as calibration_table.
    """
    if device == "tensorrt":
        os.makedirs(engine_cache_dir, exist_ok=True)

        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": engine_cache_dir,
            "trt_max_workspace_size": 10 * 1024 * 1024 * 1024,
            "trt_fp16_enable": precision in ("fp16", "int8"),
            "trt_int8_enable": precision == "int8",
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

        log(f"Execution provider: TensorRT (precision={precision})")
        return [
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    if device == "cuda":
        log("Execution provider: CUDA")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    log("Execution provider: CPU")
    return ["CPUExecutionProvider"]
