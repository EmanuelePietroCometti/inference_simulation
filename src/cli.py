"""Command-line argument parsing for the anomaly-detection inference/benchmark script."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anomaly-detection ONNX inference on a folder of images (contract-driven, "
                    "matching the C++ engine), save heatmap overlays, and benchmark batch 1 vs 17 "
                    "at the requested precision (fp32/fp16/int8)."
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Path to the ONNX model (FP32 model; TensorRT handles FP16/INT8 "
                             "conversion internally, no pre-quantized model required).")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder containing the input images to run inference on.")
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                        help="Folder where heatmaps and result files will be saved.")
    parser.add_argument("--extension", type=str, default=".bmp",
                        help="File extension of the input images (default: .bmp).")

    parser.add_argument("--device", type=str, default="tensorrt", choices=["cpu", "cuda", "tensorrt"],
                        help="Execution provider to use. Default: tensorrt, listed as "
                             "TensorRT->CUDA->CPU so ONNX Runtime falls back between EPs within one "
                             "session (see src/provider_setup.py).")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8"],
                        help="Precision requested from TensorRT (ignored for cpu/cuda).")
    parser.add_argument("--calibration_table", type=str, default=None,
                        help="Path to the native TensorRT INT8 calibration cache. "
                             "Required when --precision int8.")
    parser.add_argument("--engine_cache_dir", type=str, default="./trt_engines",
                        help="Directory where TensorRT engines are cached between runs "
                             "(namespaced per precision).")

    parser.add_argument("--threshold", type=float, default=None,
                        help="ABSOLUTE anomaly-score threshold in the model's RAW score units. "
                             "If omitted, the embedded image_threshold_raw (calibration contract) "
                             "is used, exactly like the C++ engine.")

    parser.add_argument("--colormap", type=str, default="JET",
                        choices=["JET", "TURBO", "INFERNO", "HOT"],
                        help="OpenCV colormap used to render the anomaly heatmap.")
    parser.add_argument("--overlay_alpha", type=float, default=0.5,
                        help="Blending factor between the heatmap and the original image "
                             "(0 = only original image, 1 = only heatmap).")

    parser.add_argument("--batch_sizes", type=str, default="1,17",
                        help="Comma-separated batch sizes to benchmark, e.g. '1,17'.")
    parser.add_argument("--warmup_iters", type=int, default=5,
                        help="Warm-up iterations before timing each batch size.")
    parser.add_argument("--timed_iters", type=int, default=20,
                        help="Timed iterations used to measure throughput for each batch size.")

    # OOM controls (see src/provider_setup.py).
    parser.add_argument("--gpu_mem_limit", type=int, default=None,
                        help="Optional CUDA/TensorRT memory cap in BYTES. Default: unset "
                             "(ORT uses the device's available VRAM). Set only to constrain usage.")
    parser.add_argument("--trt_workspace_gb", type=float, default=4.0,
                        help="TensorRT builder workspace size in GB (default: 4).")
    parser.add_argument("--trt_opt_batch", type=int, default=None,
                        help="Batch size TensorRT optimizes the engine for (profile 'opt' shape). "
                             "Default: the smallest requested batch, which keeps the engine BUILD "
                             "feasible (optimizing for large batches can need tens of GB and fail, "
                             "e.g. PatchCore). Larger batches still run and are skipped if they OOM.")

    return parser.parse_args()
