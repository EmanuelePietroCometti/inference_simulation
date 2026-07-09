"""Command-line argument parsing for the SuperSimpleNet inference script."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SuperSimpleNet ONNX inference on a folder of images, save anomaly "
                     "heatmaps normalized like the training pipeline, and benchmark batch 1 vs batch 17."
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

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "tensorrt"],
                         help="Execution provider to use.")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8"],
                         help="Precision requested from TensorRT (ignored for cpu/cuda).")
    parser.add_argument("--calibration_table", type=str, default=None,
                         help="Path to the native TensorRT INT8 calibration cache. "
                              "Required when --precision int8, generated with a dedicated calibration script.")
    parser.add_argument("--engine_cache_dir", type=str, default="./trt_engines",
                         help="Directory where TensorRT engines are cached between runs.")

    parser.add_argument("--threshold", type=float, default=None,
                         help="Anomaly score threshold, expressed in the folder-normalized [0, 1] "
                              "space (same convention as eval.py). If omitted, defaults to 0.5, "
                              "matching the fixed cutoff used by the training visualizer. "
                              "Prefer a threshold derived from a labeled validation set when available.")

    parser.add_argument("--blur_kernel_size", type=int, default=25,
                         help="Gaussian blur kernel size applied to the raw anomaly map, matching "
                              "model/supersimplenet.py's AnomalyMapGenerator (sigma=4 -> kernel=25). "
                              "The ONNX export disables this blur inside the graph.")
    parser.add_argument("--blur_sigma", type=float, default=4.0,
                         help="Gaussian blur sigma, matching AnomalyMapGenerator(sigma=4).")
    parser.add_argument("--no_blur", action="store_true",
                         help="Disable the post-processing Gaussian blur (use if your ONNX export "
                              "already includes it inside the graph).")

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

    return parser.parse_args()
