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
                         help="ABSOLUTE anomaly-score threshold in the model's RAW score units "
                              "(the pure ONNX graph outputs raw scores: no sigmoid / no min-max). "
                              "Use the value your model's eval prints, e.g. SK-RD4AD eval.py's "
                              "'best_threshold_raw' (F1-optimal on raw scores). If omitted, a "
                              "NON-calibrated fallback (midpoint of the folder's raw score range) "
                              "is used and a warning is logged — verdicts are then unreliable.")

    parser.add_argument("--blur_kernel_size", type=int, default=None,
                         help="Gaussian blur kernel size applied to the raw anomaly map before "
                              "display/scoring. Default: auto-detected from the ONNX model's "
                              "embedded metadata (see src/model_config.py). Pass explicitly only "
                              "to override the model's own declared value.")
    parser.add_argument("--blur_sigma", type=float, default=None,
                         help="Gaussian blur sigma. Default: auto-detected from the ONNX model's "
                              "metadata. 0 lets OpenCV derive sigma from the kernel size.")
    parser.add_argument("--score_source", type=str, default="auto",
                         choices=["auto", "graph", "map_max_blurred"],
                         help="Where the image-level anomaly score comes from. 'auto' (default) "
                              "reads it from the ONNX model's embedded metadata. 'graph' uses the "
                              "graph's anomaly_score output directly (SuperSimpleNet: a dedicated "
                              "classification head). 'map_max_blurred' derives it from the max of "
                              "the blurred anomaly map (SK-RD4AD: eval.py calibrates its threshold "
                              "this way, NOT on the graph's raw anomaly_score output). Getting this "
                              "wrong silently produces a real but uncalibrated number - it will not "
                              "crash, so 'auto' should be preferred unless you have a specific reason.")
    parser.add_argument("--no_blur", action="store_true",
                         help="Disable the post-processing Gaussian blur (use if your ONNX export "
                              "already includes it inside the graph).")

    parser.add_argument("--normalize", type=str, default="per_image", choices=["per_image", "folder"],
                         help="How to min-max normalize the anomaly map for heatmap DISPLAY only "
                              "(never affects the OK/ANOMALY verdict, which always uses the absolute "
                              "raw --threshold). 'per_image' (default) stretches each image's own "
                              "map to [0,1], so the local defect region stands out regardless of "
                              "baseline differences between images (texture/lighting) that otherwise "
                              "compress everything into one end of a narrow folder-wide range, making "
                              "every heatmap look uniformly red. 'folder' uses the folder-wide min/max "
                              "(comparable brightness across images, but can wash out low-contrast "
                              "raw signals like SK-RD4AD's narrow cosine-distance range).")
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
