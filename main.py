"""
SuperSimpleNet ONNX inference over a folder of images.

Runs the exported ONNX model (FP32 / FP16 / INT8 via TensorRT, CUDA, or CPU) on
every image in a folder and:
  - saves an anomaly heatmap overlay per image, normalized using folder-wide
    statistics and re-blurred to match the training evaluation pipeline
    (eval.py + model/supersimplenet.py's AnomalyMapGenerator)
  - classifies each image using a threshold in that same normalized [0, 1] space
  - benchmarks inference throughput at batch size 1 and batch size 17 (or any
    sizes passed via --batch_sizes)

All logic lives in the src/ package; this file only wires the modules together.
"""

import cv2

from src.cli import parse_args
from src.utils import log, list_images, ensure_dir
from src.provider_setup import build_providers
from src.inference_engine import AnomalyInferenceEngine
from src.raw_cache import save_raw, load_raw, cleanup as cleanup_cache
from src.postprocessing import (
    apply_training_blur,
    normalize_global,
    build_heatmap_overlay,
    annotate_result,
)
from src.batch_benchmark import run_batch_comparison
from src.results_writer import write_benchmark_results


def collect_raw_outputs(engine, image_paths, cache_dir, args):
    """
    Pass 1/2: run inference once per image, apply the training Gaussian blur,
    cache raw outputs to disk, and track folder-wide min/max for both the
    anomaly map and the anomaly score.
    """
    log(f"Running inference on {len(image_paths)} images (pass 1/2: collecting raw outputs)...")

    map_min, map_max = float("inf"), float("-inf")
    score_min, score_max = float("inf"), float("-inf")
    total_time = 0.0

    for image_path in image_paths:
        batch_tensor = engine.preprocess(str(image_path))
        anomaly_maps, anomaly_scores, elapsed = engine.run_batch(batch_tensor)
        total_time += elapsed

        anomaly_map = anomaly_maps[0, 0]
        if not args.no_blur:
            anomaly_map = apply_training_blur(anomaly_map, args.blur_kernel_size, args.blur_sigma)

        # Image score. For SK-RD4AD the eval threshold is calibrated on the max of
        # the *blurred* map, so --score_from_map reproduces that exactly. For
        # SuperSimpleNet the score is a dedicated classification head, so we use
        # the graph's anomaly_score output as-is.
        if args.score_from_map:
            anomaly_score = float(anomaly_map.max())
        else:
            anomaly_score = float(anomaly_scores[0])

        save_raw(cache_dir, image_path, anomaly_map, anomaly_score)

        map_min = min(map_min, float(anomaly_map.min()))
        map_max = max(map_max, float(anomaly_map.max()))
        score_min = min(score_min, anomaly_score)
        score_max = max(score_max, anomaly_score)

    return {
        "map_min": map_min,
        "map_max": map_max,
        "score_min": score_min,
        "score_max": score_max,
        "total_time_sec": total_time,
    }


def render_heatmaps(engine, image_paths, cache_dir, heatmap_dir, stats, threshold, args):
    """Pass 2/2: reload cached raw outputs, normalize with folder-wide stats, render and save."""
    log("Rendering heatmaps (pass 2/2: folder-wide normalization)...")
    per_image_records = []

    for image_path in image_paths:
        anomaly_map, anomaly_score = load_raw(cache_dir, image_path)

        # The map is min-max normalized folder-wide for DISPLAY only (comparable
        # heatmaps across images). The verdict does NOT use any normalization: it
        # compares the RAW score against the absolute threshold, exactly like the
        # model's eval.py (raw_score >= best_threshold_raw).
        normalized_map = normalize_global(anomaly_map, stats["map_min"], stats["map_max"])
        is_anomalous = bool(anomaly_score >= threshold)

        original_bgr = engine.preprocessor.load_original_bgr(str(image_path))
        overlay = build_heatmap_overlay(original_bgr, normalized_map, args.colormap, args.overlay_alpha)
        annotated = annotate_result(overlay, anomaly_score, threshold, is_anomalous)

        out_path = heatmap_dir / f"{image_path.stem}_heatmap.png"
        cv2.imwrite(str(out_path), annotated)

        per_image_records.append({
            "filename": image_path.name,
            "raw_anomaly_score": round(anomaly_score, 6),
            "is_anomalous": is_anomalous,
        })

    return per_image_records


def main() -> None:
    args = parse_args()

    providers = build_providers(
        device=args.device,
        precision=args.precision,
        engine_cache_dir=args.engine_cache_dir,
        calibration_table=args.calibration_table,
    )

    engine = AnomalyInferenceEngine(args.model, providers)
    image_paths = list_images(args.input_dir, args.extension)

    heatmap_dir = ensure_dir(f"{args.output_dir}/heatmaps")
    cache_dir = ensure_dir(f"{args.output_dir}/.raw_cache")

    stats = collect_raw_outputs(engine, image_paths, cache_dir, args)
    log(f"Folder-wide anomaly map range: [{stats['map_min']:.4f}, {stats['map_max']:.4f}]")
    log(f"Folder-wide anomaly score range: [{stats['score_min']:.4f}, {stats['score_max']:.4f}]")

    if args.threshold is not None:
        threshold = args.threshold
        threshold_source = "user-provided ABSOLUTE raw-score threshold"
    else:
        # No sensible fixed default exists in raw-score units (they are unbounded
        # and model-specific), so fall back to the midpoint of the folder's raw
        # score range purely so the run produces *something* — and warn loudly
        # that verdicts are not trustworthy without a calibrated threshold.
        threshold = 0.5 * (stats["score_min"] + stats["score_max"])
        threshold_source = "NON-calibrated fallback (folder raw-score midpoint)"
        log("WARNING: no --threshold given. Raw scores have no fixed 0.5 boundary; "
            "using the folder raw-score midpoint as a rough fallback. Pass the "
            "absolute threshold from your model's eval (e.g. SK-RD4AD "
            "'best_threshold_raw') for reliable OK/ANOMALY verdicts.")
    log(f"Using threshold: {threshold:.6f} ({threshold_source})")

    per_image_records = render_heatmaps(engine, image_paths, cache_dir, heatmap_dir, stats, threshold, args)
    cleanup_cache(cache_dir)
    log(f"Heatmaps saved to: {heatmap_dir}")

    num_images = len(image_paths)
    num_anomalous = sum(r["is_anomalous"] for r in per_image_records)

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    log(f"Running throughput benchmark for batch sizes: {batch_sizes}")
    batch_comparison = run_batch_comparison(
        engine, image_paths, batch_sizes,
        warmup_iters=args.warmup_iters, timed_iters=args.timed_iters,
    )

    summary = {
        "model": args.model,
        "device": args.device,
        "precision": args.precision,
        "active_providers": engine.session.get_providers(),
        "num_images": num_images,
        "num_anomalous": num_anomalous,
        "num_normal": num_images - num_anomalous,
        "map_normalization_range": [round(stats["map_min"], 6), round(stats["map_max"], 6)],
        "score_normalization_range": [round(stats["score_min"], 6), round(stats["score_max"], 6)],
        "threshold": round(threshold, 6),
        "threshold_source": threshold_source,
        "pass1_inference_total_time_sec": round(stats["total_time_sec"], 3),
        "batch_throughput_comparison": batch_comparison,
    }
    write_benchmark_results(args.output_dir, summary, per_image_records)

    log("Done.")


if __name__ == "__main__":
    main()
