"""
SuperSimpleNet ONNX inference over a folder of images.

Runs the exported ONNX model (FP32 / FP16 / INT8 via TensorRT, CUDA, or CPU) on
every image in a folder and:
  - saves an anomaly heatmap overlay per image, normalized for display (per-image
    by default, or folder-wide via --normalize folder) and re-blurred to match
    the training evaluation pipeline (eval.py + model/supersimplenet.py's
    AnomalyMapGenerator)
  - classifies each image using the absolute RAW-score --threshold (unrelated to
    the display normalization)
  - benchmarks inference throughput at batch size 1 and batch size 17 (or any
    sizes passed via --batch_sizes)

All logic lives in the src/ package; this file only wires the modules together.
"""

import cv2

from src.cli import parse_args
from src.utils import log, die, list_images, ensure_dir
from src.provider_setup import build_providers
from src.inference_engine import AnomalyInferenceEngine
from src.raw_cache import save_raw, load_raw, cleanup as cleanup_cache
from src.postprocessing import (
    apply_training_blur,
    normalize_global,
    normalize_threshold_centric,
    build_heatmap_overlay,
    annotate_result,
)
from src.model_config import resolve_runtime_config
from src.batch_benchmark import run_batch_comparison
from src.results_writer import write_benchmark_results


def collect_raw_outputs(engine, image_paths, cache_dir, args, runtime_config):
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
        # Host-side blur only when the graph does NOT already include it
        # (contract 2.0 bakes the canonical blur into the graph; blurring again
        # would invalidate the calibrated threshold - see src/model_config.py).
        if not args.no_blur and not runtime_config.blur_in_graph:
            anomaly_map = apply_training_blur(
                anomaly_map, runtime_config.blur_kernel_size, runtime_config.blur_sigma
            )

        # Image score: which output to threshold on is architecture-specific (see
        # src/model_config.py). Contract 2.0 models emit a directly-thresholdable
        # anomaly_score (max of the in-graph blurred map); contract 1.0 SK-RD4AD
        # models need it derived host-side from the max of the blurred map.
        if runtime_config.score_source == "map_max_blurred":
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


def render_heatmaps(engine, image_paths, cache_dir, heatmap_dir, stats, threshold, args,
                    display_mode):
    """Pass 2/2: reload cached raw outputs, normalize for display, render and save."""
    log(f"Rendering heatmaps (pass 2/2, display normalization: {display_mode})...")
    per_image_records = []

    for image_path in image_paths:
        anomaly_map, anomaly_score = load_raw(cache_dir, image_path)

        # Normalization here is for DISPLAY only; the verdict always compares the
        # RAW score against the absolute threshold.
        #   threshold : eval.py-style threshold-centric coloring (below-threshold
        #               pixels stay cold, above-threshold render warm) - the
        #               cleanest view, but needs a calibrated threshold.
        #   per_image : each image's own min-max (good without a threshold, but
        #               stretches the noise floor -> speckled background).
        #   folder    : folder-wide min-max (cross-image comparable brightness).
        if display_mode == "threshold":
            normalized_map = normalize_threshold_centric(anomaly_map, threshold)
        elif display_mode == "folder":
            normalized_map = normalize_global(anomaly_map, stats["map_min"], stats["map_max"])
        else:
            normalized_map = normalize_global(anomaly_map, float(anomaly_map.min()), float(anomaly_map.max()))
        is_anomalous = bool(anomaly_score >= threshold)

        original_bgr = engine.preprocessor.load_original_bgr(str(image_path))
        overlay = build_heatmap_overlay(original_bgr, normalized_map, args.colormap, args.overlay_alpha)
        annotated = annotate_result(overlay, anomaly_score, threshold, is_anomalous)

        out_path = heatmap_dir / f"{image_path.stem}_heatmap.png"
        cv2.imwrite(str(out_path), annotated)

        normalized_score = normalize_global(anomaly_score, stats["score_min"], stats["score_max"])
        per_image_records.append({
            "filename": image_path.name,
            "raw_anomaly_score": round(anomaly_score, 6),
            "normalized_anomaly_score": round(float(normalized_score), 6),
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

    # CLI overrides for preprocessing metadata the model may not embed (notably the
    # SK-RD4AD dynamic crop, which the anomaly_export contract deliberately keeps out
    # of the graph). Only non-None entries win over the model's own metadata.
    preproc_overrides = {}
    if args.dynamic_crop != "auto":
        preproc_overrides["dynamic_crop"] = "true" if args.dynamic_crop == "on" else "false"
    if args.dynamic_crop_bg_threshold is not None:
        preproc_overrides["dynamic_crop_bg_threshold"] = str(args.dynamic_crop_bg_threshold)
    if args.dynamic_crop_padding is not None:
        preproc_overrides["dynamic_crop_padding"] = str(args.dynamic_crop_padding)

    engine = AnomalyInferenceEngine(args.model, providers, preproc_overrides=preproc_overrides)

    # INT8 safety guard. Some architectures declare quantization_safe=false in their
    # metadata (notably PatchCore: INT8-quantizing the nearest-neighbour distance
    # nodes destroys the distance ranking, silently producing meaningless scores that
    # still look plausible). Refuse before building the TensorRT INT8 engine - a
    # garbage benchmark is worse than a clear stop. Checked here (not in
    # build_providers) because it needs the model's embedded metadata, and the TRT
    # INT8 engine is only built lazily on the first inference run, still ahead of us.
    if args.precision == "int8" and str(engine.metadata.get("quantization_safe", "")).lower() == "false":
        die(f"This model declares quantization_safe=false (architecture "
            f"{engine.metadata.get('architecture', 'unknown')}): INT8 quantization would "
            f"corrupt its scores (e.g. PatchCore's nearest-neighbour distance ranking). "
            f"Re-run with --precision fp16 or fp32, or export a variant with the NN "
            f"nodes excluded from INT8 quantization.")

    image_paths = list_images(args.input_dir, args.extension)

    runtime_config = resolve_runtime_config(engine.metadata, args)

    heatmap_dir = ensure_dir(f"{args.output_dir}/heatmaps")
    cache_dir = ensure_dir(f"{args.output_dir}/.raw_cache")

    stats = collect_raw_outputs(engine, image_paths, cache_dir, args, runtime_config)
    log(f"Folder-wide anomaly map range: [{stats['map_min']:.4f}, {stats['map_max']:.4f}]")
    log(f"Folder-wide anomaly score range: [{stats['score_min']:.4f}, {stats['score_max']:.4f}]")

    # Threshold resolution order: explicit flag > calibration embedded in the
    # model (written by calibrate_threshold.py --embed) > uncalibrated fallback.
    threshold_calibrated = True
    if args.threshold is not None:
        threshold = args.threshold
        threshold_source = "user-provided ABSOLUTE raw-score threshold"
    elif "calibrated_threshold" in engine.metadata:
        threshold = float(engine.metadata["calibrated_threshold"])
        threshold_source = (f"embedded calibration "
                            f"({engine.metadata.get('calibration_info', 'no info')})")
    elif (str(engine.metadata.get("calibrated", "")).lower() == "true"
          and "image_threshold_raw" in engine.metadata):
        # anomaly_export contract: the RAW image-level threshold lives in
        # image_threshold_raw (raw score units, directly comparable to the graph's
        # anomaly_score). image_threshold_norm is the [0,1] version, not used here
        # because this runtime thresholds on the RAW score.
        threshold = float(engine.metadata["image_threshold_raw"])
        threshold_source = (f"embedded anomaly_export calibration (image_threshold_raw, "
                            f"method={engine.metadata.get('calibration_method', '?')}, "
                            f"dataset={engine.metadata.get('calibration_dataset', '?')})")
    else:
        # No sensible fixed default exists in raw-score units (they are unbounded
        # and model-specific), so fall back to the midpoint of the folder's raw
        # score range purely so the run produces *something* — and warn loudly
        # that verdicts are not trustworthy without a calibrated threshold.
        threshold = 0.5 * (stats["score_min"] + stats["score_max"])
        threshold_calibrated = False
        threshold_source = "NON-calibrated fallback (folder raw-score midpoint)"
        log("WARNING: no --threshold given and no calibration embedded in the model. "
            "Using the folder raw-score midpoint as a rough fallback - verdicts are "
            "NOT reliable. Calibrate once with:  python calibrate_threshold.py "
            "--model <model.onnx> --good_dir <good images> --embed")
    log(f"Using threshold: {threshold:.6f} ({threshold_source})")

    # Display mode: 'auto' picks the clean eval.py-style threshold-centric view
    # when a real (calibrated/user) threshold exists, per-image min-max otherwise
    # (threshold-centric coloring around an arbitrary fallback would be misleading).
    if args.normalize == "auto":
        display_mode = "threshold" if threshold_calibrated else "per_image"
    else:
        display_mode = args.normalize

    per_image_records = render_heatmaps(engine, image_paths, cache_dir, heatmap_dir,
                                        stats, threshold, args, display_mode)
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
        "architecture": runtime_config.architecture,
        "score_source": runtime_config.score_source,
        "blur": ("baked_in_graph" if runtime_config.blur_in_graph
                 else [runtime_config.blur_kernel_size, runtime_config.blur_sigma] if not args.no_blur
                 else None),
        "architecture_config_verified": runtime_config.verified,
        "num_images": num_images,
        "num_anomalous": num_anomalous,
        "num_normal": num_images - num_anomalous,
        "heatmap_display_normalization": display_mode,
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
