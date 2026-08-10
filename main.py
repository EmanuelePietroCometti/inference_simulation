"""
Anomaly-detection ONNX inference + benchmark over a folder of images.

This is the validation/benchmark counterpart of the C++ production engine
(ONNX_inference/AnomalyEngine.cpp). It is *contract-driven*: every decision comes
from the calibration metadata embedded in the .onnx (src/model_config.py), so a
model that behaves correctly here behaves identically in the C++ runtime.

For every image it:
  - runs the ONNX model (FP32 / FP16 / INT8 via TensorRT, or CUDA / CPU),
  - classifies it by comparing the graph's RAW anomaly_score against the embedded
    image_threshold_raw (same verdict as the C++ engine: >= threshold -> ANOMALY),
  - renders a heatmap overlay normalized with the embedded map_min_raw/map_max_raw
    bounds, with defect contours drawn at pixel_threshold_raw (same as C++);
and then benchmarks inference throughput at batch size 1 and 17 (or any sizes
passed via --batch_sizes) at the requested precision.

All logic lives in the src/ package; this file only wires the modules together.
"""

import cv2

from src.cli import parse_args
from src.utils import log, die, list_images, ensure_dir
from src.provider_setup import build_session_plans
from src.onnx_introspect import inspect_input_signature
from src.inference_engine import AnomalyInferenceEngine
from src.postprocessing import (
    normalize_contract,
    build_heatmap_overlay,
    draw_pixel_contours,
    annotate_result,
)
from src.batch_benchmark import run_batch_comparison
from src.environment import collect_environment
from src.ep_partition import profile_ep_partition
from src.results_writer import write_benchmark_results


def _normalized_score(raw_score: float, contract) -> float | None:
    """Report the raw score as a [0,1] fraction of the calibrated score range, when
    those bounds are embedded. Display only — the verdict uses the raw score."""
    if contract.score_min_raw is None or contract.score_max_raw is None:
        return None
    denom = contract.score_max_raw - contract.score_min_raw
    if denom <= 1e-12:
        return None
    return max(0.0, min(1.0, (raw_score - contract.score_min_raw) / denom))


def run_inference_pass(engine, image_paths, heatmap_dir, threshold, args):
    """Single pass: infer, threshold on the raw score, render the heatmap overlay,
    and collect per-image records. Mirrors the C++ engine's per-frame Infer()."""
    log(f"Running inference on {len(image_paths)} images (contract-driven pass)...")
    contract = engine.contract
    records = []
    total_time = 0.0
    skipped = []

    for image_path in image_paths:
        # Per-image guard: one unreadable/corrupt file must not abort the whole run
        # (and lose every heatmap already produced). Skip it, log it, keep going.
        try:
            batch_tensor = engine.preprocess(str(image_path))
            anomaly_maps, anomaly_scores, elapsed = engine.run_batch(batch_tensor)
            total_time += elapsed

            anomaly_map = anomaly_maps[0, 0]
            raw_score = float(anomaly_scores[0])
            is_anomalous = raw_score >= threshold  # same comparison as AnomalyEngine.cpp

            # Display: min-max with the embedded calibration bounds (C++ alpha/beta).
            normalized_map = normalize_contract(anomaly_map, contract.map_min_raw, contract.map_max_raw)
            original_bgr = engine.preprocessor.load_original_bgr(str(image_path))
            overlay = build_heatmap_overlay(original_bgr, normalized_map, args.colormap, args.overlay_alpha)
            if contract.pixel_threshold_raw is not None:
                overlay = draw_pixel_contours(overlay, anomaly_map, contract.pixel_threshold_raw)
            annotated = annotate_result(overlay, raw_score, threshold, is_anomalous)

            out_path = heatmap_dir / f"{image_path.stem}_heatmap.png"
            cv2.imwrite(str(out_path), annotated)

            norm_score = _normalized_score(raw_score, contract)
            records.append({
                "filename": image_path.name,
                "raw_anomaly_score": round(raw_score, 6),
                "normalized_anomaly_score": (round(norm_score, 6) if norm_score is not None else ""),
                "is_anomalous": is_anomalous,
            })
        except Exception as exc:  # noqa: BLE001 - never abort the whole pass on one image
            skipped.append(image_path.name)
            log(f"WARNING: skipping '{image_path.name}' (could not process: {exc}).")

    if skipped:
        log(f"Processed {len(records)}/{len(image_paths)} images; skipped {len(skipped)}: {skipped}")

    return records, total_time


def main() -> None:
    args = parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    # Inspect the ONNX FIRST so the TensorRT profile is built with the model's REAL
    # input name and C/H/W — not the guessed "image"/3x256x256 defaults. A wrong
    # name/shape would make TensorRT ignore the profile (or rebuild per-shape).
    input_name, in_c, in_h, in_w = inspect_input_signature(args.model)

    def make_plans(plan_batches, opt_batch):
        """Build session plans for a given batch set / opt batch (keeps the many
        shared kwargs in one place)."""
        return build_session_plans(
            device=args.device,
            precision=args.precision,
            engine_cache_dir=args.engine_cache_dir,
            calibration_table=args.calibration_table,
            batch_sizes=plan_batches,
            input_name=input_name,
            input_channels=in_c,
            input_height=in_h,
            input_width=in_w,
            gpu_mem_limit=args.gpu_mem_limit,
            trt_workspace_gb=args.trt_workspace_gb,
            trt_opt_batch=opt_batch,
        )

    # Correctness/classification session: a static batch-1 engine (the pass runs one
    # image at a time), deterministic kernels for reproducible scores.
    correctness_plans = make_plans([1], opt_batch=None)
    engine = AnomalyInferenceEngine(args.model, correctness_plans, deterministic=True)
    contract = engine.contract

    # INT8 safety guard: PatchCore (and any model declaring quantization_safe=false)
    # loses its score ranking under INT8. Refuse before TensorRT lazily builds the
    # INT8 engine — a garbage benchmark is worse than a clear stop.
    if args.precision == "int8" and not contract.quantization_safe:
        die(f"This model declares quantization_safe=false (architecture "
            f"{contract.architecture}): INT8 quantization would corrupt its scores "
            f"(e.g. PatchCore's nearest-neighbour distance ranking). Re-run with "
            f"--precision fp16 or fp32.")

    image_paths = list_images(args.input_dir, args.extension, args.expected_count)

    # Threshold: explicit CLI override > embedded image_threshold_raw (contract).
    if args.threshold is not None:
        threshold = args.threshold
        threshold_source = "user-provided ABSOLUTE raw-score threshold"
    else:
        threshold = contract.image_threshold_raw
        threshold_source = "embedded image_threshold_raw (calibration contract)"
    log(f"Using image threshold: {threshold:.6f} ({threshold_source})")

    heatmap_dir = ensure_dir(f"{args.output_dir}/heatmaps")

    records, total_time = run_inference_pass(engine, image_paths, heatmap_dir, threshold, args)
    log(f"Heatmaps saved to: {heatmap_dir}")

    num_images = len(records)
    num_anomalous = sum(r["is_anomalous"] for r in records)

    # Optional per-EP partition report (TensorRT/CUDA/CPU node placement) — reveals
    # partial fallback that get_providers() hides. Uses a dedicated profiled session.
    ep_partition = None
    if args.profile_ep and records:
        sample = engine.preprocess(str(image_paths[0]))
        ep_partition = profile_ep_partition(
            args.model, correctness_plans[0]["providers"], engine.input_name, sample,
            [engine.anomaly_map_name, engine.anomaly_score_name])

    # Throughput benchmark. Two modes:
    #   * default (per-batch): each batch is timed on its OWN static-shape engine
    #     (min=opt=max=batch), so every number is that batch's best achievable speed
    #     with no opt-shape artifact and nothing to tune. Engines are cached on disk
    #     (namespaced per batch) and released between batches to free VRAM.
    #   * shared (only if --trt_opt_batch is given): one engine over the whole range
    #     (production parity); batches != opt run with non-optimal tactics.
    # Both run non-deterministic by default so the speed number is not penalized by
    # the deterministic-kernel setting the correctness pass needs.
    bench_deterministic = args.throughput_deterministic
    if args.trt_opt_batch is not None:
        opt_mode = "shared"
        log(f"Throughput benchmark: SHARED engine over batches {batch_sizes}, "
            f"opt_batch={args.trt_opt_batch} (production-parity mode).")
        shared_engine = AnomalyInferenceEngine(
            args.model, make_plans(batch_sizes, opt_batch=args.trt_opt_batch),
            deterministic=bench_deterministic)
        _bmin, _bmax = min(batch_sizes), max(batch_sizes)
        shared_opt = max(_bmin, min(args.trt_opt_batch, _bmax))

        def engine_factory(_batch):
            return shared_engine
        dispose = False
    else:
        opt_mode = "per_batch"
        shared_opt = None
        log(f"Throughput benchmark: PER-BATCH static engines for {batch_sizes} "
            f"(each batch optimized for itself; no target to specify).")

        def engine_factory(batch):
            log(f"[batch={batch}] Building a static engine optimized for batch={batch} "
                f"(deterministic={bench_deterministic})...")
            return AnomalyInferenceEngine(
                args.model, make_plans([batch], opt_batch=batch),
                deterministic=bench_deterministic)
        dispose = True

    log(f"Running throughput benchmark for batch sizes: {batch_sizes} (precision={args.precision})")
    batch_comparison = run_batch_comparison(
        engine_factory, image_paths, batch_sizes,
        warmup_iters=args.warmup_iters, timed_iters=args.timed_iters,
        dispose_between_batches=dispose, opt_mode=opt_mode, shared_opt_batch=shared_opt,
    )

    environment = collect_environment(active_providers=engine.session.get_providers())

    summary = {
        "model": args.model,
        "device": args.device,
        "precision": args.precision,
        "active_providers": engine.session.get_providers(),
        "architecture": contract.architecture,
        "calibrated": contract.calibrated,
        "quantization_safe": contract.quantization_safe,
        "num_images_processed": num_images,
        "num_images_found": len(image_paths),
        "num_anomalous": num_anomalous,
        "num_normal": num_images - num_anomalous,
        "image_threshold_raw": round(threshold, 6),
        "threshold_source": threshold_source,
        "pixel_threshold_raw": (None if contract.pixel_threshold_raw is None
                                else round(contract.pixel_threshold_raw, 6)),
        "map_normalization_range": [round(contract.map_min_raw, 6), round(contract.map_max_raw, 6)],
        "score_normalization_range": (
            [round(contract.score_min_raw, 6), round(contract.score_max_raw, 6)]
            if contract.score_min_raw is not None and contract.score_max_raw is not None else None),
        # Batch-1, single pass over DISTINCT images, no warmup — a different quantity
        # from the batch benchmark below (repeated images, warmed up). Not comparable.
        "pass_inference_total_time_sec": round(total_time, 3),
        "throughput_benchmark_deterministic": args.throughput_deterministic,
        "environment": environment,
        "ep_partition": ep_partition,
        "batch_throughput_comparison": batch_comparison,
    }
    write_benchmark_results(args.output_dir, summary, records)

    log("Done.")


if __name__ == "__main__":
    main()
