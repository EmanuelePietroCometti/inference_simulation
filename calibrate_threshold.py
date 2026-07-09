"""
calibrate_threshold.py — Compute the absolute anomaly-score threshold for an
exported ONNX model, through the EXACT same pipeline the runtime uses
(dynamic crop, resize, blur, score source — all auto-configured from the
model's embedded metadata), and optionally embed it into the .onnx file.

Why this exists
----------------
SK-RD4AD's training saves only weights (no threshold), and its eval.py computes
'best_threshold_raw' WITHOUT the dynamic object crop that training and this
runtime apply — so that value lives in a different score distribution and does
not transfer. The only threshold that is valid for production is one computed
through the identical inference pipeline. This script does that, and with
--embed writes it into the model's metadata as 'calibrated_threshold', which
main.py then picks up automatically (no --threshold flag needed at inference).

Usage
-----
    # good-only calibration (percentile of good scores):
    python calibrate_threshold.py --model model.onnx --good_dir imgs/good --embed

    # with a defect folder too (reports separation and suggests the midpoint):
    python calibrate_threshold.py --model model.onnx --good_dir imgs/good \
        --defect_dir imgs/nok --embed
"""

import argparse

import numpy as np

from src.utils import log, die, list_images
from src.provider_setup import build_providers
from src.inference_engine import AnomalyInferenceEngine
from src.model_config import resolve_runtime_config
from src.postprocessing import apply_training_blur


def compute_scores(engine, cfg, paths) -> np.ndarray:
    """Score each image exactly like main.py's collect_raw_outputs does."""
    scores = []
    for p in paths:
        maps, raw_scores, _ = engine.run_batch(engine.preprocess(str(p)))
        amap = apply_training_blur(maps[0, 0], cfg.blur_kernel_size, cfg.blur_sigma)
        if cfg.score_source == "map_max_blurred":
            scores.append(float(amap.max()))
        else:
            scores.append(float(raw_scores[0]))
    return np.array(scores, dtype=np.float64)


def embed_threshold(onnx_path: str, threshold: float, info: str) -> None:
    """Write/overwrite 'calibrated_threshold' (+ provenance) in the model metadata."""
    import onnx
    m = onnx.load(onnx_path)
    updates = {"calibrated_threshold": f"{threshold:.6f}", "calibration_info": info}
    kept = [e for e in m.metadata_props if e.key not in updates]
    del m.metadata_props[:]
    m.metadata_props.extend(kept)
    for k, v in updates.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(m, onnx_path)
    log(f"Embedded calibrated_threshold={threshold:.6f} into {onnx_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to the exported .onnx model")
    p.add_argument("--good_dir", required=True,
                   help="Folder of KNOWN-GOOD images (validation set, not training data)")
    p.add_argument("--defect_dir", default=None,
                   help="Optional folder of known-defect images, to measure separation")
    p.add_argument("--extension", default=".png")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "tensorrt"])
    p.add_argument("--percentile", type=float, default=99.0,
                   help="Percentile of GOOD scores used as threshold (default 99). "
                        "With --defect_dir, the midpoint between good-P<percentile> and "
                        "the defect minimum is suggested instead when they are separable.")
    p.add_argument("--embed", action="store_true",
                   help="Write the threshold into the .onnx metadata so the runtime "
                        "uses it automatically.")
    args = p.parse_args()

    providers = build_providers(device=args.device, precision="fp32",
                                engine_cache_dir="./trt_engines", calibration_table=None)
    engine = AnomalyInferenceEngine(args.model, providers)

    from types import SimpleNamespace
    cfg = resolve_runtime_config(
        engine.metadata,
        SimpleNamespace(score_source="auto", blur_kernel_size=None, blur_sigma=None),
    )

    good_paths = list_images(args.good_dir, args.extension)
    log(f"Scoring {len(good_paths)} GOOD images...")
    good = compute_scores(engine, cfg, good_paths)
    log(f"GOOD scores  : min={good.min():.4f}  median={np.median(good):.4f}  "
        f"p{args.percentile:g}={np.percentile(good, args.percentile):.4f}  max={good.max():.4f}")

    threshold = float(np.percentile(good, args.percentile))
    info = f"p{args.percentile:g} of {len(good)} good images"

    if args.defect_dir:
        defect_paths = list_images(args.defect_dir, args.extension)
        log(f"Scoring {len(defect_paths)} DEFECT images...")
        defect = compute_scores(engine, cfg, defect_paths)
        log(f"DEFECT scores: min={defect.min():.4f}  median={np.median(defect):.4f}  "
            f"max={defect.max():.4f}")

        if defect.min() > threshold:
            midpoint = 0.5 * (threshold + float(defect.min()))
            log(f"Classes are separable at p{args.percentile:g}: using midpoint "
                f"{midpoint:.4f} between good-p{args.percentile:g} ({threshold:.4f}) "
                f"and defect-min ({defect.min():.4f}).")
            threshold = midpoint
            info = (f"midpoint(good-p{args.percentile:g}, defect-min), "
                    f"{len(good)} good / {len(defect)} defect images")
        else:
            overlap = float((defect <= threshold).mean() * 100)
            log(f"WARNING: {overlap:.0f}% of defect images score BELOW the good-p"
                f"{args.percentile:g} threshold - the classes overlap. Keeping the "
                f"good-percentile threshold; expect misses. Consider more training "
                f"or a lower --percentile (trades false alarms for fewer misses).")

    log(f"==> threshold = {threshold:.6f}  ({info})")
    if args.embed:
        embed_threshold(args.model, threshold, info)
        log("Done: the runtime will now pick this threshold up automatically.")
    else:
        log(f"Run inference with:  --threshold {threshold:.6f}   "
            f"(or re-run with --embed to store it in the model)")


if __name__ == "__main__":
    main()
