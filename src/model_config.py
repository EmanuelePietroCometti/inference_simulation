"""
Reads the model's embedded calibration contract, exactly like the C++
``AnomalyEngine::LoadContractMetadata`` (ONNX_inference/AnomalyEngine.cpp).

This runtime is *contract-driven*: every model exported by the anomaly_export
package embeds, as ONNX ``metadata_props``, the RAW calibration bounds and
thresholds the pipeline must use. The host applies NO architecture-specific
logic (no host-side blur, no dynamic crop, no score-source guessing): the graph
already emits the final ``anomaly_score`` (max of the in-graph blurred map, a
classification-head logit, or a reweighted NN distance — the graph decides) and
the final ``anomaly_map``. The host only:

  - thresholds the raw ``anomaly_score`` against ``image_threshold_raw`` (image verdict),
  - min-max normalizes the raw ``anomaly_map`` with ``map_min_raw``/``map_max_raw`` for display,
  - masks pixels above ``pixel_threshold_raw`` for the defect contours.

These are the exact keys and the exact behaviour the C++ engine relies on, so a
model that benchmarks correctly here will behave identically in production.

Missing bounds are a hard error (same as the C++ engine), never a guess: a
threshold is only meaningful in the units it was calibrated in.
"""

from dataclasses import dataclass

from src.utils import log, die


@dataclass
class ModelContract:
    """The calibration contract embedded in the ONNX model's metadata."""
    architecture: str
    # Image-level verdict: raw anomaly_score >= image_threshold_raw -> anomalous.
    image_threshold_raw: float
    # Display normalization bounds for the raw anomaly_map (min-max -> [0,1]).
    map_min_raw: float
    map_max_raw: float
    # Pixel-level defect mask: raw map >= pixel_threshold_raw. Optional.
    pixel_threshold_raw: float | None
    # Raw score range across the calibration set (used to report a normalized score).
    score_min_raw: float | None
    score_max_raw: float | None
    # True when the graph applies /255 + mean/std itself (host feeds float32 [0,1]).
    normalize_inside_graph: bool
    # BGR->RGB swap expected by the graph (OpenCV decodes BGR).
    convert_bgr_to_rgb: bool
    # False when INT8 quantization would corrupt the scores (e.g. PatchCore NN distance).
    quantization_safe: bool
    calibrated: bool


def _get_float(metadata: dict, key: str) -> float | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        die(f"Metadata key '{key}' is not a valid float: {value!r}")


def load_contract(metadata: dict) -> ModelContract:
    """Parse the embedded contract, mirroring the C++ LoadContractMetadata checks."""
    # Export-pipeline test artifacts contain RANDOM (untrained) weights: their
    # maps saturate near-uniformly and every score collapses to ~1, which looks
    # exactly like a subtle inference bug. Refuse instead of wasting a debug session.
    if metadata.get("weights_source") == "random_self_test":
        die(
            "This ONNX file was exported with --self_test and contains RANDOM "
            "(untrained) weights - it cannot produce meaningful anomaly maps or "
            "scores. Re-export from a trained checkpoint WITHOUT --self_test."
        )
    if "weights_source" in metadata:
        log(f"Model weights source: {metadata['weights_source']}")

    architecture = metadata.get("architecture") or metadata.get("model_name", "unknown")

    if str(metadata.get("calibrated", "")).lower() != "true":
        log("WARNING: model 'calibrated' flag is not 'true'. The embedded raw bounds "
            "may be uncalibrated - verify verdicts before trusting them.")

    map_min = _get_float(metadata, "map_min_raw")
    map_max = _get_float(metadata, "map_max_raw")
    image_threshold = _get_float(metadata, "image_threshold_raw")

    # Same hard stop as the C++ engine: without these three the pipeline has no
    # calibrated units to threshold or render in.
    if map_min is None or map_max is None or image_threshold is None:
        die("Model rejected: calibration metadata is incomplete "
            "(need map_min_raw, map_max_raw, image_threshold_raw). Re-run calibration.")
    if map_max - map_min <= 0.0:
        die("Invalid calibration metadata: map_max_raw must be greater than map_min_raw.")

    pixel_threshold = _get_float(metadata, "pixel_threshold_raw")

    # Missing key defaults to bgr2rgb (all our exporter's models), matching C++.
    color_conversion = metadata.get("preproc_color_conversion", "") or ""
    convert_bgr_to_rgb = color_conversion == "" or color_conversion == "bgr2rgb"

    normalize_inside_graph = str(metadata.get("normalize_inside_graph", "")).lower() == "true"
    if not normalize_inside_graph:
        log("WARNING: normalize_inside_graph is not 'true'. Host will apply ImageNet "
            "normalization as fallback (see preprocessing.py).")

    # PatchCore declares quantization_safe=false: INT8-quantizing its NN-distance
    # nodes destroys the ranking and silently produces meaningless scores.
    quantization_safe = str(metadata.get("quantization_safe", "true")).lower() != "false"

    contract = ModelContract(
        architecture=architecture,
        image_threshold_raw=image_threshold,
        map_min_raw=map_min,
        map_max_raw=map_max,
        pixel_threshold_raw=pixel_threshold,
        score_min_raw=_get_float(metadata, "score_min_raw"),
        score_max_raw=_get_float(metadata, "score_max_raw"),
        normalize_inside_graph=normalize_inside_graph,
        convert_bgr_to_rgb=convert_bgr_to_rgb,
        quantization_safe=quantization_safe,
        calibrated=str(metadata.get("calibrated", "")).lower() == "true",
    )

    log(f"Contract loaded: architecture={architecture}, "
        f"image_threshold_raw={image_threshold:.6g}, "
        f"map_raw=[{map_min:.6g}, {map_max:.6g}], "
        f"pixel_threshold_raw={'n/a' if pixel_threshold is None else f'{pixel_threshold:.6g}'}, "
        f"normalize_inside_graph={normalize_inside_graph}, "
        f"bgr2rgb={convert_bgr_to_rgb}, quantization_safe={quantization_safe}")
    return contract
