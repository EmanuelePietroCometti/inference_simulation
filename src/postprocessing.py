"""
Converts raw anomaly maps into visualizable heatmaps, matching the training
visualization pipeline (eval.py + common/visualizer.py from the training repo).

Two key differences from a naive per-image implementation:

1. Normalization for heatmap DISPLAY is min-max, either PER-IMAGE (default) or
   folder-wide (opt-in via ``--normalize folder``), never affecting the verdict:

       anomaly_map = (anomaly_map - map_min) / (map_max - map_min)

   Folder-wide statistics (like eval.py computes across the whole test set) keep
   images comparable to each other, but some raw maps (e.g. SK-RD4AD's summed
   cosine-distance) have a folder-wide range so narrow that baseline differences
   between images (lighting/texture) dominate it: every image gets pushed toward
   one end of that narrow range and every heatmap looks uniformly red/blue, even
   though each image's own local contrast (background vs defect) is meaningful.
   Per-image normalization restores that local contrast; its one downside (a
   perfectly normal, low-contrast image will still show a "hot" local maximum,
   since some pixel always has to be the max) is a display-only quirk that never
   affects classification, which uses the separate absolute raw-score threshold.

2. A Gaussian blur may be re-applied to the raw anomaly map, but ONLY for models
   whose export leaves it out of the graph (SuperSimpleNet's export_onnx.py:
   "GaussianBlur disabled for export... apply it in post-processing", kernel=25
   sigma=4; legacy SK-RD4AD contract-1.0 exports). SK-RD4AD contract-2.0 models
   bake the canonical blur INTO the graph: for those the host applies no blur at
   all (src/model_config.py resolves this from the metadata and disables it).
"""

import numpy as np
import cv2

COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "HOT": cv2.COLORMAP_HOT,
}

# Matches the fixed 0.5 cutoff used on the normalized map in the training
# visualizer (`pred_mask = anomaly_map >= 0.5`): the natural decision boundary
# once scores are normalized to [0, 1], used here as a documented fallback when
# no threshold derived from a labeled validation set is available.
DEFAULT_THRESHOLD = 0.5


def apply_training_blur(anomaly_map: np.ndarray, kernel_size: int = 25, sigma: float = 4.0) -> np.ndarray:
    """
    Reproduce model/supersimplenet.py's AnomalyMapGenerator blur
    (torchvision GaussianBlur(kernel_size=25, sigma=4)), which the ONNX export
    intentionally omits from the graph.
    """
    if kernel_size <= 1:
        return anomaly_map
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(anomaly_map.astype(np.float32), (kernel_size, kernel_size), sigmaX=sigma)


def normalize_global(value, global_min: float, global_max: float) -> np.ndarray:
    """Normalize using folder-wide min/max (same convention as eval.py's `normalize` block)."""
    denom = global_max - global_min
    if denom < 1e-8:
        return np.zeros_like(value, dtype=np.float32)
    return (np.asarray(value, dtype=np.float32) - global_min) / denom


def normalize_threshold_centric(anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
    """
    Threshold-centric display normalization, replicating SK-RD4AD eval.py's
    save_confusion_map: values below the decision threshold map to [0, 0.5]
    (cold colors), values above map to [0.5, 1.0] (warm colors), and the
    image's own max is forced to 1.0 (deep red).

    Compared to plain per-image min-max — which stretches the background noise
    floor across the whole colormap, producing speckly blue/yellow texture
    everywhere — this pins the decision boundary at the colormap midpoint, so
    background stays uniformly cold and only truly above-threshold regions
    render warm. This is what makes the training-repo visualizations look
    "clean". Requires a calibrated absolute threshold (score and map share the
    same units: the score IS the max of this map for SK-RD4AD).
    """
    amap = np.asarray(anomaly_map, dtype=np.float32)
    lo, hi = float(amap.min()), float(amap.max())
    out = np.zeros_like(amap)

    below = amap < threshold
    if threshold > lo:
        out[below] = 0.5 * (amap[below] - lo) / (threshold - lo)

    above = ~below
    if hi > threshold:
        out[above] = 0.5 + 0.5 * (amap[above] - threshold) / (hi - threshold)
    else:
        out[above] = 0.5  # pixel exactly at threshold and also the image max

    return np.clip(out, 0.0, 1.0)


def map_to_uint8(normalized_map: np.ndarray) -> np.ndarray:
    """Convert an already [0, 1]-normalized map to a uint8 image for colormap rendering."""
    clipped = np.clip(normalized_map, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def build_heatmap_overlay(original_bgr: np.ndarray, normalized_map: np.ndarray,
                           colormap_name: str, alpha: float) -> np.ndarray:
    """Overlay a color-mapped, already-globally-normalized anomaly map on the original image."""
    map_uint8 = map_to_uint8(normalized_map)
    if map_uint8.shape[:2] != original_bgr.shape[:2]:
        map_uint8 = cv2.resize(map_uint8, (original_bgr.shape[1], original_bgr.shape[0]))

    colormap = COLORMAPS.get(colormap_name.upper(), cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(map_uint8, colormap)
    return cv2.addWeighted(heatmap_bgr, alpha, original_bgr, 1 - alpha, 0)


def annotate_result(image: np.ndarray, score: float, threshold: float, is_anomalous: bool) -> np.ndarray:
    """Draw the (folder-normalized) anomaly score, threshold, and verdict label."""
    annotated = image.copy()
    label = "ANOMALY" if is_anomalous else "OK"
    color = (0, 0, 255) if is_anomalous else (0, 200, 0)

    text_lines = [f"score: {score:.4f}", f"threshold: {threshold:.4f}", label]
    for i, line in enumerate(text_lines):
        y = 25 + i * 25
        cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

    return annotated
