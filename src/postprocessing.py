"""
Converts raw anomaly maps into visualizable heatmaps, matching the training
visualization pipeline (eval.py + common/visualizer.py from the training repo).

Two key differences from a naive per-image implementation:

1. Normalization is computed GLOBALLY across the whole processed folder, exactly
   like eval.py does across the whole test set:

       anomaly_map = (anomaly_map - global_min) / (global_max - global_min)

   Per-image min-max normalization (stretching each image independently) produces
   misleading heatmaps: a perfectly normal image would always show a "hot" region
   simply because some pixel has to be the local maximum. Folder-wide statistics
   keep images comparable to each other, exactly like training.

2. A Gaussian blur (kernel_size=25, sigma=4) is re-applied to the raw anomaly map.
   The ONNX export disables this blur inside the graph (see export_onnx.py,
   "GaussianBlur disabled for export... apply it in post-processing"), so it must
   be replicated here to match model/supersimplenet.py's AnomalyMapGenerator,
   otherwise heatmaps look noisy/blocky instead of the smooth blobs seen in
   training visualizations.
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
