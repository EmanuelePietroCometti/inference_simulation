"""
Turns a raw anomaly map into a heatmap overlay, replicating the C++ engine's
rendering (ONNX_inference/AnomalyEngine.cpp, Infer()):

  1. Min-max normalize the RAW map with the embedded ``map_min_raw``/``map_max_raw``
     bounds (the C++ ``alpha``/``beta`` convertTo), NOT per-image or folder-wide
     statistics. This keeps every heatmap in the same calibrated units the model
     was validated in, so brightness is comparable across images and matches
     production.
  2. Color-map it (JET by default) and alpha-blend over the original frame.
  3. If a ``pixel_threshold_raw`` is calibrated, mask the pixels of the RAW map at
     or above it and draw the defect contours, exactly like the C++ engine.

The image-level OK/ANOMALY verdict is decided elsewhere on the raw score against
``image_threshold_raw`` — the display normalization here never affects it.
"""

import numpy as np
import cv2

COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "HOT": cv2.COLORMAP_HOT,
}

# Color of the defect contours drawn from the pixel threshold (BGR). The C++
# engine draws them in red (cv::Scalar(255,0,0) on an RGB overlay); on our BGR
# overlays red is (0,0,255).
CONTOUR_COLOR_BGR = (0, 0, 255)


def normalize_contract(anomaly_map: np.ndarray, map_min_raw: float, map_max_raw: float) -> np.ndarray:
    """Min-max normalize the raw map to [0,1] using the embedded calibration
    bounds — the Python equivalent of the C++ linear ``alpha*x + beta`` scaling."""
    denom = map_max_raw - map_min_raw
    if denom <= 1e-12:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    normalized = (np.asarray(anomaly_map, dtype=np.float32) - map_min_raw) / denom
    return np.clip(normalized, 0.0, 1.0)


def map_to_uint8(normalized_map: np.ndarray) -> np.ndarray:
    """Convert an already [0,1]-normalized map to a uint8 image for colormap rendering."""
    clipped = np.clip(normalized_map, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def build_heatmap_overlay(original_bgr: np.ndarray, normalized_map: np.ndarray,
                          colormap_name: str, alpha: float) -> np.ndarray:
    """Overlay a color-mapped, already-normalized anomaly map on the original image."""
    map_uint8 = map_to_uint8(normalized_map)
    if map_uint8.shape[:2] != original_bgr.shape[:2]:
        map_uint8 = cv2.resize(map_uint8, (original_bgr.shape[1], original_bgr.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

    colormap = COLORMAPS.get(colormap_name.upper(), cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(map_uint8, colormap)
    return cv2.addWeighted(heatmap_bgr, alpha, original_bgr, 1 - alpha, 0)


def draw_pixel_contours(overlay_bgr: np.ndarray, anomaly_map: np.ndarray,
                        pixel_threshold_raw: float) -> np.ndarray:
    """Mask the RAW map at ``pixel_threshold_raw`` and draw the defect contours on
    the overlay, replicating the C++ engine (findContours + drawContours). The
    mask is computed on the raw map and resized NEAREST to the overlay size so the
    boundary is not interpolated."""
    mask = (np.asarray(anomaly_map, dtype=np.float32) >= pixel_threshold_raw).astype(np.uint8) * 255
    if mask.shape[:2] != overlay_bgr.shape[:2]:
        mask = cv2.resize(mask, (overlay_bgr.shape[1], overlay_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_bgr, contours, -1, CONTOUR_COLOR_BGR, 2)
    return overlay_bgr


def annotate_result(image: np.ndarray, score: float, threshold: float, is_anomalous: bool) -> np.ndarray:
    """Draw the raw anomaly score, threshold, and OK/ANOMALY verdict label."""
    annotated = image.copy()
    label = "ANOMALY" if is_anomalous else "OK"
    color = (0, 0, 255) if is_anomalous else (0, 200, 0)

    text_lines = [f"score: {score:.4f}", f"threshold: {threshold:.4f}", label]
    for i, line in enumerate(text_lines):
        y = 25 + i * 25
        cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

    return annotated
