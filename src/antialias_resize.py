"""
Antialiased bilinear resize — the single, self-contained reference implementation.

This is the STANDARD separable triangle-filter (bilinear) resampling with
antialiasing: for a downscale the filter support grows with the scale factor, so
high frequencies are averaged out instead of aliased. It is the same math that
torchvision ``Resize(antialias=True)`` (the training pipeline) and PIL's
``Image.resize(BILINEAR)`` implement — but it depends on NEITHER: it is written
out explicitly here so the Python reference and the C++ production engine
(ONNX_inference/AnomalyEngine.cpp) share one definition and can never drift, and
so nothing breaks if a third-party library changes its internals.

IMPORTANT: the C++ engine mirrors this file bit-for-bit. Any change here must be
mirrored in AnomalyEngine.cpp (precompute_coeffs + the two separable passes,
double accumulation, round-half-up, uint8 clip between passes).
"""

from __future__ import annotations

import numpy as np


def precompute_coeffs(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Triangle-filter resample coefficients for one axis (Pillow's algorithm).

    Returns (bounds, weights, ksize):
      - bounds[o] = first input index contributing to output pixel o,
      - weights[o, 0:ksize] = the (normalized) tap weights,
      - ksize = max taps per output pixel (rows are zero-padded to ksize).
    """
    support = 1.0                                   # bilinear (triangle) half-width
    scale = in_size / out_size
    filterscale = scale if scale >= 1.0 else 1.0    # antialias: widen on downscale
    support *= filterscale
    ss = 1.0 / filterscale

    ksize = int(np.ceil(support)) * 2 + 1
    bounds = np.zeros(out_size, dtype=np.int64)
    weights = np.zeros((out_size, ksize), dtype=np.float64)

    for o in range(out_size):
        center = (o + 0.5) * scale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        n = xmax - xmin
        total = 0.0
        for t in range(n):
            w = 1.0 - abs((xmin + t - center + 0.5) * ss)   # bilinear_filter
            if w < 0.0:
                w = 0.0
            weights[o, t] = w
            total += w
        if total > 0.0:
            weights[o, :n] /= total
        bounds[o] = xmin
    return bounds, weights, ksize


def _round_clip_u8(a: np.ndarray) -> np.ndarray:
    # round half up (pixels are non-negative), clip to [0, 255]. Matches C++
    # (uint8) clamp(floor(v + 0.5)).
    return np.clip(np.floor(a + 0.5), 0.0, 255.0).astype(np.uint8)


def resize_antialias(img_u8: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a uint8 HxWxC image to (out_h, out_w) with the antialiased triangle
    filter. Horizontal pass then vertical pass, with a uint8 round/clip between
    them (PIL order/behaviour), so the calibrated thresholds stay valid."""
    in_h, in_w = img_u8.shape[:2]
    c = 1 if img_u8.ndim == 2 else img_u8.shape[2]
    x = img_u8.reshape(in_h, in_w, c).astype(np.float64)

    # horizontal: [in_h, in_w, c] -> [in_h, out_w, c]
    hb, hw, hk = precompute_coeffs(in_w, out_w)
    hp = np.zeros((in_h, out_w, c), dtype=np.float64)
    for o in range(out_w):
        s = hb[o]
        avail = min(hk, in_w - s)                    # clamp taps at the right edge
        hp[:, o, :] = np.tensordot(x[:, s:s + avail, :], hw[o, :avail], axes=([1], [0]))
    hp = _round_clip_u8(hp).astype(np.float64)

    # vertical: [in_h, out_w, c] -> [out_h, out_w, c]
    vb, vw, vk = precompute_coeffs(in_h, out_h)
    vp = np.zeros((out_h, out_w, c), dtype=np.float64)
    for o in range(out_h):
        s = vb[o]
        avail = min(vk, in_h - s)                    # clamp taps at the bottom edge
        vp[o, :, :] = np.tensordot(vw[o, :avail], hp[s:s + avail, :, :], axes=([0], [0]))
    out = _round_clip_u8(vp)
    return out.reshape(out_h, out_w) if img_u8.ndim == 2 else out
