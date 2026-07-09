"""
Temporary on-disk cache for raw (un-normalized, un-blurred) model outputs.

Folder-wide normalization (see postprocessing.py) requires knowing the min/max
across every image in the folder before any heatmap can be rendered. Rather than
either re-running inference twice or holding every anomaly map in RAM at once,
raw outputs are cached to small .npz files on disk after the first inference
pass and reloaded during the rendering pass.
"""

import shutil
from pathlib import Path

import numpy as np


def _cache_path(cache_dir: Path, image_path: Path) -> Path:
    return cache_dir / f"{image_path.stem}.npz"


def save_raw(cache_dir: Path, image_path: Path, anomaly_map: np.ndarray, anomaly_score: float) -> None:
    np.savez(_cache_path(cache_dir, image_path), anomaly_map=anomaly_map, anomaly_score=anomaly_score)


def load_raw(cache_dir: Path, image_path: Path):
    data = np.load(_cache_path(cache_dir, image_path))
    return data["anomaly_map"], float(data["anomaly_score"])


def cleanup(cache_dir: Path) -> None:
    shutil.rmtree(cache_dir, ignore_errors=True)
