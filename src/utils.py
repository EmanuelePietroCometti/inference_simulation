"""Small filesystem and logging helpers shared across the inference pipeline."""

import sys
import time
from pathlib import Path


def log(message: str) -> None:
    """Print a timestamped log message to stdout."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def list_images(folder: str, extension: str, expected_count: int | None = None) -> list[Path]:
    """Return a sorted list of image paths with the given extension inside folder (recursive).

    ``extension`` may be a comma-separated list (e.g. ".bmp,.png") so images with
    mixed extensions are not silently dropped. When ``expected_count`` is given, a
    mismatch is logged as a loud WARNING (dataset integrity check) rather than passing
    unnoticed — this is what surfaces "expected N, found M" discrepancies."""
    folder_path = Path(folder)
    if not folder_path.exists():
        die(f"Input folder does not exist: {folder_path}")

    extensions = [e.strip() for e in extension.split(",") if e.strip()]
    paths: list[Path] = []
    for ext in extensions:
        ext = ext if ext.startswith(".") else f".{ext}"
        paths.extend(folder_path.rglob(f"*{ext}"))
    paths = sorted(set(paths))
    if not paths:
        die(f"No '{extension}' images found in {folder_path}")

    log(f"Found {len(paths)} images in {folder_path} (extensions: {extensions})")
    if expected_count is not None and len(paths) != expected_count:
        log(f"WARNING: expected {expected_count} images but found {len(paths)} "
            f"(delta {len(paths) - expected_count:+d}). Check --extension and that no "
            f"files failed to match - some inputs may be silently excluded.")
    return paths


def ensure_dir(path: str) -> Path:
    """Create a directory (and parents) if it does not exist, and return it as a Path."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def die(message: str) -> None:
    """Print an error message and terminate the program."""
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(1)
