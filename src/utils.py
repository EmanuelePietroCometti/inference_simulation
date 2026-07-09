"""Small filesystem and logging helpers shared across the inference pipeline."""

import sys
import time
from pathlib import Path


def log(message: str) -> None:
    """Print a timestamped log message to stdout."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def list_images(folder: str, extension: str) -> list[Path]:
    """Return a sorted list of image paths with the given extension inside folder (recursive)."""
    folder_path = Path(folder)
    if not folder_path.exists():
        die(f"Input folder does not exist: {folder_path}")

    paths = sorted(folder_path.rglob(f"*{extension}"))
    if not paths:
        die(f"No '{extension}' images found in {folder_path}")

    log(f"Found {len(paths)} images in {folder_path}")
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
