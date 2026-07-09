"""
Pure inference-throughput benchmark: compares batch size 1 vs batch size 17
(or any sizes requested), reusing images from the input folder. No heatmaps or
thresholds are involved here, this only measures raw inference speed.
"""

import time
import numpy as np

from src.utils import log


def _make_batch(engine, image_paths: list, batch_size: int) -> np.ndarray:
    """
    Preprocess `batch_size` images and stack them into a single batch tensor.
    If the folder has fewer images than batch_size, the last image is repeated
    to fill the batch (timing purposes only, never used for real results).
    """
    tensors = [engine.preprocess(str(p)) for p in image_paths[:batch_size]]
    while len(tensors) < batch_size:
        tensors.append(tensors[-1])
    return np.concatenate(tensors, axis=0)


def run_batch_benchmark(engine, image_paths: list, batch_size: int,
                         warmup_iters: int, timed_iters: int) -> dict:
    """Run a warm-up + timed loop at a fixed batch size and return timing stats."""
    batch_tensor = _make_batch(engine, image_paths, batch_size)

    log(f"[batch={batch_size}] Warming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        engine.run_batch(batch_tensor)

    log(f"[batch={batch_size}] Timing ({timed_iters} iterations)...")
    start = time.perf_counter()
    for _ in range(timed_iters):
        engine.run_batch(batch_tensor)
    total_time = time.perf_counter() - start

    avg_batch_time = total_time / timed_iters
    avg_time_per_image = avg_batch_time / batch_size

    return {
        "batch_size": batch_size,
        "timed_iterations": timed_iters,
        "avg_time_per_batch_ms": round(avg_batch_time * 1000, 3),
        "avg_time_per_image_ms": round(avg_time_per_image * 1000, 3),
    }


def run_batch_comparison(engine, image_paths: list, batch_sizes: list,
                          warmup_iters: int, timed_iters: int) -> dict:
    """Run run_batch_benchmark for every requested batch size and add a speedup column."""
    results = {}
    for batch_size in batch_sizes:
        results[str(batch_size)] = run_batch_benchmark(engine, image_paths, batch_size, warmup_iters, timed_iters)

    if "1" in results:
        base = results["1"]["avg_time_per_image_ms"]
        for batch_size, stats in results.items():
            if batch_size == "1":
                continue
            stats["speedup_vs_batch1"] = round(base / stats["avg_time_per_image_ms"], 3)
            log(f"[batch={batch_size}] {stats['speedup_vs_batch1']}x faster per image than batch=1")

    return results
