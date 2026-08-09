"""
Pure inference-throughput benchmark: compares batch size 1 vs batch size 17
(or any sizes requested), reusing images from the input folder. No heatmaps or
thresholds are involved here — this only measures raw inference speed
(``session.run`` only; preprocessing/post-processing are excluded, matching the
timing the rest of the pipeline reports).

Out-of-memory is handled gracefully: the provider configuration
(src/provider_setup.py) is what actually prevents OOM, but if a batch size still
exhausts memory on a given machine we catch it, report it clearly, and continue
with the other batch sizes instead of crashing the whole run.
"""

import gc
import time
import numpy as np

from src.utils import log


# ORT surfaces allocation failures as onnxruntime exceptions whose message
# contains these markers; CUDA/TensorRT OOM also raise RuntimeError/MemoryError.
_OOM_MARKERS = ("out of memory", "oom", "cudaerrormemoryallocation",
                "cublas_status_alloc_failed", "failed to allocate memory")


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    return any(marker in str(exc).lower() for marker in _OOM_MARKERS)


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
    """Run a warm-up + timed loop at a fixed batch size and return timing stats.

    On out-of-memory (e.g. PatchCore at large batch on a small GPU, where VRAM
    usage grows with the batch — see README), the batch size is reported as
    ``oom`` and skipped instead of crashing, so the other batch sizes still run."""
    try:
        batch_tensor = _make_batch(engine, image_paths, batch_size)

        log(f"[batch={batch_size}] Warming up ({warmup_iters} iterations)...")
        for _ in range(warmup_iters):
            engine.run_batch(batch_tensor)

        log(f"[batch={batch_size}] Timing ({timed_iters} iterations)...")
        start = time.perf_counter()
        for _ in range(timed_iters):
            engine.run_batch(batch_tensor)
        total_time = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - we classify below
        if _is_oom(exc):
            log(f"[batch={batch_size}] OUT OF MEMORY - skipping this batch size. "
                f"VRAM usage grows with batch size; this model does not fit at this "
                f"batch on this GPU (see README). ({exc})")
            gc.collect()
            return {
                "batch_size": batch_size,
                "timed_iterations": 0,
                "oom": True,
                "error": str(exc),
            }
        raise

    avg_batch_time = total_time / timed_iters
    avg_time_per_image = avg_batch_time / batch_size

    return {
        "batch_size": batch_size,
        "timed_iterations": timed_iters,
        "oom": False,
        "avg_time_per_batch_ms": round(avg_batch_time * 1000, 3),
        "avg_time_per_image_ms": round(avg_time_per_image * 1000, 3),
    }


def run_batch_comparison(engine, image_paths: list, batch_sizes: list,
                         warmup_iters: int, timed_iters: int) -> dict:
    """Run run_batch_benchmark for every requested batch size and add a speedup column."""
    results = {}
    for batch_size in batch_sizes:
        results[str(batch_size)] = run_batch_benchmark(engine, image_paths, batch_size, warmup_iters, timed_iters)

    base_rec = results.get("1")
    if base_rec and not base_rec.get("oom"):
        base = base_rec["avg_time_per_image_ms"]
        for batch_size, stats in results.items():
            if batch_size == "1" or stats.get("oom"):
                continue
            stats["speedup_vs_batch1"] = round(base / stats["avg_time_per_image_ms"], 3)
            log(f"[batch={batch_size}] {stats['speedup_vs_batch1']}x faster per image than batch=1")

    return results
