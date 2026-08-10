"""
Pure inference-throughput benchmark: compares batch sizes (default 1 vs 17, or any
sizes requested), reusing images from the input folder. No heatmaps or thresholds
are involved here — this only measures raw inference speed (``session.run`` only;
preprocessing/post-processing are excluded, matching the timing the rest of the
pipeline reports).

Statistics
----------
Industrial inspection cares about the *tail*, not just the average, so each batch
size reports mean / median / std / p95 / p99 / min over per-iteration times, plus
throughput in images/second. ``avg_time_per_image_ms`` is kept for backward
compatibility with older reports.

Caveat (documented in every record): these are Python ``session.run`` latencies.
They include the host<->device input/output copies that the C++ engine avoids with
``Ort::IoBinding`` + pinned memory, so they are an UPPER BOUND on production latency,
not the production latency itself.

Out-of-memory is handled gracefully: if a batch size exhausts memory on a given
machine we catch it, report it clearly, and continue with the other batch sizes
instead of crashing the whole run.
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


def _timing_stats(per_iter_ms: np.ndarray, batch_size: int) -> dict:
    """Summarize per-iteration batch times (ms) into mean/median/std/p95/p99/min
    and derived per-image latency + throughput."""
    mean = float(np.mean(per_iter_ms))
    per_image_mean = mean / batch_size
    return {
        "avg_time_per_batch_ms": round(mean, 3),
        "median_time_per_batch_ms": round(float(np.median(per_iter_ms)), 3),
        "std_time_per_batch_ms": round(float(np.std(per_iter_ms)), 3),
        "p95_time_per_batch_ms": round(float(np.percentile(per_iter_ms, 95)), 3),
        "p99_time_per_batch_ms": round(float(np.percentile(per_iter_ms, 99)), 3),
        "min_time_per_batch_ms": round(float(np.min(per_iter_ms)), 3),
        "avg_time_per_image_ms": round(per_image_mean, 3),
        "median_time_per_image_ms": round(float(np.median(per_iter_ms)) / batch_size, 3),
        "throughput_img_per_sec": round(1000.0 / per_image_mean, 2) if per_image_mean > 0 else None,
    }


def run_batch_benchmark(engine, image_paths: list, batch_size: int,
                        warmup_iters: int, timed_iters: int) -> dict:
    """Run a warm-up + timed loop at a fixed batch size and return timing stats.

    Each timed iteration is measured individually so we can report percentiles and
    the std, not only the mean. On out-of-memory the batch size is reported as
    ``oom`` and skipped instead of crashing, so the other batch sizes still run."""
    try:
        batch_tensor = _make_batch(engine, image_paths, batch_size)

        log(f"[batch={batch_size}] Warming up ({warmup_iters} iterations)...")
        for _ in range(warmup_iters):
            engine.run_batch(batch_tensor)

        log(f"[batch={batch_size}] Timing ({timed_iters} iterations)...")
        per_iter_ms = np.empty(timed_iters, dtype=np.float64)
        for i in range(timed_iters):
            start = time.perf_counter()
            engine.run_batch(batch_tensor)
            per_iter_ms[i] = (time.perf_counter() - start) * 1000.0
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

    stats = {
        "batch_size": batch_size,
        "timed_iterations": timed_iters,
        "oom": False,
        # Python session.run latency includes H2D/D2H copies (no IoBinding) -> upper bound.
        "latency_note": "python_session_run_upper_bound_includes_host_device_copies",
    }
    stats.update(_timing_stats(per_iter_ms, batch_size))
    log(f"[batch={batch_size}] mean={stats['avg_time_per_batch_ms']:.2f} ms "
        f"(p95={stats['p95_time_per_batch_ms']:.2f}, p99={stats['p99_time_per_batch_ms']:.2f}, "
        f"std={stats['std_time_per_batch_ms']:.2f}) | "
        f"{stats['throughput_img_per_sec']} img/s")
    return stats


def run_batch_comparison(engine_factory, image_paths: list, batch_sizes: list,
                         warmup_iters: int, timed_iters: int,
                         dispose_between_batches: bool = True,
                         opt_mode: str = "per_batch",
                         shared_opt_batch: int | None = None) -> dict:
    """Run run_batch_benchmark for every requested batch size and add a speedup column.

    ``engine_factory(batch_size)`` returns the engine to benchmark for that batch.
    In the default ``per_batch`` mode it builds a TensorRT engine optimized (static
    ``min=opt=max``) for exactly that batch, so every number is that batch's *best*
    achievable throughput — no opt-shape artifact, nothing to tune. In
    ``shared`` mode it returns one engine covering the whole range (production
    parity); then batches != opt run with non-optimal tactics.

    Build-time OOM is caught and the batch is reported as ``oom`` (phase=build) —
    e.g. PatchCore autotuning a large static batch — so the smaller batches still
    run. When ``dispose_between_batches`` is set, each engine is released before the
    next is built so a large batch is not starved of VRAM by the previous engine."""
    results = {}
    for batch_size in batch_sizes:
        try:
            engine = engine_factory(batch_size)
        except Exception as exc:  # noqa: BLE001 - classify build failures
            if _is_oom(exc):
                log(f"[batch={batch_size}] OUT OF MEMORY while BUILDING the engine for this "
                    f"batch - skipping. (Autotuning a static batch={batch_size} did not fit; "
                    f"this batch would also OOM at runtime.) ({exc})")
                gc.collect()
                results[str(batch_size)] = {"batch_size": batch_size, "timed_iterations": 0,
                                            "oom": True, "oom_phase": "build", "error": str(exc)}
                continue
            raise

        results[str(batch_size)] = run_batch_benchmark(
            engine, image_paths, batch_size, warmup_iters, timed_iters)

        if dispose_between_batches:
            del engine
            gc.collect()

    base_rec = results.get("1")
    if base_rec and not base_rec.get("oom"):
        base = base_rec["avg_time_per_image_ms"]
        for batch_size, stats in results.items():
            if batch_size == "1" or stats.get("oom"):
                continue
            stats["speedup_vs_batch1"] = round(base / stats["avg_time_per_image_ms"], 3)
            log(f"[batch={batch_size}] {stats['speedup_vs_batch1']}x faster per image than batch=1")

    return {
        "opt_mode": opt_mode,  # "per_batch": each batch on its own static-shape engine
        "shared_opt_batch": shared_opt_batch,  # set only in shared/production-parity mode
        "results": results,
    }
