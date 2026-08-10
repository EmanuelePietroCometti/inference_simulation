"""Writes the run summary and per-image results to disk (BENCHMARK results)."""

import csv
import json

from src.utils import log, ensure_dir


def _format_summary_txt(summary: dict) -> str:
    lines = ["=" * 45, "            BENCHMARK RESULTS              ", "=" * 45]

    skip = {"batch_throughput_comparison", "environment", "ep_partition"}
    for key, value in summary.items():
        if key in skip:
            continue
        lines.append(f"{key}: {value}")

    env = summary.get("environment")
    if env:
        lines.append("-" * 45)
        lines.append("ENVIRONMENT")
        lines.append("-" * 45)
        gpu = env.get("gpu", {})
        lines.append(f"  timestamp : {env.get('timestamp_utc')}")
        lines.append(f"  os        : {env.get('os')}")
        lines.append(f"  gpu       : {gpu.get('name')} ({gpu.get('vram_total_mib')} MiB), "
                     f"driver {gpu.get('driver_version')}, clock "
                     f"{gpu.get('current_graphics_clock_mhz')}/{gpu.get('max_graphics_clock_mhz')} MHz "
                     f"(locked={gpu.get('clocks_locked')})")
        v = env.get("versions", {})
        lines.append(f"  versions  : ORT {v.get('onnxruntime')}, TRT {v.get('tensorrt')}, "
                     f"CUDA {v.get('cuda_runtime')}, onnx {v.get('onnx')}, numpy {v.get('numpy')}, "
                     f"py {v.get('python')}")
        lines.append(f"  git       : {env.get('git_commit')} (dirty={env.get('git_dirty')})")

    epp = summary.get("ep_partition")
    if epp:
        lines.append("-" * 45)
        lines.append("EXECUTION-PROVIDER PARTITION")
        lines.append("-" * 45)
        lines.append(f"  executed nodes: {epp['executed_nodes_total']} "
                     f"(fully_on_tensorrt={epp['fully_on_tensorrt']})")
        for ep, d in epp["per_provider"].items():
            lines.append(f"    {ep}: {d['nodes']} nodes ({d['nodes_pct']}%), "
                         f"{d['kernel_time_pct']}% kernel time")
        if epp.get("trt_requested") and epp.get("top_fallback_ops"):
            lines.append(f"  top fallback ops (off TensorRT): {epp['top_fallback_ops']}")

    lines.append("-" * 45)
    lines.append("BATCH THROUGHPUT COMPARISON")
    lines.append("-" * 45)
    btc = summary.get("batch_throughput_comparison", {})
    opt_mode = btc.get("opt_mode")
    if opt_mode == "per_batch":
        lines.append("(each batch timed on its OWN static-shape engine, "
                     "min=opt=max=batch — best achievable per batch, no opt-shape artifact)")
    elif opt_mode == "shared":
        lines.append(f"(SHARED engine over all batches, opt_batch={btc.get('shared_opt_batch')}; "
                     f"batches != opt run with non-optimal tactics — production-parity mode)")
    for batch_size, stats in btc.get("results", {}).items():
        lines.append(f"BATCH SIZE {batch_size}:")
        if stats.get("oom"):
            phase = stats.get("oom_phase", "runtime")
            lines.append(f"  OUT OF MEMORY ({phase}) - batch size skipped")
            continue
        lines.append(f"  time/batch : mean={stats['avg_time_per_batch_ms']:.2f} ms  "
                     f"median={stats['median_time_per_batch_ms']:.2f}  "
                     f"p95={stats['p95_time_per_batch_ms']:.2f}  "
                     f"p99={stats['p99_time_per_batch_ms']:.2f}  "
                     f"std={stats['std_time_per_batch_ms']:.2f}")
        lines.append(f"  time/image : {stats['avg_time_per_image_ms']:.3f} ms  "
                     f"-> {stats['throughput_img_per_sec']} img/s")
        if "speedup_vs_batch1" in stats:
            lines.append(f"  speedup vs batch 1: {stats['speedup_vs_batch1']:.2f}x")

    lines.append("=" * 45)
    return "\n".join(lines)


def write_benchmark_results(output_dir: str, summary: dict, per_image_records: list) -> None:
    """
    Save the run summary in two formats plus a per-image CSV inside output_dir:
      - benchmark_results.txt  (human-readable: env, EP partition, batch stats)
      - benchmark_results.json (machine-readable, full detail)
      - per_image_results.csv  (filename, scores, verdict)
    """
    out_dir = ensure_dir(output_dir)

    txt_path = out_dir / "benchmark_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_format_summary_txt(summary))
        f.write("\n")
    log(f"Text summary written to: {txt_path}")

    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"JSON summary written to: {json_path}")

    fieldnames = ["filename", "raw_anomaly_score", "normalized_anomaly_score", "is_anomalous"]
    csv_path = out_dir / "per_image_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_image_records)
    log(f"Per-image results written to: {csv_path}")
