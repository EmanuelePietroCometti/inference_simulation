"""Writes the run summary and per-image results to disk (BENCHMARK results)."""

import csv
import json

from src.utils import log, ensure_dir


def _format_summary_txt(summary: dict) -> str:
    lines = ["=" * 45, "            BENCHMARK RESULTS              ", "=" * 45]

    for key, value in summary.items():
        if key == "batch_throughput_comparison":
            continue
        lines.append(f"{key}: {value}")

    lines.append("-" * 45)
    lines.append("BATCH THROUGHPUT COMPARISON")
    lines.append("-" * 45)
    for batch_size, stats in summary.get("batch_throughput_comparison", {}).items():
        lines.append(f"BATCH SIZE {batch_size}:")
        lines.append(f"  Avg time per batch: {stats['avg_time_per_batch_ms']:.2f} ms")
        lines.append(f"  Avg time per image: {stats['avg_time_per_image_ms']:.2f} ms")
        if "speedup_vs_batch1" in stats:
            lines.append(f"  Speedup vs batch 1: {stats['speedup_vs_batch1']:.2f}x")

    lines.append("=" * 45)
    return "\n".join(lines)


def write_benchmark_results(output_dir: str, summary: dict, per_image_records: list) -> None:
    """
    Save the run summary in two formats plus a per-image CSV inside output_dir:
      - benchmark_results.txt  (human-readable, includes batch 1 vs batch 17 comparison)
      - benchmark_results.json (machine-readable)
      - per_image_results.csv  (filename, raw/normalized score, verdict)
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

    csv_path = out_dir / "per_image_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "raw_anomaly_score", "normalized_anomaly_score", "is_anomalous"]
        )
        writer.writeheader()
        writer.writerows(per_image_records)
    log(f"Per-image results written to: {csv_path}")
