"""
Per-execution-provider partition report.

``session.get_providers()`` only says which EPs are *registered*, not which one ran
each node. With TensorRT -> CUDA -> CPU fallback a model can look "on TensorRT" while
half its graph silently runs on CUDA or CPU — and that partial fallback is the most
likely explanation for two models with the same backbone timing very differently
(the SuperSimpleNet vs SK-RD4AD anomaly).

ONNX Runtime's built-in profiler records, per executed node, the op type and the EP
that ran it. This module runs ONE profiled inference, parses the profile JSON, and
returns a node/time breakdown per provider — turning "active_providers" into an
actual partition you can cite.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict

import numpy as np
import onnxruntime as ort

from src.utils import log


def profile_ep_partition(model_path: str, providers: list, input_name: str,
                         sample_input: np.ndarray, output_names: list) -> dict | None:
    """Run a single profiled inference and summarize node placement per EP.

    A dedicated short-lived session with profiling enabled is used so the main
    benchmark session is never slowed by profiling. Returns a dict with per-provider
    node counts and cumulative kernel time, plus the top ops that fell OFF TensorRT
    (i.e. ran on CUDA/CPU) — the fallback fingerprint. Best-effort: any failure logs
    a warning and returns None rather than aborting the run."""
    def _prov_name(p):
        return p[0] if isinstance(p, (tuple, list)) else p
    trt_requested = any("ensorrt" in _prov_name(p) for p in providers)

    tmp_dir = tempfile.mkdtemp(prefix="ep_profile_")
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(tmp_dir, "ort_profile")
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        sess.run(output_names, {input_name: sample_input})
        profile_path = sess.end_profiling()
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: EP-partition profiling failed ({exc}); skipping partition report.")
        return None

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not read EP profile ({exc}); skipping partition report.")
        return None

    nodes_per_ep: Counter = Counter()
    time_per_ep_us: defaultdict = defaultdict(float)
    fallback_ops: Counter = Counter()  # ops NOT on a TensorRT subgraph

    for ev in events:
        if ev.get("cat") != "Node" or not ev.get("name", "").endswith("_kernel_time"):
            continue
        args = ev.get("args", {}) or {}
        ep = args.get("provider", "Unknown")
        nodes_per_ep[ep] += 1
        time_per_ep_us[ep] += float(ev.get("dur", 0))
        # "Fallback" only makes sense when TensorRT was actually requested; on a
        # plain CPU/CUDA run every node trivially runs off TensorRT (not a fallback).
        if trt_requested and "ensorrt" not in ep and "ensorRT" not in ep:
            op = args.get("op_name", "Unknown")
            fallback_ops[op] += 1

    if not nodes_per_ep:
        log("WARNING: EP profile contained no node kernel events; skipping partition report.")
        return None

    total_nodes = sum(nodes_per_ep.values())
    total_time = sum(time_per_ep_us.values()) or 1.0
    per_ep = {}
    for ep, count in nodes_per_ep.most_common():
        per_ep[ep] = {
            "nodes": count,
            "nodes_pct": round(100.0 * count / total_nodes, 1),
            "kernel_time_ms": round(time_per_ep_us[ep] / 1000.0, 3),
            "kernel_time_pct": round(100.0 * time_per_ep_us[ep] / total_time, 1),
        }

    trt_nodes = sum(c for ep, c in nodes_per_ep.items() if "ensorrt" in ep or "ensorRT" in ep)
    result = {
        "executed_nodes_total": total_nodes,
        "trt_requested": trt_requested,
        "per_provider": per_ep,
        "fully_on_tensorrt": bool(trt_nodes == total_nodes) if trt_requested else None,
        "top_fallback_ops": dict(fallback_ops.most_common(10)),
    }

    summary = ", ".join(f"{ep}: {d['nodes']} nodes ({d['nodes_pct']}%), "
                        f"{d['kernel_time_pct']}% time" for ep, d in per_ep.items())
    log(f"EP partition -> {summary}")
    if trt_requested and not result["fully_on_tensorrt"] and result["top_fallback_ops"]:
        log(f"NOTE: {total_nodes - trt_nodes} node(s) fell off TensorRT to CUDA/CPU. "
            f"Top fallback ops: {result['top_fallback_ops']}. This partial fallback can "
            f"explain latency differences between same-backbone models.")
    return result
