"""
Resolves the correct blur / score-source configuration for a loaded ONNX model.

Why this exists
----------------
This runtime was originally written for SuperSimpleNet, where the graph's
``anomaly_score`` output IS the number to threshold on, and the training blur is
kernel=25/sigma=4. Other architectures (SK-RD4AD, and possibly PatchCore /
EfficientAD) use a *different* score convention and blur - e.g. SK-RD4AD's
calibrated threshold is computed on ``max(GaussianBlur(map, (15,15), sigma=0))``,
NOT the graph's raw ``anomaly_score`` output. Relying on the operator to
remember architecture-specific CLI flags (``--score_from_map
--blur_kernel_size 15 --blur_sigma 0``) is exactly the kind of mistake that
silently produces plausible-looking but wrong scores with no error or crash -
this was diagnosed as the likely dominant cause of "sballati" SK-RD4AD output
values, on top of a separate resize-aliasing bug (see preprocessing.py).

The pure-graph export scripts (SuperSimpleNet's export_onnx.py, SK-RD4AD's
export_onnx_from_checkpoint.py, the anomalib exporter) now embed this
configuration directly in the .onnx file as metadata_props, so it travels with
the model and cannot be forgotten. This module reads that metadata (via
onnxruntime's ``get_modelmeta().custom_metadata_map``) and resolves the
runtime's actual blur/score configuration, honouring explicit CLI overrides
when given.
"""

from dataclasses import dataclass

from src.utils import log


@dataclass
class RuntimeConfig:
    score_source: str          # "graph" or "map_max_blurred"
    blur_kernel_size: int
    blur_sigma: float
    architecture: str
    verified: bool


# Legacy fallback for .onnx files exported before metadata was added (no
# "anomaly_export_contract" key present). Matches the original SuperSimpleNet-only
# behaviour of this runtime, so old exports keep working exactly as before.
_LEGACY_FALLBACK = RuntimeConfig(
    score_source="graph", blur_kernel_size=25, blur_sigma=4.0,
    architecture="unknown (pre-metadata export)", verified=False,
)


def resolve_runtime_config(metadata: dict, args) -> RuntimeConfig:
    has_metadata = "anomaly_export_contract" in metadata

    if not has_metadata:
        log("WARNING: this ONNX model has no embedded architecture metadata "
            "(exported before the metadata contract was added). Falling back to "
            "SuperSimpleNet's original defaults (score_source=graph, blur=25/4.0). "
            "Re-export the model to get correct auto-configuration, or pass "
            "--blur_kernel_size / --blur_sigma / --score_source explicitly.")
        base = _LEGACY_FALLBACK
    else:
        architecture = metadata.get("architecture", "unknown")
        score_source = metadata.get("score_source", "graph")
        blur_kernel_size = int(metadata.get("blur_kernel_size", 0))
        blur_sigma = float(metadata.get("blur_sigma", 0.0))
        verified = metadata.get("verified", "false") == "true"

        base = RuntimeConfig(
            score_source=score_source, blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma, architecture=architecture, verified=verified,
        )
        log(f"Auto-configured from model metadata: architecture={architecture}, "
            f"score_source={score_source}, blur=({blur_kernel_size}, sigma={blur_sigma})")
        if not verified:
            log(f"WARNING: architecture '{architecture}' metadata is NOT verified "
                f"against a live training/eval run (no trained checkpoint was "
                f"available when the export script was written). Validate scores "
                f"and the anomaly map against this model's own eval pipeline "
                f"before trusting production verdicts.")

    # Explicit CLI flags always win over metadata/fallback.
    score_source = args.score_source if args.score_source != "auto" else base.score_source
    blur_kernel_size = args.blur_kernel_size if args.blur_kernel_size is not None else base.blur_kernel_size
    blur_sigma = args.blur_sigma if args.blur_sigma is not None else base.blur_sigma

    if score_source not in ("graph", "map_max_blurred"):
        log(f"WARNING: unrecognized score_source '{score_source}' in model metadata; "
            f"treating it as 'graph'.")
        score_source = "graph"

    return RuntimeConfig(
        score_source=score_source, blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma, architecture=base.architecture, verified=base.verified,
    )
