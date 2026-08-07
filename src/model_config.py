"""
Resolves the correct blur / score-source configuration for a loaded ONNX model.

Why this exists
----------------
This runtime was originally written for SuperSimpleNet, where the graph's
``anomaly_score`` output IS the number to threshold on, and the training blur is
kernel=25/sigma=4. Other architectures use different conventions, and relying on
the operator to remember architecture-specific CLI flags is exactly the kind of
mistake that silently produces plausible-looking but wrong scores with no error
or crash. The export scripts therefore embed the configuration directly in the
.onnx file as metadata_props; this module reads it and resolves the runtime's
actual blur/score configuration, honouring explicit CLI overrides when given.

Anomaly export contracts
------------------------
- Contract 1.0 (legacy SK-RD4AD): the graph emits a RAW (un-blurred) map and an
  un-blurred score; ``score_source="map_max_blurred"`` tells the runtime to blur
  the map host-side (kernel/sigma from metadata) and score on its max.
- Contract 2.0 (current SK-RD4AD): ``map_blur="baked_in_graph"`` — the canonical
  Gaussian blur (k=15, sigma=4, zero padding; the training repo's test.py is the
  single source of truth) is INSIDE the graph, ``anomaly_map`` is already
  blurred, and ``anomaly_score`` (max of the blurred map) is directly comparable
  with the calibrated threshold. The host must apply NO blur: blurring again
  would smear the map and lower the score below the units the threshold was
  calibrated in. ``score_source="graph"``.
- SuperSimpleNet keeps its original convention: ``score_source="graph"`` (its
  dedicated classification head) with the display blur applied host-side.

An unrecognized ``score_source`` in the metadata is a hard error, not a guess:
it means the model was exported with a newer contract than this runtime
understands, and any fallback would produce a score no threshold was ever
calibrated against.
"""

from dataclasses import dataclass

from src.utils import log, die


@dataclass
class RuntimeConfig:
    score_source: str          # "graph" or "map_max_blurred"
    blur_kernel_size: int      # host-side blur; 0 when blur_in_graph
    blur_sigma: float
    architecture: str
    verified: bool
    blur_in_graph: bool = False


# Legacy fallback for .onnx files exported before metadata was added (no
# "anomaly_export_contract" key present). Matches the original SuperSimpleNet-only
# behaviour of this runtime, so old exports keep working exactly as before.
_LEGACY_FALLBACK = RuntimeConfig(
    score_source="graph", blur_kernel_size=25, blur_sigma=4.0,
    architecture="unknown (pre-metadata export)", verified=False,
)


def _from_export_contract(metadata: dict) -> RuntimeConfig:
    """Translate the standalone ``anomaly_export`` package's metadata contract
    (``contract_version``) into this runtime's RuntimeConfig.

    That package (SuperSimpleNet / SK-RD4AD / PatchCore / EfficientAD, one rigid
    I/O contract) always emits the FINAL image score on the ``anomaly_score``
    output and the FINAL map on ``anomaly_map`` — any Gaussian blur is baked into
    the graph (``blur_applied=true``) or intentionally absent (EfficientAD,
    ``blur_applied=false``). In every case the host must add NO blur and must
    threshold the graph's ``anomaly_score`` directly, i.e. score_source="graph".

    The package's own ``score_source`` values (classification_head / map_max /
    reweighted_nn_distance) describe how the graph computed the score internally;
    they all collapse to "graph" from this runtime's point of view. An unknown
    value is a hard error: it means a newer export contract this runtime has not
    been taught to read.
    """
    architecture = metadata.get("architecture") or metadata.get("model_name", "unknown")
    export_source = metadata.get("score_source", "")
    if export_source not in ("classification_head", "map_max", "reweighted_nn_distance"):
        die(f"Unrecognized score_source '{export_source}' for anomaly_export "
            f"contract_version {metadata.get('contract_version', '?')}: this runtime "
            f"does not know how to score it. Update the runtime rather than guess a "
            f"convention no threshold was calibrated against.")

    blur_applied = str(metadata.get("blur_applied", "false")).lower() == "true"
    # Provenance only: the host never re-applies these (blur is in the graph).
    bk = int(float(metadata.get("blur_kernel_size", 0) or 0))
    bs = float(metadata.get("blur_sigma", 0.0) or 0.0)

    log(f"Auto-configured from anomaly_export contract_version "
        f"{metadata.get('contract_version', '?')}: architecture={architecture}, "
        f"export score_source={export_source} -> runtime score_source=graph, "
        f"blur={'baked in graph (k=%d, sigma=%g)' % (bk, bs) if blur_applied else 'none (EfficientAD)'}, "
        f"host-side blur disabled.")
    if str(metadata.get("normalize_inside_graph", "")).lower() != "true":
        log("WARNING: anomaly_export contract without normalize_inside_graph=true; "
            "verify host preprocessing matches this model's expected input.")

    return RuntimeConfig(
        score_source="graph", blur_kernel_size=0, blur_sigma=0.0,
        architecture=architecture,
        # The export runs a PyTorch-vs-ONNXRuntime numeric parity check at export
        # time, so the graph's numbers are trustworthy for the contract it declares.
        verified=True, blur_in_graph=True,
    )


def resolve_runtime_config(metadata: dict, args) -> RuntimeConfig:
    # Hard stop for export-pipeline test artifacts. A --self_test export contains
    # RANDOM (untrained) weights: for reconstruction-based models the anomaly map
    # saturates near-uniformly (all-red heatmap) and the max score is essentially
    # the same ~1 value for every image. Those outputs look exactly like a subtle
    # inference bug, so refusing here saves a debugging session.
    if metadata.get("weights_source") == "random_self_test":
        die(
            "This ONNX file was exported with --self_test and contains RANDOM "
            "(untrained) weights - it exists only to test the export pipeline and "
            "cannot produce meaningful anomaly maps or scores.\n"
            "Re-export from your trained checkpoint WITHOUT --self_test, e.g.:\n"
            "  python export_onnx_from_checkpoint.py <checkpoint.pth> <output_dir>"
        )
    if "weights_source" in metadata:
        log(f"Model weights source: {metadata['weights_source']}")

    # Two distinct metadata contracts can appear:
    #   - "anomaly_export_contract": this runtime's own native export scripts.
    #   - "contract_version": the standalone anomaly_export package (4 architectures,
    #     one rigid contract). Translated by _from_export_contract.
    has_metadata = "anomaly_export_contract" in metadata
    has_export_contract = "contract_version" in metadata

    if has_export_contract and not has_metadata:
        base = _from_export_contract(metadata)
        # CLI overrides (blur/score_source) are applied by the shared tail below.
        score_source = args.score_source if args.score_source != "auto" else base.score_source
        if base.blur_in_graph and (args.blur_kernel_size is not None or args.blur_sigma is not None):
            log("WARNING: --blur_kernel_size/--blur_sigma IGNORED: this model's blur "
                "is baked into the graph; applying another host-side blur would "
                "invalidate the calibrated threshold.")
        if score_source not in ("graph", "map_max_blurred"):
            die(f"Unrecognized --score_source '{score_source}'.")
        return RuntimeConfig(
            score_source=score_source, blur_kernel_size=base.blur_kernel_size,
            blur_sigma=base.blur_sigma, architecture=base.architecture,
            verified=base.verified, blur_in_graph=base.blur_in_graph,
        )

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
        blur_in_graph = metadata.get("map_blur") == "baked_in_graph"
        blur_kernel_size = int(metadata.get("blur_kernel_size", 0))
        blur_sigma = float(metadata.get("blur_sigma", 0.0))
        verified = metadata.get("verified", "false") == "true"

        if score_source not in ("graph", "map_max_blurred"):
            die(f"Unrecognized score_source '{score_source}' in model metadata "
                f"(contract {metadata.get('anomaly_export_contract', '?')}): this "
                f"model was exported with a contract this runtime does not "
                f"understand. Update the runtime; scoring with a guessed "
                f"convention would produce numbers no threshold was calibrated "
                f"against.")
        if blur_in_graph:
            # The metadata's blur_kernel_size/blur_sigma DESCRIBE the in-graph
            # blur for provenance; the host must not apply them again.
            log(f"Blur is baked INTO the graph (k={blur_kernel_size}, "
                f"sigma={blur_sigma}, contract "
                f"{metadata.get('anomaly_export_contract', '?')}): host-side "
                f"blur disabled, anomaly_score used as-is.")
            blur_kernel_size, blur_sigma = 0, 0.0

        base = RuntimeConfig(
            score_source=score_source, blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma, architecture=architecture, verified=verified,
            blur_in_graph=blur_in_graph,
        )
        log(f"Auto-configured from model metadata: architecture={architecture}, "
            f"score_source={score_source}, blur=" +
            ("in-graph" if blur_in_graph else f"({blur_kernel_size}, sigma={blur_sigma})"))
        if not verified:
            log(f"WARNING: architecture '{architecture}' metadata is NOT verified "
                f"against a live training/eval run for this contract version. "
                f"Validate scores and the anomaly map against this model's own "
                f"eval pipeline (training repo: parity_check.py) before trusting "
                f"production verdicts.")

    # Explicit CLI flags win over metadata/fallback — except host-side blur on a
    # model whose blur is already inside the graph: that is wrong in every case
    # (the map would be smeared twice and the score would leave the units the
    # threshold was calibrated in), so it is ignored, loudly.
    score_source = args.score_source if args.score_source != "auto" else base.score_source
    if base.blur_in_graph and (args.blur_kernel_size is not None or args.blur_sigma is not None):
        log("WARNING: --blur_kernel_size/--blur_sigma IGNORED: this model's blur "
            "is baked into the graph; applying another host-side blur would "
            "invalidate the calibrated threshold.")
        blur_kernel_size, blur_sigma = base.blur_kernel_size, base.blur_sigma
    else:
        blur_kernel_size = args.blur_kernel_size if args.blur_kernel_size is not None else base.blur_kernel_size
        blur_sigma = args.blur_sigma if args.blur_sigma is not None else base.blur_sigma

    if score_source not in ("graph", "map_max_blurred"):
        die(f"Unrecognized --score_source '{score_source}'.")

    return RuntimeConfig(
        score_source=score_source, blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma, architecture=base.architecture, verified=base.verified,
        blur_in_graph=base.blur_in_graph,
    )
