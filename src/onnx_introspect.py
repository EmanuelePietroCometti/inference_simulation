"""
Reads the real input signature straight from the ONNX graph.

The TensorRT dynamic-batch profile (src/provider_setup.py) is built from the input
name and the C/H/W dims. Guessing those (``"image"``, ``3x256x256``) before looking
at the model is a latent bug: if the export names its input differently or uses a
different spatial size, the profile references a non-existent/wrong input and
TensorRT silently ignores it (or rebuilds per shape), so the "one engine for the
whole batch range" guarantee quietly breaks.

This module inspects the ONNX first so main.py can pass the ACTUAL name and dims to
``build_session_plans``. It uses ``load_external_data=False`` so it stays cheap even
for large models.
"""

from __future__ import annotations

from src.utils import log, die


def inspect_input_signature(model_path: str,
                            fallback=("image", 3, 256, 256)) -> tuple[str, int, int, int]:
    """Return (input_name, channels, height, width) from the model's first input.

    The batch dim is expected to be dynamic (a string/0) and is ignored here — the
    profile's batch axis comes from --batch_sizes. Non-integer or missing C/H/W dims
    fall back to the provided defaults with a warning (a fully dynamic spatial shape
    cannot seed a fixed profile)."""
    try:
        import onnx
    except ImportError:
        log("WARNING: 'onnx' not installed; cannot introspect the model input. "
            f"Falling back to {fallback}.")
        return fallback

    try:
        model = onnx.load(model_path, load_external_data=False)
    except Exception as exc:  # noqa: BLE001
        die(f"Failed to open ONNX model for introspection '{model_path}': {exc}")

    if not model.graph.input:
        die(f"ONNX model '{model_path}' declares no graph inputs.")

    inp = model.graph.input[0]
    name = inp.name or fallback[0]
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else None)

    # Expect NCHW: [batch, C, H, W]. Take the last three as C, H, W.
    fb_name, fb_c, fb_h, fb_w = fallback
    if len(dims) == 4 and all(v is not None for v in dims[1:]):
        c, h, w = dims[1], dims[2], dims[3]
    else:
        log(f"WARNING: input '{name}' has non-static CHW dims {dims}; using fallback "
            f"C/H/W = {fb_c}/{fb_h}/{fb_w} for the TensorRT profile.")
        c, h, w = fb_c, fb_h, fb_w

    log(f"ONNX input signature: name='{name}', C={c}, H={h}, W={w} "
        f"(batch axis dynamic; profile batch from --batch_sizes)")
    return name, c, h, w
