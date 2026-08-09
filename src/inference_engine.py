"""Wraps the ONNX Runtime InferenceSession for the anomaly detection models.

Mirrors the C++ engine's contract: the graph emits the final ``anomaly_score``
and ``anomaly_map``; this class just runs the session and hands the raw outputs
back. Thresholding and rendering happen in main.py using the embedded contract
(src/model_config.py), exactly like ONNX_inference/AnomalyEngine.cpp.
"""

import time
import numpy as np
import onnxruntime as ort

from src.preprocessing import Preprocessor
from src.model_config import load_contract
from src.utils import log, die


class AnomalyInferenceEngine:
    """Loads the ONNX model and exposes a simple preprocess + run interface."""

    def __init__(self, model_path: str, session_plans: list):
        self.session = self._create_session_with_fallback(model_path, session_plans)

        active_providers = self.session.get_providers()
        log(f"Model loaded successfully. Active execution providers: {active_providers}")

        # Raw metadata (kept pristine) + parsed calibration contract.
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.contract = load_contract(self.metadata)

        self.preprocessor = Preprocessor(self.session, self.metadata)
        self.input_name = self.preprocessor.input_name

        self.anomaly_map_name = self.metadata.get("output_map_name", "anomaly_map")
        self.anomaly_score_name = self.metadata.get("output_score_name", "anomaly_score")
        log(f"Outputs -> anomaly_map: '{self.anomaly_map_name}', "
            f"anomaly_score: '{self.anomaly_score_name}'")

    @staticmethod
    def _make_session_options():
        so = ort.SessionOptions()
        # Keep the arena off: batch 17 of the large backbones would otherwise
        # make the CPU arena reserve (and never release) a large block.
        so.enable_cpu_mem_arena = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.use_deterministic_compute = True
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return so

    def _create_session_with_fallback(self, model_path: str, session_plans: list):
        """Try each session plan (see src/provider_setup.py) in order, degrading on
        failure like the C++ engine's TensorRT -> CUDA -> CPU cascade. The primary
        TensorRT plan already lists [TensorRT, CUDA, CPU] so ORT falls back between
        EPs inside one session; the extra CUDA plan only covers a hard TRT build
        failure. A fresh SessionOptions is used per attempt so a failed plan never
        leaks state into the next."""
        for i, plan in enumerate(session_plans):
            so = self._make_session_options()
            label = plan.get("label", "?")
            try:
                return ort.InferenceSession(model_path, sess_options=so, providers=plan["providers"])
            except Exception as e:  # noqa: BLE001
                if i < len(session_plans) - 1:
                    log(f"WARNING: session plan '{label}' failed to initialize/build for "
                        f"this model ({e}). Falling back to '{session_plans[i + 1].get('label', '?')}'.")
                    continue
                die(f"Failed to load ONNX model '{model_path}' (last plan '{label}'): {e}")
        die(f"Failed to load ONNX model '{model_path}': no session plan succeeded.")

    def preprocess(self, image_path: str) -> np.ndarray:
        return self.preprocessor(image_path)

    def run_batch(self, batch_tensor: np.ndarray):
        """Run inference on a pre-stacked batch. Returns (anomaly_maps, anomaly_scores, elapsed_seconds).

        Timing wraps ONLY session.run (no preprocessing/post-processing), which is
        the raw-inference latency the benchmark reports."""
        start = time.perf_counter()
        anomaly_maps, anomaly_scores = self.session.run(
            [self.anomaly_map_name, self.anomaly_score_name],
            {self.input_name: batch_tensor},
        )
        elapsed = time.perf_counter() - start
        return anomaly_maps, anomaly_scores, elapsed
