"""Wraps the ONNX Runtime InferenceSession for the SuperSimpleNet anomaly detection model."""

import time
import numpy as np
import onnxruntime as ort

from src.preprocessing import Preprocessor
from src.utils import log, die


class AnomalyInferenceEngine:
    """Loads the ONNX model and exposes a simple preprocess + run interface."""

    def __init__(self, model_path: str, providers: list):
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            die(f"Failed to load ONNX model '{model_path}': {e}")

        active_providers = self.session.get_providers()
        log(f"Model loaded successfully. Active execution providers: {active_providers}")

        # Architecture-specific config (score source, blur, dynamic crop) embedded
        # by the export script; see src/model_config.py and preprocessing.py.
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)

        self.preprocessor = Preprocessor(self.session, self.metadata)
        self.input_name = self.preprocessor.input_name

        output_names = [o.name for o in self.session.get_outputs()]
        self.anomaly_map_name = self._pick_output(output_names, "anomaly_map", "map")
        self.anomaly_score_name = self._pick_output(output_names, "output", "score")
        log(f"Outputs -> anomaly_map: '{self.anomaly_map_name}', anomaly_score: '{self.anomaly_score_name}'")

    @staticmethod
    def _pick_output(output_names: list, *keywords: str) -> str:
        for name in output_names:
            if any(keyword in name.lower() for keyword in keywords):
                return name
        die(f"Could not find an output matching {keywords} among model outputs: {output_names}")

    def preprocess(self, image_path: str) -> np.ndarray:
        return self.preprocessor(image_path)

    def run_batch(self, batch_tensor: np.ndarray):
        """Run inference on a pre-stacked batch. Returns (anomaly_maps, anomaly_scores, elapsed_seconds)."""
        start = time.perf_counter()
        anomaly_maps, anomaly_scores = self.session.run(
            [self.anomaly_map_name, self.anomaly_score_name],
            {self.input_name: batch_tensor},
        )
        elapsed = time.perf_counter() - start
        return anomaly_maps, anomaly_scores, elapsed
