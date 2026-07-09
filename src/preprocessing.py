"""
Image preprocessing.

The exact preprocessing required depends on how the ONNX model was exported:
  - Newer export pipeline: input is uint8 HWC, RGB, normalization is done inside the graph.
  - Older / default pipeline: input is float32 CHW, normalized here using ImageNet
    statistics (matches the training preprocessing in datamodules/base/datamodule.py).

Since both variants exist across export scripts used in this project, the Preprocessor
inspects the ONNX model's input dtype at runtime and automatically selects the matching
mode, logging the decision so it is always visible which one was used for a given run.
"""

import numpy as np
import cv2
import onnxruntime as ort

from src.utils import log

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_IMAGE_SIZE = (256, 256)  # (height, width), used as a fallback for dynamic shapes


class Preprocessor:
    """Prepares images for a specific ONNX model based on its declared input signature."""

    def __init__(self, session: ort.InferenceSession):
        input_meta = session.get_inputs()[0]
        self.input_name = input_meta.name
        self.onnx_dtype = input_meta.type
        self.onnx_shape = input_meta.shape

        self.image_size = self._resolve_image_size(self.onnx_shape)
        self.mode = self._resolve_mode(self.onnx_dtype)

        log(f"Model input '{self.input_name}': dtype={self.onnx_dtype}, shape={self.onnx_shape}")
        log(f"Preprocessing mode auto-selected: {self.mode} | target size (H, W): {self.image_size}")
        if self.mode == "float32_chw_imagenet":
            log("NOTE: input is float32, assuming ImageNet mean/std normalization "
                "(matches the training pipeline). If this specific model was exported "
                "with different preprocessing, results will be incorrect.")

    @staticmethod
    def _resolve_image_size(shape) -> tuple[int, int]:
        dims = [d for d in shape if isinstance(d, int) and d > 1]
        if len(dims) >= 2:
            return dims[-2], dims[-1]
        return DEFAULT_IMAGE_SIZE

    @staticmethod
    def _resolve_mode(dtype: str) -> str:
        if "uint8" in dtype:
            return "uint8_hwc_ingraph"
        return "float32_chw_imagenet"

    def __call__(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image, returning a batched (1, ...) array."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = self.image_size
        img_resized = cv2.resize(img_rgb, (w, h))

        if self.mode == "uint8_hwc_ingraph":
            tensor = img_resized.astype(np.uint8)
            return np.expand_dims(tensor, axis=0)  # (1, H, W, 3)

        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        img_chw = img_norm.transpose(2, 0, 1)
        return np.expand_dims(img_chw, axis=0).astype(np.float32)  # (1, 3, H, W)

    def load_original_bgr(self, image_path: str) -> np.ndarray:
        """Load the original image (BGR, resized to the model's input size) for visualization."""
        img_bgr = cv2.imread(str(image_path))
        h, w = self.image_size
        return cv2.resize(img_bgr, (w, h))
