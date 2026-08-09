"""
Image preprocessing, aligned with the C++ engine's contract
(ONNX_inference/AnomalyEngine.cpp) and the model's embedded preproc metadata.

The exact preprocessing depends on how the ONNX model was exported. The
anomaly_export contract (all current models) declares:
  - float32 CHW input, RGB channel order,
  - ``preproc_scale = 1/255`` and ``normalize_inside_graph=true``: the host scales
    to [0,1] and the graph applies mean/std itself. Applying ImageNet stats on the
    host too would normalize twice and silently corrupt every score.

Two other modes are kept as safe fallbacks for models exported differently:
  - uint8 HWC, all scaling+normalization inside the graph;
  - float32 CHW with host-side ImageNet mean/std (legacy, no normalize_inside_graph flag).
The mode is auto-selected from the ONNX input dtype and the metadata flag, and
logged so it is always visible which one ran.

Resize / anti-aliasing
----------------------
The models declare ``preproc_resize_interpolation=bilinear`` with
``preproc_antialias=true`` — this is how the calibrated thresholds
(``image_threshold_raw`` etc.) were computed, and it matches the training
pipeline's ``torchvision Resize(antialias=True)``. This runtime therefore resizes
with PIL's antialiased ``Image.resize(BILINEAR)``, the closest available match.
(Note: the C++ engine currently downsamples with cv2.INTER_AREA, which
contradicts this metadata; that is a known C++-side discrepancy to reconcile
separately. Python follows the contract.)
"""

import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

from src.utils import log

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_IMAGE_SIZE = (256, 256)  # (height, width), used as a fallback for dynamic shapes


class Preprocessor:
    """Prepares images for a specific ONNX model based on its declared input signature."""

    def __init__(self, session: ort.InferenceSession, metadata: dict | None = None):
        input_meta = session.get_inputs()[0]
        self.input_name = input_meta.name
        self.onnx_dtype = input_meta.type
        self.onnx_shape = input_meta.shape

        metadata = metadata or {}
        normalize_inside_graph = str(metadata.get("normalize_inside_graph", "")).lower() == "true"

        # BGR->RGB swap: OpenCV decodes BGR, the graph is trained on RGB. Missing
        # key defaults to enabled (all our exporter's models), matching C++.
        color_conversion = metadata.get("preproc_color_conversion", "") or ""
        self.convert_bgr_to_rgb = color_conversion == "" or color_conversion == "bgr2rgb"

        self.image_size = self._resolve_image_size(self.onnx_shape)
        self.mode = self._resolve_mode(self.onnx_dtype, normalize_inside_graph)

        log(f"Model input '{self.input_name}': dtype={self.onnx_dtype}, shape={self.onnx_shape}")
        log(f"Preprocessing mode auto-selected: {self.mode} | target size (H, W): {self.image_size} "
            f"| bgr2rgb={self.convert_bgr_to_rgb}")
        if self.mode == "float32_chw_raw01":
            log("NOTE: float32 input with normalize_inside_graph=true (anomaly_export "
                "contract): host scales to [0,1] only; mean/std is applied INSIDE the "
                "graph. Host-side ImageNet normalization is intentionally skipped.")
        elif self.mode == "float32_chw_imagenet":
            log("NOTE: float32 input without normalize_inside_graph: applying host-side "
                "ImageNet mean/std (legacy pipeline). Verify this matches the model's export.")

    @staticmethod
    def _resolve_image_size(shape) -> tuple[int, int]:
        dims = [d for d in shape if isinstance(d, int) and d > 1]
        if len(dims) >= 2:
            return dims[-2], dims[-1]
        return DEFAULT_IMAGE_SIZE

    @staticmethod
    def _resolve_mode(dtype: str, normalize_inside_graph: bool) -> str:
        if "uint8" in dtype:
            return "uint8_hwc_ingraph"
        if normalize_inside_graph:
            return "float32_chw_raw01"
        return "float32_chw_imagenet"

    @staticmethod
    def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Resize to (w, h) with PIL's antialiased BILINEAR filter — matches the
        model's ``preproc_resize_interpolation=bilinear``/``preproc_antialias=true``
        metadata and the training pipeline. Expects a uint8 HWC array."""
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))

    def __call__(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image, returning a batched (1, ...) array."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if self.convert_bgr_to_rgb else img_bgr
        h, w = self.image_size
        img_resized = self._resize(img, w, h)

        if self.mode == "uint8_hwc_ingraph":
            tensor = img_resized.astype(np.uint8)
            return np.expand_dims(tensor, axis=0)  # (1, H, W, 3)

        img_float = img_resized.astype(np.float32) / 255.0
        # ImageNet mean/std ONLY when the graph does not normalize itself.
        if self.mode == "float32_chw_imagenet":
            img_float = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        img_chw = img_float.transpose(2, 0, 1)
        return np.expand_dims(img_chw, axis=0).astype(np.float32)  # (1, 3, H, W)

    def load_original_bgr(self, image_path: str) -> np.ndarray:
        """Load the original image (BGR, resized to the model's input frame) for
        visualization, using the identical spatial pipeline as __call__ so the
        heatmap overlay lines up with what the model saw."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = self.image_size
        return self._resize(img_bgr, w, h)
