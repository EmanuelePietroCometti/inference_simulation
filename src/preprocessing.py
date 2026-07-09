"""
Image preprocessing.

The exact preprocessing required depends on how the ONNX model was exported:
  - Newer export pipeline: input is uint8 HWC, RGB, normalization is done inside the graph.
  - Older / default pipeline: input is float32 CHW, normalized here using ImageNet
    statistics (matches the training preprocessing in datamodules/base/datamodule.py).

Since both variants exist across export scripts used in this project, the Preprocessor
inspects the ONNX model's input dtype at runtime and automatically selects the matching
mode, logging the decision so it is always visible which one was used for a given run.

Resize / anti-aliasing
-----------------------
All training pipelines (SuperSimpleNet's ``F.interpolate(..., antialias=False)`` inside
the graph aside, SK-RD4AD's ``torchvision.transforms.v2.Resize(antialias=True)``) resize
with a low-pass filter applied before subsampling. Inspection-camera images are typically
much larger than the model's 256x256 input, so a naive ``cv2.resize`` with the default
``INTER_LINEAR`` interpolation aliases high-frequency, periodic content (e.g. woven fabric)
into a moire pattern the model never saw in training - it then reports the whole aliased
region as anomalous instead of the real defect. ``INTER_AREA`` (OpenCV's recommended mode
for shrinking) area-averages pixels before subsampling, which closely approximates an
anti-aliased resize and removes this artifact; ``INTER_LINEAR`` stays correct for the (rare)
upscaling case, where there is no aliasing to guard against.

Dynamic object crop (SK-RD4AD only)
-------------------------------------
SK-RD4AD's training loop ALWAYS applies an extra step after the initial resize
(main.py: "Apply Dynamic Crop (Always active to normalize object scale)",
test.py's ``apply_dynamic_crop_gpu``, also used by its own AUROC evaluation in
test.py's ``evaluation_me``/``evaluation``): find the bounding box of non-background
pixels (mean intensity below a threshold, i.e. anything that isn't near-white),
pad it, and rescale that crop to fill the full frame. This normalizes the object's
scale/framing in every training example. If the raw inference image has a
different amount of background margin than the crop assumed at training, the
model receives an object at a scale/framing it never saw and the reconstruction
degrades uniformly across the whole image - not a localized defect signal. This
was diagnosed after raw scores came back nearly identical (~0.999) for every
image regardless of content, which is the signature of the whole input being
out-of-distribution rather than a specific defect being detected. Controlled by
the model's embedded ``dynamic_crop`` / ``dynamic_crop_bg_threshold`` /
``dynamic_crop_padding`` metadata (see src/model_config.py); a no-op for
architectures that don't declare it (e.g. SuperSimpleNet never crops).
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

    def __init__(self, session: ort.InferenceSession, metadata: dict | None = None):
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

        metadata = metadata or {}
        self.dynamic_crop_enabled = metadata.get("dynamic_crop") == "true"
        self.dynamic_crop_threshold = float(metadata.get("dynamic_crop_bg_threshold", 0.94))
        self.dynamic_crop_padding = int(metadata.get("dynamic_crop_padding", 30))
        if self.dynamic_crop_enabled:
            log(f"Dynamic object crop ENABLED (bg_threshold={self.dynamic_crop_threshold}, "
                f"padding={self.dynamic_crop_padding}px) - required to match this model's "
                f"training preprocessing (see module docstring).")

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

    @staticmethod
    def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Resize to (w, h), using an area-averaging filter when shrinking to avoid
        aliasing on high-frequency/periodic content (see module docstring)."""
        src_h, src_w = img.shape[:2]
        shrinking = w < src_w or h < src_h
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return cv2.resize(img, (w, h), interpolation=interpolation)

    @classmethod
    def _dynamic_crop(cls, img: np.ndarray, threshold: float, padding: int) -> np.ndarray:
        """Replicate SK-RD4AD's apply_dynamic_crop_gpu (test.py): crop to the
        bounding box of non-background pixels (mean intensity below `threshold`
        in [0,1]), pad, and resize back to the original frame size. See module
        docstring for why this must match training exactly. Works on any HWC
        array (uint8 or float) - the crop indices are dtype-independent, only
        the background test needs a normalized [0,1] view.
        """
        h, w = img.shape[:2]
        gray01 = (img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)).mean(axis=2)
        ys, xs = np.nonzero(gray01 < threshold)
        if ys.size == 0:
            return img  # nothing below threshold (e.g. blank frame): leave untouched

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        size = max(y_max - y_min, x_max - x_min)
        cy, cx = y_min + (y_max - y_min) // 2, x_min + (x_max - x_min) // 2

        y1, y2 = max(cy - size // 2 - padding, 0), min(cy + size // 2 + padding, h)
        x1, x2 = max(cx - size // 2 - padding, 0), min(cx + size // 2 + padding, w)

        cropped = img[y1:y2, x1:x2]
        return cls._resize(cropped, w, h)

    def _resize_and_crop(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Full spatial pipeline shared by the model input and the display image:
        resize to the model's frame size, then (if the model requires it) the
        dynamic object crop. Both must use this same helper so the heatmap
        overlay stays aligned with what the model actually saw."""
        resized = self._resize(img, w, h)
        if self.dynamic_crop_enabled:
            resized = self._dynamic_crop(resized, self.dynamic_crop_threshold, self.dynamic_crop_padding)
        return resized

    def __call__(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image, returning a batched (1, ...) array."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = self.image_size
        img_resized = self._resize_and_crop(img_rgb, w, h)

        if self.mode == "uint8_hwc_ingraph":
            tensor = img_resized.astype(np.uint8)
            return np.expand_dims(tensor, axis=0)  # (1, H, W, 3)

        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        img_chw = img_norm.transpose(2, 0, 1)
        return np.expand_dims(img_chw, axis=0).astype(np.float32)  # (1, 3, H, W)

    def load_original_bgr(self, image_path: str) -> np.ndarray:
        """Load the original image (BGR, resized/cropped to the model's input
        frame) for visualization - uses the identical spatial pipeline as
        __call__ so the heatmap overlay lines up with what the model saw."""
        img_bgr = cv2.imread(str(image_path))
        h, w = self.image_size
        return self._resize_and_crop(img_bgr, w, h)
