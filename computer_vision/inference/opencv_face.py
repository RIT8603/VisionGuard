"""OpenCV face backends: YuNet detection and SFace embedding.

Both ship *as code* inside OpenCV (``cv2.FaceDetectorYN``,
``cv2.FaceRecognizerSF``) but need their ONNX weights supplied. OpenCV 5's
Python wheel no longer bundles any face model data, so there is no zero-setup
face detector - the paths must be configured. See the README for the two files
and where they come from.

Chosen over a heavier stack because they are small (~340 KB and ~37 MB), fast
on CPU, and add no dependency beyond the OpenCV already present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from computer_vision.inference.face_base import (
    FaceDetectorBackend,
    FaceEmbedderBackend,
    ModelLoadError,
    RawFace,
)
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)


def _require_model(path: str | None, what: str, hint: str) -> Path:
    if not path:
        raise ModelLoadError(
            f"{what} needs a model file, and OpenCV 5 bundles none.\n{hint}"
        )
    resolved = Path(path)
    if not resolved.is_file():
        raise ModelLoadError(
            f"{what} model not found: {resolved}\n"
            f"(resolved from working directory {Path.cwd()})\n{hint}"
        )
    return resolved


_YUNET_HINT = (
    "Download face_detection_yunet_2023mar.onnx (~340 KB) from the OpenCV Zoo\n"
    "  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet\n"
    "and set modules.face_detection.model to its path."
)

_SFACE_HINT = (
    "Download face_recognition_sface_2021dec.onnx (~37 MB) from the OpenCV Zoo\n"
    "  https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface\n"
    "and set modules.face_recognition.model to its path."
)


class YuNetFaceDetector(FaceDetectorBackend):
    """``cv2.FaceDetectorYN`` - a small, fast CNN face detector.

    Parameters:
        model: Path to the YuNet ONNX file. Required.
        score_threshold: Detector-side score cut. Default 0.6.
        nms_threshold: Overlap suppression. Default 0.3.
        top_k: Maximum candidates considered before NMS.
        detect_max_side: Shrink the frame so its longest side is at most this
            before detecting (0 = never). Detection is the only CPU-bound face
            step; on a 960x1707 frame, detecting at 960 is ~3x faster. Boxes
            and landmarks are scaled back, so recognition still crops from the
            full-resolution frame.
    """

    name = "yunet"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._detector = None
        self._input_size = (0, 0)
        self._score = float(self.params.get("score_threshold", 0.6))
        self._nms = float(self.params.get("nms_threshold", 0.3))
        self._top_k = int(self.params.get("top_k", 5000))
        self._max_side = int(self.params.get("detect_max_side", 0))

    def load(self) -> None:
        if self._loaded:
            return
        path = _require_model(self.params.get("model"), "Face detection", _YUNET_HINT)

        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), self._score, self._nms, self._top_k
            )
        except cv2.error as exc:
            raise ModelLoadError(f"Could not load YuNet model {path}: {exc}") from exc

        self._loaded = True
        logger.info("Loaded YuNet face detector from %s", path.name)

    def detect(self, frame: np.ndarray) -> List[RawFace]:
        if not self._loaded or self._detector is None:
            raise ModelLoadError("YuNetFaceDetector.detect() called before load()")

        image, scale = frame, 1.0
        longest = max(frame.shape[:2])
        if self._max_side and longest > self._max_side:
            scale = self._max_side / longest
            image = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        height, width = image.shape[:2]
        if (width, height) != self._input_size:
            # YuNet must be told the frame size before every differently
            # sized frame, or it silently returns nothing.
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, raw = self._detector.detect(image)
        if raw is None:
            return []
        if scale != 1.0:
            # Back to full-frame pixels: box and the five landmark points.
            raw = raw.copy()
            raw[:, :14] /= scale

        faces: List[RawFace] = []
        for row in raw:
            x, y, w, h = row[0:4]
            faces.append(
                RawFace(
                    confidence=float(row[14]),
                    xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    landmarks=np.array(row[4:14], dtype=np.float32).reshape(5, 2),
                    raw_row=np.asarray(row, dtype=np.float32),
                )
            )
        return faces

    def unload(self) -> None:
        self._detector = None
        self._input_size = (0, 0)
        super().unload()

    @property
    def description(self) -> str:
        return f"yunet:{Path(str(self.params.get('model', '?'))).name}"


class SFaceEmbedder(FaceEmbedderBackend):
    """``cv2.FaceRecognizerSF`` - produces a 128-d face embedding.

    Uses the detector's landmarks to align the crop first. Skipping alignment
    costs a lot of accuracy, which is why :class:`RawFace` carries the
    detector's own output row through.
    """

    name = "sface"
    #: OpenCV's documented cosine threshold for SFace.
    default_threshold = 0.363

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._recognizer = None

    def load(self) -> None:
        if self._loaded:
            return
        path = _require_model(self.params.get("model"), "Face recognition", _SFACE_HINT)

        try:
            self._recognizer = cv2.FaceRecognizerSF.create(str(path), "")
        except cv2.error as exc:
            raise ModelLoadError(f"Could not load SFace model {path}: {exc}") from exc

        self._loaded = True
        logger.info("Loaded SFace embedder from %s", path.name)

    def embed(self, frame: np.ndarray, face: RawFace) -> np.ndarray:
        if not self._loaded or self._recognizer is None:
            raise ModelLoadError("SFaceEmbedder.embed() called before load()")

        if face.raw_row is not None:
            aligned = self._recognizer.alignCrop(frame, face.raw_row.reshape(1, -1))
        else:
            # No landmarks: fall back to the raw box. Works, but less
            # accurately - warn once rather than silently degrading.
            aligned = _crop(frame, face.xyxy)
            if aligned.size == 0:
                raise ValueError("Face crop is empty")

        return np.asarray(self._recognizer.feature(aligned), dtype=np.float32).ravel()

    def unload(self) -> None:
        self._recognizer = None
        super().unload()

    @property
    def description(self) -> str:
        return f"sface:{Path(str(self.params.get('model', '?'))).name}"


def _crop(frame: np.ndarray, xyxy) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(width, int(xyxy[2]))
    y2 = min(height, int(xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]
