"""YOLOv8-face detector backend (Ultralytics, runs on the GPU).

A single-shot detector trained on WIDER FACE with five landmarks per face, so
it drops into the same slot as YuNet. Compared with YuNet on real CCTV it
finds many more small, blurred and hazy faces and does it on the GPU instead
of the CPU. The landmarks come in the same order as YuNet's (image-left eye
first), so alignment for recognition works unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from computer_vision.inference.device import resolve_torch_device
from computer_vision.inference.face_base import FaceDetectorBackend, ModelLoadError, RawFace
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

_HINT = (
    "Get yolov8n-face.pt (~6 MB, YOLOv8 trained on WIDER FACE) from\n"
    "  https://github.com/akanametov/yolo-face/releases/tag/1.0.0\n"
    "(python -m tools.download_face_models fetches it) and set\n"
    "modules.face_detection.model to its path."
)


class YoloFaceDetector(FaceDetectorBackend):
    """``yolov8n-face`` through Ultralytics.

    Parameters:
        model: Path to the ``.pt`` file. Required.
        score_threshold: Detector-side score cut. Default 0.4.
        imgsz: Inference size. Default 960 (the processing width).
        device: ``auto``, ``cpu``, ``cuda`` ... as for the other models.
        half: FP16 on CUDA. Default true.
    """

    name = "yoloface"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._model = None
        self._score = float(self.params.get("score_threshold", 0.4))
        self._imgsz = int(self.params.get("imgsz", 960))
        self._device = resolve_torch_device(self.params.get("device", "auto"))
        self._half = bool(self.params.get("half", True)) and self._device.startswith("cuda")

    def load(self) -> None:
        if self._loaded:
            return
        path = self.params.get("model")
        if not path or not Path(path).is_file():
            raise ModelLoadError(f"YOLO face model not found: {path}\n{_HINT}")
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(path))
            # Build the graph now, not on the first live frame.
            self._model.predict(np.zeros((self._imgsz, self._imgsz, 3), np.uint8), imgsz=self._imgsz,
                                device=self._device, verbose=False)
        except Exception as exc:  # noqa: BLE001 - reported as a load failure
            raise ModelLoadError(f"Could not load YOLO face model {path}: {exc}") from exc
        self._loaded = True
        logger.info("Loaded YOLO face detector %s on %s (imgsz=%d)", Path(path).name, self._device, self._imgsz)

    def detect(self, frame: np.ndarray) -> List[RawFace]:
        if not self._loaded or self._model is None:
            raise ModelLoadError("YoloFaceDetector.detect() called before load()")

        result = self._model.predict(
            source=frame, conf=self._score, imgsz=self._imgsz, device=self._device,
            verbose=False, **({"quantize": 16} if self._half else {}),
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        points = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

        faces: List[RawFace] = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = (float(v) for v in box)
            landmarks = raw = None
            if points is not None and points.shape[1] >= 5:
                landmarks = np.asarray(points[i][:5], dtype=np.float32).reshape(5, 2)
                # YuNet's row layout (x, y, w, h, 5 x (px, py), score): what the
                # embedder's alignment reads.
                raw = np.concatenate([[x1, y1, x2 - x1, y2 - y1], landmarks.reshape(-1), [score]]).astype(np.float32)
            faces.append(RawFace(confidence=float(score), xyxy=(x1, y1, x2, y2), landmarks=landmarks, raw_row=raw))
        return faces

    def unload(self) -> None:
        self._model = None
        super().unload()

    @property
    def description(self) -> str:
        return f"yoloface:{Path(str(self.params.get('model', '?'))).name}@{self._device}"
