"""Face detector (OpenCV YuNet) returning boxes + 5 landmarks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray        # x1, y1, x2, y2 (float, frame coordinates)
    score: float            # detector confidence 0..1
    landmarks: np.ndarray   # (5, 2): right eye, left eye, nose, mouth R, mouth L

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class FaceDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, resize_scale: float = 1.0):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        self.resize_scale = resize_scale
        self._net = cv2.FaceDetectorYN.create(model_path, "", (320, 320),
                                             score_threshold, nms_threshold, 5000)
        self._size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        img = frame
        if self.resize_scale != 1.0:
            img = cv2.resize(frame, None, fx=self.resize_scale, fy=self.resize_scale)
        h, w = img.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._net.detect(img)
        if faces is None:
            return []
        inv = 1.0 / self.resize_scale
        ox, oy = offset
        out = []
        for f in faces:
            x, y, bw, bh = f[:4] * inv
            lm = f[4:14].reshape(5, 2) * inv + (ox, oy)
            out.append(Detection(np.array([x + ox, y + oy, x + ox + bw, y + oy + bh], dtype=np.float32),
                                 float(f[14]), lm.astype(np.float32)))
        return out
