"""Motion gate: skip the (expensive) detector on static frames.

FindFace runs a motion detector first and only wakes the face tracker on
"motion of certain intensity" (imotion_threshold, default 0.002).
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(self, threshold: float = 0.002, scale: float = 0.25, pixel_delta: int = 25):
        self.threshold = threshold
        self.scale = scale
        self.pixel_delta = pixel_delta
        self._prev: np.ndarray | None = None
        self.last_intensity = 0.0

    def __call__(self, frame: np.ndarray) -> bool:
        """Return True if the frame has enough motion to run detection."""
        if self.threshold <= 0:
            return True
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        prev, self._prev = self._prev, gray
        if prev is None or prev.shape != gray.shape:
            self.last_intensity = 1.0
            return True
        changed = cv2.absdiff(prev, gray) > self.pixel_delta
        self.last_intensity = float(np.count_nonzero(changed)) / changed.size
        return self.last_intensity >= self.threshold
