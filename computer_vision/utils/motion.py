"""Motion gate: tell whether a frame changed enough to be worth re-analysing.

Ported from the FindFace-lite reference (``FindFace/findface_lite/motion.py``).
FindFace Multi runs a motion detector before its face detector and only wakes
the face tracker on "motion of certain intensity" (``imotion_threshold``,
default 0.002 = 0.2 % of pixels changed).
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    """Frame differencing on a small, blurred greyscale copy."""

    def __init__(self, threshold: float = 0.002, scale: float = 0.25, pixel_delta: int = 25) -> None:
        self.threshold = float(threshold)
        self.scale = float(scale)
        self.pixel_delta = int(pixel_delta)
        self._prev: np.ndarray | None = None
        #: Fraction of pixels that changed in the last frame looked at.
        self.last_intensity = 0.0

    def __call__(self, frame: np.ndarray) -> bool:
        """True when the frame moved enough (or the gate is off, or it is the first)."""
        if self.threshold <= 0:
            return True
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        prev, self._prev = self._prev, gray
        if prev is None or prev.shape != gray.shape:
            self.last_intensity = 1.0
            return True
        changed = cv2.absdiff(prev, gray) > self.pixel_delta
        self.last_intensity = float(np.count_nonzero(changed)) / changed.size
        return self.last_intensity >= self.threshold

    def reset(self) -> None:
        self._prev = None
        self.last_intensity = 0.0
