"""Face image quality score in 0..1, used to pick the best snapshot per track.

FindFace scores every tracked face and posts the best one; its scorer is a
proprietary neural net. This is a transparent stand-in combining the signals
that matter for downstream recognition: detector confidence, sharpness,
resolution and how frontal the face is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection


@dataclass
class Quality:
    score: float        # combined 0..1
    confidence: float
    sharpness: float    # 0..1
    size: float         # 0..1
    frontal: float      # 0..1
    yaw: float          # degrees, rough estimate from landmarks
    pitch: float        # degrees, rough estimate from landmarks


def estimate_pose(landmarks: np.ndarray) -> tuple[float, float]:
    """Very rough yaw/pitch (degrees) from YuNet's 5 landmarks."""
    eye_r, eye_l, nose, mouth_r, mouth_l = landmarks
    eye_mid = (eye_r + eye_l) / 2
    mouth_mid = (mouth_r + mouth_l) / 2
    half_eye = max(np.linalg.norm(eye_l - eye_r) / 2, 1e-3)
    yaw = math.degrees(math.asin(float(np.clip((nose[0] - eye_mid[0]) / half_eye, -1, 1))))
    span = max(mouth_mid[1] - eye_mid[1], 1e-3)
    ratio = (nose[1] - eye_mid[1]) / span          # ~0.55 for a frontal face
    pitch = float(np.clip((ratio - 0.55) / 0.35, -1, 1)) * 60.0
    return yaw, pitch


def sharpness(gray_crop: np.ndarray) -> float:
    if gray_crop.size == 0:
        return 0.0
    face = cv2.resize(gray_crop, (112, 112), interpolation=cv2.INTER_AREA)
    var = cv2.Laplacian(face, cv2.CV_64F).var()
    return float(1 - math.exp(-var / 150.0))       # saturates for crisp images


def assess(frame: np.ndarray, det: Detection) -> Quality:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det.bbox.astype(int)
    crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else crop
    sharp = sharpness(gray)
    size = float(min(1.0, min(det.width, det.height) / 112.0))
    yaw, pitch = estimate_pose(det.landmarks)
    frontal = float(max(0.0, math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))) ** 2)
    conf = float(np.clip(det.score, 0, 1))
    # Geometric mean: a face that is bad on any one axis scores low overall.
    score = (conf ** 0.25) * (sharp ** 0.25) * (size ** 0.2) * (frontal ** 0.3)
    return Quality(float(score), conf, sharp, size, frontal, yaw, pitch)
