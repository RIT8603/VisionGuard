"""How good is a detected face for *recognising* it?

A face embedding is only as reliable as the picture it came from. Running the
recogniser on a blurred, tiny or half-turned face does not just fail to match -
it produces a confident-looking vector that points at the wrong person. So
recognition (and enrolment) first asks this module whether the face is worth
embedding at all.

Four independent measures, cheap enough to run on every detected face:

* **size**       - the shorter side of the face box, in pixels;
* **sharpness**  - variance of the Laplacian over the face crop (motion blur,
                   defocus and heavy compression all flatten it);
* **frontality** - head yaw from the five landmarks (a profile hides half the
                   features the embedder relies on);
* **confidence** - the detector's own score.

They are combined into one 0..1 ``score`` so a track can remember its best
snapshot and only re-recognise when a clearly better one arrives. The score
and the angle estimate are ported from the FindFace-lite reference
(``FindFace/findface_lite/quality.py``), which mirrors how FindFace Multi picks
a track's best snapshot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

#: Face size (px, shorter side) at which the size term saturates. SFace is
#: trained on 112 px crops, so larger than that adds nothing.
SIZE_SATURATION_PX = 112.0
#: Laplacian variance scale of the sharpness term (1 - exp(-var / scale)).
SHARPNESS_SCALE = 150.0
#: Pitch ratio of a face looking straight at the camera (measured median 0.53).
PITCH_NEUTRAL = 0.55


def estimate_pose_angles(landmarks: np.ndarray) -> Tuple[float, float, float]:
    """``(yaw, pitch, roll_degrees)`` from YuNet's five landmarks.

    Landmark order is right eye, left eye, nose tip, right mouth corner, left
    mouth corner, where "right" is the right of the *image*.

    ``yaw`` = (nose.x - eye-midpoint.x) / eye distance; positive means the nose
    points to the right of the image, i.e. the person turned to *their* left.
    ``pitch`` = (nose.y - eye-midpoint.y) / (mouth.y - eye-midpoint.y); smaller
    is chin up, larger is chin down.
    """
    right_eye, left_eye, nose, right_mouth, left_mouth = (
        np.asarray(p, dtype=np.float32) for p in landmarks
    )
    eye_mid = (right_eye + left_eye) / 2.0
    mouth_mid = (right_mouth + left_mouth) / 2.0
    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    if eye_dist < 1e-3:
        return 0.0, 0.5, 0.0

    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    vertical = float(mouth_mid[1] - eye_mid[1])
    pitch = float((nose[1] - eye_mid[1]) / vertical) if abs(vertical) > 1e-3 else 0.5
    roll = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))
    return yaw, pitch, roll


def estimate_pose_degrees(landmarks: np.ndarray) -> Tuple[float, float]:
    """Rough ``(yaw, pitch)`` in degrees from the five landmarks (FindFace-lite).

    Yaw is the arcsine of the nose's offset from the eye midpoint relative to
    half the eye distance, so it reaches 90 degrees in full profile. Pitch maps
    the nose's height between eyes and mouth onto +-60 degrees.
    """
    right_eye, left_eye, nose, right_mouth, left_mouth = (
        np.asarray(p, dtype=np.float32) for p in landmarks
    )
    eye_mid = (right_eye + left_eye) / 2.0
    mouth_mid = (right_mouth + left_mouth) / 2.0
    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    if eye_dist < 1e-3:
        return 0.0, 0.0  # no usable landmarks: no pose information
    half_eye = eye_dist / 2.0
    yaw = math.degrees(math.asin(float(np.clip((nose[0] - eye_mid[0]) / half_eye, -1.0, 1.0))))
    span = max(float(mouth_mid[1] - eye_mid[1]), 1e-3)
    ratio = float(nose[1] - eye_mid[1]) / span
    pitch = float(np.clip((ratio - PITCH_NEUTRAL) / 0.35, -1.0, 1.0)) * 60.0
    return yaw, pitch


def face_sharpness(face_bgr: np.ndarray) -> float:
    """Variance of the Laplacian of the face crop, at a fixed 112x112 size."""
    if face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    gray = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass(frozen=True)
class FaceQuality:
    """The measurements for one face, and what to do with them."""

    size_px: float
    #: Raw variance of the Laplacian (the gate compares against this).
    sharpness: float
    yaw: float          # degrees
    pitch: float        # degrees
    confidence: float
    #: 0..1 terms of the score.
    sharp_term: float
    size_term: float
    frontal: float
    score: float
    #: Why the face is not fit to recognise; empty when it is.
    problems: Tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return not self.problems


def assess_face(
    frame: np.ndarray,
    xyxy: Tuple[float, float, float, float],
    landmarks: Optional[np.ndarray],
    confidence: float,
    *,
    min_size: float = 60.0,
    max_angle: float = 45.0,
    min_sharpness: float = 20.0,
    min_score: float = 0.0,
) -> FaceQuality:
    """Measure a face and decide whether it is good enough to embed.

    The score is FindFace-lite's geometric mean, so a face that is bad on any
    one axis - blurred, tiny, or turned - scores low overall, however good it
    is on the others::

        score = conf^0.25 * sharp^0.25 * size^0.2 * frontal^0.3

    with ``frontal = (cos(yaw) * cos(pitch))^2``. 0.42 is FindFace's reference
    minimum.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    size = float(min(x2 - x1, y2 - y1))

    crop = frame[
        max(0, int(y1)) : min(height, int(y2)), max(0, int(x1)) : min(width, int(x2))
    ]
    sharp = face_sharpness(crop)
    yaw, pitch = estimate_pose_degrees(landmarks) if landmarks is not None else (0.0, 0.0)

    conf = float(np.clip(confidence, 0.0, 1.0))
    sharp_term = 1.0 - math.exp(-sharp / SHARPNESS_SCALE)
    size_term = min(1.0, max(size, 0.0) / SIZE_SATURATION_PX)
    frontal = max(0.0, math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))) ** 2
    score = (conf ** 0.25) * (sharp_term ** 0.25) * (size_term ** 0.2) * (frontal ** 0.3)

    problems = []
    if size < min_size:
        problems.append("small")
    if sharp < min_sharpness:
        problems.append("blurry")
    if max(abs(yaw), abs(pitch)) > max_angle:
        problems.append("turned")
    if score < min_score:
        problems.append("low-quality")

    return FaceQuality(
        size_px=size,
        sharpness=sharp,
        yaw=round(yaw, 2),
        pitch=round(pitch, 2),
        confidence=conf,
        sharp_term=round(sharp_term, 4),
        size_term=round(size_term, 4),
        frontal=round(frontal, 4),
        score=round(score, 4),
        problems=tuple(problems),
    )
