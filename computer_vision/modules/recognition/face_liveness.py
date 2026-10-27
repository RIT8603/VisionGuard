"""Face liveness: a basic check that a face is a live person, run before recognition.

**Basic, passive, heuristic.** It catches the easy attacks - a printed photo or
a still picture on a phone held up to the camera - and nothing more. A replayed
video of the person, or a good 3D mask, will pass. Real anti-spoofing needs a
trained model or a depth/IR camera; this is the MVP gate in front of
recognition, and it is deliberately reluctant to call a real person a spoof.

Each face is followed across frames (FindFace-style face tracker) and three
kinds of evidence are gathered:

* **Blink** - the eye regions briefly change appearance (lids close) while the
  rest of the face stays put. A photo never blinks. YuNet gives eye *centres*
  only, so this compares small eye patches against their running average.
* **3D motion** - when a real head moves, its five landmarks move *non-rigidly*
  (turning the head shifts the nose relative to the eyes). A flat photo, however
  it is moved, tilted or zoomed, moves all five by one 2D similarity transform.
  The residual of fitting that transform, relative to eye distance, is the cue.
* **Texture** - screens and prints add regular high-frequency patterns (moire,
  pixel grid, halftone). A strong isolated peak in the face crop's spectrum
  counts against the face. **Off by default**: measured on real CCTV footage,
  real faces scored as high as screens do (busy backgrounds, compression), so
  on its own it would call real people spoofs.

Thresholds were set from measurements: a photo moved, tilted and zoomed in
front of the camera reached a landmark residual of 0.04 and an eye-patch change
of 0.58, so 3D motion needs > 0.06 and a blink needs > 0.7 *and* the eyes to
look as before again within two frames (lids reopening).

Decision per face track:

* ``live``     - a blink was seen, or enough frames showed 3D motion.
* ``spoof``    - after ``decision_seconds`` of a clearly visible face, no blink
  and no 3D motion (or strong moire with no 3D motion).
* ``checking`` - not enough evidence yet. Recognition is *allowed* meanwhile:
  blocking everyone for the first seconds would make the system unusable.

A ``spoof_suspected`` event (an alert) is raised once per face judged
``spoof``. With ``block_spoof`` on, ``face_recognition`` also refuses to name
it. **Off by default**, because measured on real footage the passive cues do
not separate a still photo from a real person standing still well enough: two
real people in ~90 s of four test cameras were flagged, and at the 640 px
detection size a moved photo's landmark jitter (median 0.027) matched a still
real face's (0.032). Blocking would stop real students being recognised; as a
flag it is a useful "check this" alert. A trained anti-spoofing model is the
way to a dependable gate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from computer_vision.core.context import FrameContext, Severity
from computer_vision.modules.analysis.common import make_event, parse_severity
from computer_vision.modules.base import BaseAIModule, ModuleResult, ModuleType
from computer_vision.modules.recognition.face_tracker import FaceTracker

EVENT_SPOOF = "spoof_suspected"

LIVE, SPOOF, CHECKING = "live", "spoof", "checking"


@dataclass
class _FaceState:
    first_seen: float
    frames: int = 0
    landmarks: Optional[np.ndarray] = None
    eye_templates: Optional[List[np.ndarray]] = None
    blink: bool = False
    #: Frames since the eyes looked closed (None = not closed); a blink is
    #: confirmed only when they reopen soon after.
    closed_for: Optional[int] = None
    motion_hits: int = 0
    moire_hits: int = 0
    residuals: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    state: str = CHECKING
    reported: bool = False


def similarity_residual(previous: np.ndarray, current: np.ndarray) -> float:
    """RMS error of the best 2D similarity transform previous->current, per eye distance.

    Near zero for anything flat (a photo moved, tilted or zoomed); larger when
    the landmarks move relative to each other, as a real head does when it turns.
    """
    src = np.asarray(previous, np.float64)
    dst = np.asarray(current, np.float64)
    eye = float(np.linalg.norm(dst[1] - dst[0]))
    if eye < 1e-3:
        return 0.0
    # Umeyama / Procrustes for a similarity transform (rotation, scale, shift).
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rot = u @ sign @ vt
    var_s = (s ** 2).sum() / len(src)
    scale = (sig * np.diag(sign)).sum() / var_s if var_s > 0 else 1.0
    fitted = (scale * (rot @ s.T)).T + mu_d
    return float(np.sqrt(((fitted - dst) ** 2).sum(1).mean()) / eye)


def moire_score(face_gray: np.ndarray) -> float:
    """How much a single sharp periodic peak dominates the high frequencies.

    Natural skin and hair spread high-frequency energy smoothly; a screen's
    pixel grid or a print's halftone concentrates it in a few spikes.
    """
    if face_gray.size == 0:
        return 0.0
    img = cv2.resize(face_gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = (img - img.mean()) * np.hanning(128)[:, None] * np.hanning(128)[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    yy, xx = np.mgrid[-64:64, -64:64]
    radius = np.hypot(yy, xx)
    band = spectrum[(radius > 24) & (radius < 62)]
    if band.size == 0 or band.mean() <= 0:
        return 0.0
    return float(band.max() / (band.mean() + 1e-6))


def _eye_patches(gray: np.ndarray, landmarks: np.ndarray) -> Optional[List[np.ndarray]]:
    right, left = landmarks[0], landmarks[1]
    eye = float(np.linalg.norm(left - right))
    half = max(3, int(eye * 0.22))
    patches = []
    h, w = gray.shape[:2]
    for cx, cy in (right, left):
        x1, y1 = int(cx) - half, int(cy) - half
        x2, y2 = int(cx) + half, int(cy) + half
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return None
        patch = cv2.resize(gray[y1:y2, x1:x2], (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
        patch -= patch.mean()
        norm = np.linalg.norm(patch)
        patches.append(patch / norm if norm > 1e-3 else patch)
    return patches


class FaceLivenessModule(BaseAIModule):
    """Marks each detected face live / spoof / checking (see the module docstring).

    Config block::

        face_liveness:
          enabled: true
          decision_seconds: 6     # visible this long with no sign of life -> spoof
          min_face_size: 60       # smaller faces are left as 'checking'
          motion_residual: 0.06   # landmark non-rigidity that counts as 3D motion
          motion_frames: 3        # frames of 3D motion that prove a live face
          blink_drop: 0.7         # eye-patch change that counts as eyes closing
          moire_peak: 0           # spectral peak ratio that looks like a screen (0 = off)
          block_spoof: false      # true = face_recognition refuses to name a spoof
          severity: warning
    """

    module_name = "face_liveness"
    module_type = ModuleType.RECOGNITION
    version = "1.0.0"
    requires = ("face_detection",)

    def initialize(self) -> None:
        p = self.params
        self._decision = float(p.get("decision_seconds", 6.0))
        self._min_size = float(p.get("min_face_size", 60))
        self._motion_residual = float(p.get("motion_residual", 0.06))
        self._motion_frames = int(p.get("motion_frames", 3))
        self._blink_drop = float(p.get("blink_drop", 0.7))
        self._moire_peak = float(p.get("moire_peak", 0.0))
        self._block = bool(p.get("block_spoof", False))
        self._severity = parse_severity(p, Severity.WARNING)
        self._tracker = FaceTracker(
            overlap_threshold=0.25, miss_interval=1.5,
            center_gate=float(p.get("track_center_gate", 0.0)),
        )
        self._faces: Dict[int, _FaceState] = {}
        self.log.info(
            "Ready: decide after %.0fs, 3D motion > %.3f for %d frames, blink drop %.2f, "
            "block spoofs: %s", self._decision, self._motion_residual, self._motion_frames,
            self._blink_drop, self._block,
        )

    def process(self, context: FrameContext) -> ModuleResult:
        faces = context.get_detections("face_detection")
        now = context.monotonic_ts
        ids, closed = self._tracker.update(
            [(d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2) for d in faces], context.frame_id, now
        )
        for face_id in closed:
            self._faces.pop(face_id, None)

        gray = cv2.cvtColor(context.frame, cv2.COLOR_BGR2GRAY) if faces else None
        counts = {LIVE: 0, SPOOF: 0, CHECKING: 0}
        events = []
        for det, face_id in zip(faces, ids):
            raw = det.extra.get("face")
            if raw is None or raw.landmarks is None or face_id < 0:
                continue
            state = self._faces.setdefault(face_id, _FaceState(first_seen=now))
            self._observe(state, gray, raw, det)
            self._decide(state, now)
            counts[state.state] += 1
            det.extra["liveness"] = {
                "state": state.state,
                "blink": state.blink,
                "motion_frames": state.motion_hits,
                "block": self._block,
            }
            if state.state == SPOOF and not state.reported:
                state.reported = True
                events.append(make_event(
                    self.module_name, EVENT_SPOOF, context, self._severity,
                    subject={"class": "face", "face_track_id": face_id},
                    data={
                        "seconds_observed": round(now - state.first_seen, 1),
                        "moire_frames": state.moire_hits,
                        "blocked": self._block,
                    },
                ))

        for event in events:
            context.add_event(event)
            self.log.info("%s (face track %s)", event.event_type, event.subject["face_track_id"])
        return self.make_result(context, data={"faces": counts, "events": [e.to_dict() for e in events]})

    # ------------------------------------------------------------------ #

    def _observe(self, state: _FaceState, gray: np.ndarray, raw: Any, det: Any) -> None:
        size = min(det.bbox.width, det.bbox.height)
        if size < self._min_size:
            return  # too small to judge; stays 'checking'
        state.frames += 1
        landmarks = np.asarray(raw.landmarks, np.float32)

        if state.landmarks is not None:
            residual = similarity_residual(state.landmarks, landmarks)
            state.residuals.append(residual)
            if residual > self._motion_residual:
                state.motion_hits += 1
        state.landmarks = landmarks

        patches = _eye_patches(gray, landmarks)
        if patches is not None:
            if state.eye_templates is None:
                state.eye_templates = patches
            else:
                drops = [1.0 - float((p * t).sum()) for p, t in zip(patches, state.eye_templates)]
                still = not state.residuals or state.residuals[-1] < self._motion_residual * 2
                if state.closed_for is not None:
                    state.closed_for += 1
                    if max(drops) < self._blink_drop / 2:
                        state.blink = True          # closed, then open again: a blink
                        state.closed_for = None
                    elif state.closed_for > 2:
                        state.closed_for = None     # stayed different: a pose change, not a blink
                # Both eyes change together in a blink; one eye alone is glare or a hand.
                elif min(drops) > self._blink_drop and still:
                    state.closed_for = 0
                if state.closed_for is None:
                    # Only open-eye frames update the "eyes as usual" template.
                    state.eye_templates = [0.8 * t + 0.2 * p for t, p in zip(state.eye_templates, patches)]

        if self._moire_peak > 0:
            x1, y1 = int(max(0, det.bbox.x1)), int(max(0, det.bbox.y1))
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
            if moire_score(gray[y1:y2, x1:x2]) > self._moire_peak:
                state.moire_hits += 1

    def _decide(self, state: _FaceState, now: float) -> None:
        if state.state == LIVE:
            return  # once proven live, a still moment does not undo it
        if state.blink or state.motion_hits >= self._motion_frames:
            state.state = LIVE
            return
        observed = now - state.first_seen
        enough_frames = state.frames >= 5
        screen_like = self._moire_peak > 0 and state.moire_hits >= max(3, state.frames // 2)
        if enough_frames and (observed >= self._decision or (screen_like and observed >= self._decision / 2)):
            state.state = SPOOF

    def cleanup(self) -> None:
        if hasattr(self, "_faces"):
            self._faces.clear()
            self._tracker.reset()
        super().cleanup()
