"""Face registry: users, multi-angle enrolment, and the live recognition store.

This is the write side of face recognition. The pipeline's
``face_recognition`` module only *reads* faces; everything that creates an
identity lives here, behind an explicit operator action (Agent.md §14, §24):

* **Users** are plain records (name, employee id, role, ...) in SQLite.
* **Enrolment** takes photos of one user from several angles - front, left,
  right, up, down - detects the face, checks it is good enough to learn from,
  checks the head really is turned the way the step asks, and stores only the
  resulting embedding. No image is ever kept.
* **The live store** (``computer_vision.identity.shared``) is rebuilt from the
  database after every change, so a user registered a moment ago is recognised
  on the next frame without restarting anything.

Accuracy is mostly decided here, at enrolment: a template made from a blurry,
tiny, half-hidden or wrongly-posed photo poisons every later match. So a
capture is rejected - with a reason the operator can act on - unless the face
is large, sharp, evenly lit, fully in frame, alone, and in the requested pose.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.persistence.face_db import (
    FaceDatabase,
    FaceDbError,
    UserRow,
    clean_fields,
)
from computer_vision.core.config import VisionGuardConfig
from computer_vision.identity.quality import estimate_pose_angles, face_sharpness
from computer_vision.identity.shared import get_shared_store, replace_shared_store
from computer_vision.inference.face_base import BackendError, RawFace, cosine_similarity
from computer_vision.inference.face_factory import create_face_detector, create_face_embedder
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DETECTOR_MODEL = "computer_vision/Model/face/face_detection_yunet_2023mar.onnx"
DEFAULT_EMBEDDER_MODEL = "computer_vision/Model/face/face_recognition_sface_2021dec.onnx"

#: Views captured for every user, in the order the wizard walks through them.
POSES: Tuple[str, ...] = ("front", "left", "right", "up", "down")
#: The minimum for a user to count as fully enrolled.
REQUIRED_POSES: Tuple[str, ...] = ("front", "left", "right")
POSE_LABELS = {
    "front": "Look straight at the camera",
    "left": "Turn your head to your left",
    "right": "Turn your head to your right",
    "up": "Tilt your chin up slightly",
    "down": "Tilt your chin down slightly",
}

SAMPLES_PER_POSE = 2
MAX_SAMPLES_PER_POSE = 5
MAX_IMAGE_BYTES = 8 * 1024 * 1024
#: Frames are shrunk to this longest side before detection: faster, and a face
#: larger than this is far more than recognition needs.
MAX_SIDE = 1280

# ---- capture quality gates ------------------------------------------------ #
MIN_FACE_PX = 90          # shorter side of the face box
MIN_DET_SCORE = 0.80
MIN_SHARPNESS = 25.0      # variance of the Laplacian over the 112px face crop
BRIGHTNESS_RANGE = (45.0, 215.0)
MULTI_FACE_RATIO = 0.45   # a second face this fraction of the first's area = ambiguous

# ---- pose, from the five landmarks ---------------------------------------- #
# yaw   = (nose.x - eye-midpoint.x) / eye distance. Positive = nose towards the
#         right of the *image*, i.e. the person turned to *their* left.
# pitch = (nose.y - eye-midpoint.y) / (mouth.y - eye-midpoint.y). Smaller =
#         chin up, larger = chin down.
YAW_TURNED = 0.20
# Past this the head is in profile: the landmarks collapse and the embedding is
# unreliable, so it is neither a good "left" nor "right" capture.
YAW_PROFILE = 0.80
YAW_FRONT = 0.15
# Measured on 1,300 real detected faces: a head facing the camera sits at
# pitch 0.47-0.61 (median 0.53), so up/down start just outside that band.
PITCH_UP = 0.44
PITCH_DOWN = 0.63
PITCH_SIDE_TOLERANCE = 0.30   # up/down only counts while the head is not also turned

# ---- identity integrity --------------------------------------------------- #
#: A new template must resemble the user's existing ones at least this much,
#: or it is probably a different person.
SAME_PERSON_FLOOR = 0.20


class FaceServiceError(FaceDbError):
    """A rejected request; ``status`` is the HTTP code that fits it."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without any model)
# --------------------------------------------------------------------------- #

def classify_pose(yaw: float, pitch: float) -> str:
    """Which enrolment view this head orientation is (``other`` if between two)."""
    if yaw >= YAW_TURNED:
        return "left" if yaw <= YAW_PROFILE else "other"
    if yaw <= -YAW_TURNED:
        return "right" if yaw >= -YAW_PROFILE else "other"
    if abs(yaw) <= PITCH_SIDE_TOLERANCE:
        if pitch < PITCH_UP:
            return "up"
        if pitch > PITCH_DOWN:
            return "down"
    if abs(yaw) <= YAW_FRONT:
        return "front"
    return "other"


def decode_image(data: bytes) -> np.ndarray:
    """Decode an uploaded JPEG/PNG into a BGR frame, shrunk to ``MAX_SIDE``."""
    if not data:
        raise FaceServiceError("No image was sent")
    if len(data) > MAX_IMAGE_BYTES:
        raise FaceServiceError("That image is too large (limit 8 MB)", 413)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise FaceServiceError("That file is not a readable JPEG or PNG image")
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        frame = cv2.resize(
            frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )
    return frame


@dataclass
class FaceReading:
    """What was found in one image, and whether it is fit to enrol from."""

    face: Optional[RawFace]
    width: int
    height: int
    count: int
    pose: Optional[str] = None
    yaw: float = 0.0
    pitch: float = 0.5
    roll: float = 0.0
    size_px: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    problems: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.face is not None and not self.problems

    @property
    def quality(self) -> float:
        """0..1, for storing alongside a template and showing the operator."""
        if self.face is None:
            return 0.0
        size = min(1.0, self.size_px / (MIN_FACE_PX * 2))
        sharp = min(1.0, self.sharpness / (MIN_SHARPNESS * 4))
        return round(0.4 * float(self.face.confidence) + 0.3 * size + 0.3 * sharp, 3)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "image": {"width": self.width, "height": self.height},
            "faces": self.count,
            "ok": self.ok,
            "problems": list(self.problems),
            "face": None,
        }
        if self.face is not None:
            x1, y1, x2, y2 = self.face.xyxy
            out["face"] = {
                # Normalised, so the browser can draw it over any preview size.
                "box": [
                    round(x1 / self.width, 4),
                    round(y1 / self.height, 4),
                    round(x2 / self.width, 4),
                    round(y2 / self.height, 4),
                ],
                "confidence": round(float(self.face.confidence), 3),
                "pose": self.pose,
                "yaw": round(self.yaw, 3),
                "pitch": round(self.pitch, 3),
                "size_px": round(self.size_px, 1),
                "sharpness": round(self.sharpness, 1),
                "brightness": round(self.brightness, 1),
                "quality": self.quality,
            }
        return out


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #

class FaceService:
    """Everything the dashboard can do to the face registry."""

    def __init__(
        self,
        db: FaceDatabase,
        *,
        detector_params: Dict[str, Any] | None = None,
        embedder_params: Dict[str, Any] | None = None,
        threshold: float = 0.363,
        editable: bool = True,
    ) -> None:
        self.db = db
        self.editable = editable
        self.threshold = float(threshold)
        self._detector_params = dict(detector_params or {})
        self._embedder_params = dict(embedder_params or {})
        self._detector_params.setdefault("model", DEFAULT_DETECTOR_MODEL)
        self._embedder_params.setdefault("model", DEFAULT_EMBEDDER_MODEL)
        # Enrolment wants only clear faces: stricter than the live detector.
        self._detector_params["score_threshold"] = max(
            float(self._detector_params.get("score_threshold", 0.6)), 0.7
        )

        self._detector = None
        self._embedder = None
        self._model_error: Optional[str] = None
        # YuNet keeps per-frame state (input size), and one OpenCV net must not
        # be driven by two requests at once.
        self._infer_lock = threading.Lock()

        self.rebuild_live_store()

    @classmethod
    def from_config(cls, config: VisionGuardConfig) -> "FaceService":
        modules = config.modules
        det = modules.get("face_detection")
        rec = modules.get("face_recognition")
        det_params = dict(det.params) if det is not None else {}
        rec_params = dict(rec.params) if rec is not None else {}
        return cls(
            FaceDatabase(config.dashboard.face_db),
            detector_params=det_params,
            embedder_params=rec_params,
            threshold=float(rec_params.get("threshold", 0.363)),
            editable=config.dashboard.allow_face_edit,
        )

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #

    def _ensure_models(self) -> None:
        """Load both models on first use. Raises :class:`FaceServiceError`."""
        if self._detector is not None and self._embedder is not None:
            return
        try:
            detector = create_face_detector("yunet", self._detector_params)
            detector.load()
            embedder = create_face_embedder("sface", self._embedder_params)
            embedder.load()
        except BackendError as exc:
            self._model_error = str(exc)
            raise FaceServiceError(
                "The face models are not installed. Run "
                "'python -m tools.download_face_models' once, then try again.",
                503,
            ) from exc
        self._detector, self._embedder = detector, embedder
        self._model_error = None
        logger.info("Face registry models loaded")

    def models_status(self) -> Dict[str, Any]:
        def entry(params: Dict[str, Any]) -> Dict[str, Any]:
            path = Path(str(params.get("model", "")))
            return {"path": str(path), "present": path.is_file()}

        detection = entry(self._detector_params)
        recognition = entry(self._embedder_params)
        return {
            "detection": detection,
            "recognition": recognition,
            "ready": bool(detection["present"] and recognition["present"]),
            "error": self._model_error,
        }

    # ------------------------------------------------------------------ #
    # Live store
    # ------------------------------------------------------------------ #

    def rebuild_live_store(self) -> None:
        """Make the running recognition match the database exactly."""
        replace_shared_store(self.db.active_identities(), self.threshold)
        store = get_shared_store()
        logger.info(
            "Live face store: %d identities, %d templates", len(store), store.sample_count
        )

    # ------------------------------------------------------------------ #
    # Status and users
    # ------------------------------------------------------------------ #

    def status(self) -> Dict[str, Any]:
        users = self.db.list_users()
        complete = [u for u in users if self._complete(u)]
        return {
            "editable": self.editable,
            "models": self.models_status(),
            "poses": [
                {
                    "id": pose,
                    "label": POSE_LABELS[pose],
                    "required": pose in REQUIRED_POSES,
                }
                for pose in POSES
            ],
            "samples_per_pose": SAMPLES_PER_POSE,
            "max_samples_per_pose": MAX_SAMPLES_PER_POSE,
            "threshold": self.threshold,
            "users": len(users),
            "enrolled": len(complete),
            "templates": sum(u.sample_count for u in users),
        }

    @staticmethod
    def _complete(user: UserRow) -> bool:
        return all(user.poses.get(pose, 0) > 0 for pose in REQUIRED_POSES)

    def _user_dict(self, user: UserRow) -> Dict[str, Any]:
        data = user.to_dict()
        data["enrolled"] = self._complete(user)
        data["missing_poses"] = [p for p in REQUIRED_POSES if user.poses.get(p, 0) == 0]
        return data

    def list_users(self) -> List[Dict[str, Any]]:
        return [self._user_dict(u) for u in self.db.list_users()]

    def get_user(self, user_id: int) -> Dict[str, Any]:
        return self._user_dict(self.db.get_user(user_id))

    def _require_editable(self) -> None:
        if not self.editable:
            raise FaceServiceError(
                "Editing the face registry is turned off (dashboard.allow_face_edit)", 403
            )

    def create_user(self, body: Dict[str, Any]) -> Dict[str, Any]:
        self._require_editable()
        fields = clean_fields(body, require_name=True)
        user = self.db.create_user(fields)
        logger.info("Face registry: user %d '%s' created", user.id, user.name)
        return self._user_dict(user)

    def update_user(self, user_id: int, body: Dict[str, Any]) -> Dict[str, Any]:
        self._require_editable()
        fields = clean_fields(body, require_name=False)
        active = body.get("active")
        if active is not None and not isinstance(active, bool):
            raise FaceServiceError("'active' must be true or false")
        user = self.db.update_user(user_id, fields, active)
        # A rename or an activate/deactivate changes who can be recognised.
        self.rebuild_live_store()
        return self._user_dict(user)

    def delete_user(self, user_id: int) -> None:
        self._require_editable()
        user = self.db.get_user(user_id)
        self.db.delete_user(user_id)
        self.rebuild_live_store()
        logger.info("Face registry: user %d '%s' deleted with all templates", user.id, user.name)

    def clear_samples(self, user_id: int, pose: Optional[str]) -> Dict[str, Any]:
        self._require_editable()
        if pose is not None and pose not in POSES:
            raise FaceServiceError(f"Unknown pose '{pose}'")
        self.db.get_user(user_id)
        self.db.delete_samples(user_id, pose)
        self.rebuild_live_store()
        return self.get_user(user_id)

    # ------------------------------------------------------------------ #
    # Reading an image
    # ------------------------------------------------------------------ #

    def _read(self, frame: np.ndarray) -> FaceReading:
        """Detect faces in ``frame`` and judge the main one. Caller holds no lock."""
        height, width = frame.shape[:2]
        with self._infer_lock:
            self._ensure_models()
            faces = [f for f in self._detector.detect(frame) if f.confidence >= 0.5]

        if not faces:
            return FaceReading(None, width, height, 0, problems=("No face found - face the camera",))

        faces.sort(
            key=lambda f: (f.xyxy[2] - f.xyxy[0]) * (f.xyxy[3] - f.xyxy[1]), reverse=True
        )
        main = faces[0]
        x1, y1, x2, y2 = main.xyxy
        face_w, face_h = x2 - x1, y2 - y1
        area = face_w * face_h

        problems: List[str] = []
        if len(faces) > 1:
            second = (faces[1].xyxy[2] - faces[1].xyxy[0]) * (faces[1].xyxy[3] - faces[1].xyxy[1])
            if second >= area * MULTI_FACE_RATIO:
                problems.append("More than one face is visible - only one person at a time")

        size = float(min(face_w, face_h))
        if size < MIN_FACE_PX:
            problems.append("Move closer - the face is too small")
        if main.confidence < MIN_DET_SCORE:
            problems.append("The face is not clear enough - improve the lighting")

        margin = 2.0
        if x1 < margin or y1 < margin or x2 > width - margin or y2 > height - margin:
            problems.append("Keep your whole face inside the frame")

        crop = frame[
            max(0, int(y1)) : min(height, int(y2)), max(0, int(x1)) : min(width, int(x2))
        ]
        sharp = bright = 0.0
        if crop.size:
            sharp = face_sharpness(crop)
            bright = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
            if sharp < MIN_SHARPNESS:
                problems.append("The image is blurry - hold still")
            if bright < BRIGHTNESS_RANGE[0]:
                problems.append("Too dark - add light on your face")
            elif bright > BRIGHTNESS_RANGE[1]:
                problems.append("Too bright - avoid strong light behind or on the face")

        yaw, pitch, roll = (
            estimate_pose_angles(main.landmarks) if main.landmarks is not None else (0.0, 0.5, 0.0)
        )
        return FaceReading(
            face=main,
            width=width,
            height=height,
            count=len(faces),
            pose=classify_pose(yaw, pitch),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            size_px=size,
            sharpness=sharp,
            brightness=bright,
            problems=tuple(problems),
        )

    def analyze(self, data: bytes) -> Dict[str, Any]:
        """Report on an image without storing anything (drives the live guide)."""
        return self._read(decode_image(data)).to_dict()

    # ------------------------------------------------------------------ #
    # Enrolment
    # ------------------------------------------------------------------ #

    def add_sample(self, user_id: int, pose: str, data: bytes) -> Dict[str, Any]:
        """Learn one face template for ``user_id`` from a photo.

        ``pose`` is one of :data:`POSES`, or ``"auto"`` to file the photo under
        whichever view it turns out to be (for bulk photo uploads).
        """
        self._require_editable()
        if pose != "auto" and pose not in POSES:
            raise FaceServiceError(f"Unknown pose '{pose}'")
        user = self.db.get_user(user_id)

        frame = decode_image(data)
        reading = self._read(frame)
        if not reading.ok:
            raise FaceServiceError(reading.problems[0] if reading.problems else "No usable face")

        detected = reading.pose or "other"
        if pose == "auto":
            if detected not in POSES:
                raise FaceServiceError(
                    "Could not tell which way the head is turned - use a clearer photo"
                )
            pose = detected
        elif detected != pose:
            if detected == "other" and abs(reading.yaw) > YAW_PROFILE:
                raise FaceServiceError(
                    "The head is turned too far (side-on). Turn only part of the way - "
                    "about 30-45 degrees - so both eyes stay visible."
                )
            raise FaceServiceError(
                f"That does not look like the '{pose}' view (it looks like '{detected}'). "
                f"{POSE_LABELS[pose]}."
            )

        if user.poses.get(pose, 0) >= MAX_SAMPLES_PER_POSE:
            raise FaceServiceError(
                f"'{pose}' already has {MAX_SAMPLES_PER_POSE} captures - clear it to retake"
            )

        with self._infer_lock:
            self._ensure_models()
            embedding = self._embedder.embed(frame, reading.face)

        self._check_identity(user, embedding)
        self.db.add_sample(user_id, pose, embedding, reading.quality)
        self.rebuild_live_store()
        return {"pose": pose, "quality": reading.quality, "user": self.get_user(user_id)}

    def _check_identity(self, user: UserRow, embedding: np.ndarray) -> None:
        """Refuse a template that belongs to someone else.

        Two failures this prevents, both of which would quietly corrupt
        recognition: enrolling person B's face under person A, and enrolling
        the same face under two names.
        """
        own = [vec for _, vec in self.db.samples_of(user.id)]
        if own:
            best_own = max(cosine_similarity(embedding, vec) for vec in own)
            if best_own < SAME_PERSON_FLOOR:
                raise FaceServiceError(
                    f"This face does not match the earlier captures of {user.name}. "
                    "Only one person can be registered under a name.",
                    409,
                )

        for other_name, vectors in self.db.active_identities():
            if other_name.lower() == user.name.lower():
                continue
            if any(cosine_similarity(embedding, vec) >= self.threshold for vec in vectors):
                raise FaceServiceError(
                    f"This face is already registered as '{other_name}'.", 409
                )

    # ------------------------------------------------------------------ #
    # Testing recognition
    # ------------------------------------------------------------------ #

    def identify(self, data: bytes, margin: float = 0.05) -> Dict[str, Any]:
        """Who is in this photo? The live matcher's answer, with the top candidates.

        Uses the same store, threshold and margin rule as the cameras, so an
        operator can check a person is recognised before relying on it.
        """
        frame = decode_image(data)
        reading = self._read(frame)
        if reading.face is None:
            raise FaceServiceError(reading.problems[0] if reading.problems else "No face found")
        with self._infer_lock:
            self._ensure_models()
            embedding = self._embedder.embed(frame, reading.face)

        scores: Dict[str, float] = {}
        for name, vectors in self.db.active_identities():
            scores[name] = max(cosine_similarity(embedding, vec) for vec in vectors)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        decision = "unknown"
        reason = "No one is registered yet" if not ranked else ""
        if ranked:
            best_name, best = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
            if best < self.threshold:
                reason = f"Best match {best_name} at {best:.2f} is below the threshold {self.threshold:.2f}"
            elif best - runner_up < margin:
                reason = f"Too close between {best_name} and {ranked[1][0]} - not naming either"
            else:
                decision = best_name
                reason = f"Matched {best_name} at {best:.2f}"
        return {
            "decision": decision,
            "reason": reason,
            "threshold": self.threshold,
            "matches": [{"name": n, "similarity": round(s, 4)} for n, s in ranked[:5]],
            "face": reading.to_dict()["face"],
            "quality_problems": list(reading.problems),
        }

    def accuracy(self) -> Dict[str, Any]:
        """Leave-one-out accuracy estimate over the registered templates.

        Each template is matched against everyone else's templates (the person's
        own other templates count as the genuine match). Optimistic - the
        templates were all taken at enrolment, in good conditions - but it shows
        whether people are distinguishable at the current threshold and where a
        better threshold would sit. Real accuracy needs photos from the cameras
        (``python -m tools.evaluate_recognition``).
        """
        identities = [(n, v) for n, v in self.db.active_identities() if v]
        genuine: List[float] = []
        impostor: List[float] = []
        rank1 = trials = 0
        for index, (name, vectors) in enumerate(identities):
            others = [(n, v) for j, (n, v) in enumerate(identities) if j != index]
            for k, probe in enumerate(vectors):
                own = [vec for i, vec in enumerate(vectors) if i != k]
                best_own = max((cosine_similarity(probe, v) for v in own), default=None)
                best_other = max(
                    (cosine_similarity(probe, v) for _, vs in others for v in vs), default=None
                )
                if best_other is not None:
                    impostor.append(best_other)
                if best_own is None:
                    continue
                genuine.append(best_own)
                trials += 1
                if best_own >= self.threshold and (best_other is None or best_own > best_other):
                    rank1 += 1

        def stats(values: List[float]) -> Optional[Dict[str, float]]:
            if not values:
                return None
            arr = np.asarray(values)
            return {"min": round(float(arr.min()), 3), "mean": round(float(arr.mean()), 3),
                    "max": round(float(arr.max()), 3)}

        frr = (sum(1 for g in genuine if g < self.threshold) / len(genuine)) if genuine else None
        far = (sum(1 for i in impostor if i >= self.threshold) / len(impostor)) if impostor else None
        suggestion = None
        if genuine and impostor:
            # The threshold that minimises FAR + FRR on these templates.
            candidates = np.linspace(0.2, 0.8, 61)
            costs = [
                (sum(g < t for g in genuine) / len(genuine)) + (sum(i >= t for i in impostor) / len(impostor))
                for t in candidates
            ]
            suggestion = round(float(candidates[int(np.argmin(costs))]), 3)
        return {
            "identities": len(identities),
            "trials": trials,
            "rank1_accuracy": round(rank1 / trials, 4) if trials else None,
            "false_reject_rate": round(frr, 4) if frr is not None else None,
            "false_accept_rate": round(far, 4) if far is not None else None,
            "genuine": stats(genuine),
            "impostor": stats(impostor),
            "threshold": self.threshold,
            "suggested_threshold": suggestion,
            "note": (
                "Estimated from enrolment templates only, so optimistic. Needs at least two "
                "people for a false-accept rate and two captures per person for a false-reject rate."
            ),
        }
