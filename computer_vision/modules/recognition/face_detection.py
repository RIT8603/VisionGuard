"""Face Detection module.

One responsibility: find faces. **"Where is the face?"** and nothing more.

It does not identify anyone, and enabling it does not enable identification -
that requires enabling ``face_recognition`` separately and supplying an
identity store. Keeping the two apart is a hard rule (Agent.md §12), and it is
also the privacy boundary: face detection alone produces no personal data
beyond a box.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.inference.face_base import BackendError, FaceDetectorBackend, RawFace
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType
from computer_vision.modules.recognition.face_tracker import FaceTracker
from computer_vision.utils.motion import MotionGate

DEFAULT_BACKEND = "yunet"
FACE_LABEL = "face"


def _iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _plausible_face(box, lm: np.ndarray) -> bool:
    """Landmark geometry a real face satisfies.

    Low thresholds and upscaled blur make a detector fire on textures; real
    faces have eyes above the nose above the mouth, in proportion.
    """
    w, h = box[2] - box[0], box[3] - box[1]
    if w < 6 or h < 6 or not 0.6 < w / h < 1.6:
        return False
    re, le, nose, rm, lmo = lm
    eye = float(np.linalg.norm(le - re))
    if not 0.25 * w < eye < 0.85 * w:
        return False
    eye_mid, mouth_mid = (re + le) / 2, (rm + lmo) / 2
    if mouth_mid[1] <= eye_mid[1] + 0.15 * h:
        return False
    if not eye_mid[1] - 0.05 * h < nose[1] < mouth_mid[1] + 0.1 * h:
        return False
    roll = abs(float(np.degrees(np.arctan2(le[1] - re[1], le[0] - re[0]))))
    return roll < 40 or roll > 140


class FaceDetectionModule(BaseAIModule):
    """Locates faces using a pluggable face-detector backend.

    Config block::

        face_detection:
          enabled: true
          backend: yunet
          model: computer_vision/Model/face/face_detection_yunet_2023mar.onnx   # required
          confidence_threshold: 0.6
          min_face_size: 24        # ignore faces smaller than this, in pixels
          motion_threshold: 0.002  # FindFace motion gate; 0 = run the detector every frame
          max_reuse_frames: 10     # still frames reusing the last result before a fresh run

    Motion gate (from FindFace): on a frame where almost nothing moved, the
    detector is not run and the previous frame's faces are published again -
    a still scene has the same faces in the same places. After
    ``max_reuse_frames`` such frames the detector runs anyway, so a face that
    appears without much motion is still found.

    The model is not bundled with OpenCV; the error at startup says where to
    get it.
    """

    module_name = "face_detection"
    module_type = ModuleType.RECOGNITION
    version = "1.0.0"
    requires = ()

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._backend: FaceDetectorBackend | None = None
        self._min_confidence = float(self.params.get("confidence_threshold", 0.6))
        self._min_face_size = float(self.params.get("min_face_size", 24))
        self._motion = MotionGate(
            threshold=float(self.params.get("motion_threshold", 0.002)),
            scale=float(self.params.get("motion_scale", 0.25)),
        )
        self._max_reuse = int(self.params.get("max_reuse_frames", 10))
        # Head-crop pass for small / hazy CCTV faces (off = the plain pass only).
        self._crops = bool(self.params.get("head_crops", False))
        self._crop_conf = float(self.params.get("crop_confidence_threshold", 0.4))
        self._crop_target = int(self.params.get("crop_target_size", 256))
        self._crop_min_face = float(self.params.get("crop_min_face_size", 10))
        self._crop_min_person = float(self.params.get("crop_min_person_height", 40))
        self._crop_max_persons = int(self.params.get("crop_max_persons", 12))
        self._crop_clahe = bool(self.params.get("crop_clahe", True))
        self._person_source = str(self.params.get("person_source", "person_detection"))
        self._plain_every = max(1, int(self.params.get("plain_pass_interval", 4)))
        # Faces below this score or size are "weak" and must be seen repeatedly
        # (and not be a parked texture) before they are published. With head
        # crops on this is always so; with a stronger detector it is opt-in.
        self._confirm_low = bool(self.params.get("confirm_low_confidence", False))
        self._trust_conf = float(self.params.get("trusted_confidence", self._min_confidence))
        self._trust_size = float(self.params.get("trusted_min_face_size", self._min_face_size))
        self._crop_native_side = int(self.params.get("crop_native_side", 128))
        self._crop_min_ratio = float(self.params.get("crop_min_face_to_person", 0.06))
        self._confirm_hits = int(self.params.get("crop_confirm_hits", 2))
        self._static_seconds = float(self.params.get("crop_static_seconds", 3))
        self._confirm_tracker = FaceTracker(overlap_threshold=0.25, miss_interval=1.0, center_gate=0.8)
        self._heat: Dict[tuple, set] = {}
        self._persons_seen = False
        self._clahe = None
        self._last: List[Detection] | None = None
        self._reused = 0
        self.detector_runs = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        from computer_vision.inference.face_factory import create_face_detector

        backend_name = str(self.params.get("backend", DEFAULT_BACKEND))
        params = dict(self.params)
        if self._crops:
            # The backend cuts at its own score; the crop pass needs the lower
            # bar, the plain pass is re-filtered at confidence_threshold below.
            params["score_threshold"] = min(self._min_confidence, self._crop_conf)
        else:
            params.setdefault("score_threshold", self._min_confidence)
        try:
            self._backend = create_face_detector(backend_name, params)
            self._backend.load()
        except BackendError as exc:
            raise ModuleInitializationError(str(exc)) from exc

        self.log.info(
            "Ready: backend=%s, confidence>=%.2f, min_face=%.0fpx",
            self._backend.description,
            self._min_confidence,
            self._min_face_size,
        )

    def cleanup(self) -> None:
        self._last = None
        self._confirm_tracker.reset()
        self._heat.clear()
        self._motion.reset()
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        super().cleanup()

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        if self._backend is None:
            raise RuntimeError("FaceDetectionModule.process() called before initialize()")

        moved = self._motion(context.frame)
        if not moved and self._last is not None and self._reused < self._max_reuse:
            # Nothing moved: the faces are where they were. Republish copies
            # instead of running the detector again.
            self._reused += 1
            detections = [
                Detection(
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    source_module=self.module_name,
                    extra=dict(det.extra),
                )
                for det in self._last
            ]
            context.add_detections(self.module_name, detections)
            return self.make_result(
                context,
                detections=[self._public_dict(det) for det in detections],
                data={"count": len(detections), "reused": True,
                      "motion": round(self._motion.last_intensity, 5)},
            )

        self._reused = 0
        self.detector_runs += 1
        detections: List[Detection] = []
        crop_faces: List[Detection] = []
        run_plain = True
        if self._crops:
            crop_faces = self._detect_in_head_crops(context)
            # The head crops already cover every person; the whole-frame pass
            # only adds faces whose person the person detector missed, so it
            # runs every few detections instead of every time.
            run_plain = (not self._persons_seen) or self.detector_runs % self._plain_every == 1

        if run_plain:
            for face in self._backend.detect(context.frame):
                if face.confidence < self._min_confidence:
                    continue

                bbox = BBox.from_xyxy(face.xyxy).clip(context.width, context.height)
                if bbox.width < self._min_face_size or bbox.height < self._min_face_size:
                    continue

                detections.append(self._make_detection(face, bbox))

        if self._crops:
            # Strong whole-frame faces (>= confidence_threshold, full size) are
            # trusted; everything else came from the low-bar crop pass.
            detections = [d for d in detections if _plausible_face(d.bbox.as_tuple(), d.extra["face"].landmarks)]
            trusted = {id(d) for d in detections}
            detections = self._merge(detections, crop_faces, context)
            detections = self._confirm(detections, trusted, context.monotonic_ts)
        elif self._confirm_low:
            detections = self._confirm(detections, set(), context.monotonic_ts)

        detections.sort(key=lambda det: det.confidence, reverse=True)
        self._last = detections
        context.add_detections(self.module_name, detections)

        return self.make_result(
            context,
            detections=[self._public_dict(det) for det in detections],
            data={"count": len(detections), "reused": False,
                  "motion": round(self._motion.last_intensity, 5)},
        )

    def _make_detection(self, face: RawFace, bbox: BBox) -> Detection:
        return Detection(
            label=FACE_LABEL,
            confidence=face.confidence,
            bbox=bbox,
            source_module=self.module_name,
            # Landmarks ride along so recognition can align the crop
            # without running the detector a second time.
            extra={"face": face},
        )

    # ------------------------------------------------------------------ #
    # Head-crop pass (small / hazy CCTV faces)
    # ------------------------------------------------------------------ #

    def _boost(self, crop: np.ndarray) -> np.ndarray:
        """Local contrast on the luminance channel - hazy faces gain edges."""
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _detect_in_head_crops(self, context: FrameContext) -> List[Detection]:
        """Look for faces only where a person's head is, upscaled.

        A 15 px face in a 960 px frame is below what a whole-frame detector
        reliably finds. The same face cut out of the original frame and
        enlarged to ~256 px is well inside its range, and the search area is a
        few small crops instead of the whole picture, so it costs about the
        same as the whole-frame pass.
        """
        persons = [d for d in context.get_detections(self._person_source) if d.label == "person"]
        self._persons_seen = bool(persons)
        if not persons:
            return []
        persons.sort(key=lambda d: d.bbox.area, reverse=True)

        source = context.original_frame if context.original_frame is not None else context.frame
        sx = source.shape[1] / context.width
        sy = source.shape[0] / context.height
        found: List[Detection] = []

        for person in persons[: self._crop_max_persons]:
            b = person.bbox
            if b.height < self._crop_min_person:
                continue
            # Head region: top 45 % of the person box, a little wider.
            x1 = max(0.0, (b.x1 - 0.15 * b.width) * sx)
            x2 = min(float(source.shape[1]), (b.x2 + 0.15 * b.width) * sx)
            y1 = max(0.0, (b.y1 - 0.08 * b.height) * sy)
            y2 = min(float(source.shape[0]), (b.y1 + 0.45 * b.height) * sy)
            crop = source[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0 or min(crop.shape[:2]) < 8:
                continue

            # Only distant people need enlarging: a head crop already this big
            # holds a face the detector handles as it is (saves the CPU time
            # in crowded close-up scenes).
            side = min(crop.shape[:2])
            k = 1.0 if side >= self._crop_native_side else float(np.clip(self._crop_target / side, 1.0, 6.0))
            if k > 1.0:
                crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            if self._crop_clahe:
                crop = self._boost(crop)

            for face in self._backend.detect(crop):
                if face.confidence < self._crop_conf or face.landmarks is None:
                    continue
                # crop-pixels -> source pixels -> frame pixels
                fx, fy = 1.0 / (k * sx), 1.0 / (k * sy)
                ox, oy = x1 / sx, y1 / sy
                bx = face.xyxy
                xyxy = (bx[0] * fx + ox, bx[1] * fy + oy, bx[2] * fx + ox, bx[3] * fy + oy)
                landmarks = face.landmarks * np.array([fx, fy], np.float32) + np.array([ox, oy], np.float32)
                bbox = BBox.from_xyxy(xyxy).clip(context.width, context.height)
                if min(bbox.width, bbox.height) < self._crop_min_face:
                    continue
                # A head is about 1/7 of a standing person's height; a "face"
                # under half of that on a tall person is texture, not a face.
                if bbox.height < self._crop_min_ratio * b.height:
                    continue
                if not _plausible_face(xyxy, landmarks):
                    continue
                raw = None
                if face.raw_row is not None:
                    raw = face.raw_row.copy()
                    raw[0:4] = (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
                    raw[4:14] = landmarks.reshape(-1)
                mapped = RawFace(confidence=face.confidence, xyxy=xyxy, landmarks=landmarks, raw_row=raw)
                found.append(self._make_detection(mapped, bbox))
        return found

    def _confirm(self, detections: List[Detection], trusted: set, now: float) -> List[Detection]:
        """Drop crop-pass faces that are not a real, moving face.

        A low detector bar fires on textures (a brick wall, a bright jacket, a
        floor tile) and does so again at the same spot every frame. Two rules,
        applied to crop-pass faces only (full-size, confident faces skip them):

        * seen in at least ``crop_confirm_hits`` detections in a row - a
          one-frame flicker is not a face;
        * not parked: a face-sized spot that keeps producing "faces" in more
          than ``crop_static_seconds`` different seconds is a texture. People
          walk through a spot in a second or two; the same spot at it again and
          again for longer is not somebody's face.
        """
        ids, _ = self._confirm_tracker.update(
            [d.bbox.as_tuple() for d in detections], self.detector_runs, now
        )
        second = int(now)
        kept: List[Detection] = []
        for det, tid in zip(detections, ids):
            track = self._confirm_tracker.active.get(tid)
            if track is None:
                continue
            size = min(det.bbox.width, det.bbox.height)
            if id(det) in trusted or (det.confidence >= self._trust_conf and size >= self._trust_size):
                kept.append(det)
                continue
            if track.hits < self._confirm_hits:
                continue
            cell_px = max(24.0, 2.0 * size)
            cx, cy = det.bbox.center
            seconds = self._heat.setdefault((int(cx // cell_px), int(cy // cell_px), int(cell_px)), set())
            seconds.add(second)
            seconds.intersection_update(range(second - 30, second + 1))
            if len(seconds) > self._static_seconds:
                continue
            kept.append(det)
        if len(self._heat) > 4096:
            self._heat.clear()
        return kept

    @staticmethod
    def _merge(base: List[Detection], extra: List[Detection], context: FrameContext) -> List[Detection]:
        """Plain-pass and crop-pass faces, duplicates (IoU >= 0.35) dropped."""
        merged = sorted(base + extra, key=lambda d: d.confidence, reverse=True)
        keep: List[Detection] = []
        for det in merged:
            if all(_iou(det.bbox, k.bbox) < 0.35 for k in keep):
                keep.append(det)
        return keep

    @staticmethod
    def _public_dict(detection: Detection) -> Dict[str, Any]:
        """Serialise without the backend's raw arrays.

        ``extra`` holds landmark data that is an implementation detail and is
        not JSON-safe, so it is dropped from anything that leaves the process.
        """
        payload = detection.to_dict()
        payload.pop("extra", None)
        return payload
