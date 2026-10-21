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

from computer_vision.core.context import BBox, Detection, FrameContext
from computer_vision.inference.face_base import BackendError, FaceDetectorBackend
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType
from computer_vision.utils.motion import MotionGate

DEFAULT_BACKEND = "yunet"
FACE_LABEL = "face"


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
        self._last: List[Detection] | None = None
        self._reused = 0
        self.detector_runs = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        from computer_vision.inference.face_factory import create_face_detector

        backend_name = str(self.params.get("backend", DEFAULT_BACKEND))
        try:
            self._backend = create_face_detector(backend_name, self.params)
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
        for face in self._backend.detect(context.frame):
            if face.confidence < self._min_confidence:
                continue

            bbox = BBox.from_xyxy(face.xyxy).clip(context.width, context.height)
            if bbox.width < self._min_face_size or bbox.height < self._min_face_size:
                continue

            detections.append(
                Detection(
                    label=FACE_LABEL,
                    confidence=face.confidence,
                    bbox=bbox,
                    source_module=self.module_name,
                    # Landmarks ride along so recognition can align the crop
                    # without running the detector a second time.
                    extra={"face": face},
                )
            )

        detections.sort(key=lambda det: det.confidence, reverse=True)
        self._last = detections
        context.add_detections(self.module_name, detections)

        return self.make_result(
            context,
            detections=[self._public_dict(det) for det in detections],
            data={"count": len(detections), "reused": False,
                  "motion": round(self._motion.last_intensity, 5)},
        )

    @staticmethod
    def _public_dict(detection: Detection) -> Dict[str, Any]:
        """Serialise without the backend's raw arrays.

        ``extra`` holds landmark data that is an implementation detail and is
        not JSON-safe, so it is dropped from anything that leaves the process.
        """
        payload = detection.to_dict()
        payload.pop("extra", None)
        return payload
