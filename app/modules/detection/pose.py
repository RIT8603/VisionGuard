"""Pose Detection module.

One responsibility: locate people's body keypoints.

It reports joints. It does not decide whether a posture means anything -
"lying down", "fallen", "reaching into a restricted shelf" are conclusions,
and conclusions belong to analysis modules that consume this output.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.context import BBox, Keypoint, Pose
from app.core.context import FrameContext
from app.models.base import BackendError
from app.models.pose_backend import (
    DEFAULT_POSE_MODEL,
    PoseBackend,
    create_pose_backend,
    keypoint_name,
)
from app.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

DEFAULT_BACKEND = "ultralytics"


class PoseDetectionModule(BaseAIModule):
    """Estimates body keypoints for every person in the frame.

    Config block::

        pose_detection:
          enabled: true
          backend: ultralytics
          model: yolo11n-pose.pt     # downloaded on first use
          device: cpu
          imgsz: 640
          confidence_threshold: 0.4
          keypoint_threshold: 0.3    # below this a joint counts as unseen

    Publishes to ``context.poses`` *and* a person box to
    ``context.detections`` under its own name, so it can feed a tracker
    exactly like any other detector.
    """

    module_name = "pose_detection"
    module_type = ModuleType.DETECTION
    version = "1.0.0"
    requires = ()

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._backend: PoseBackend | None = None
        self._min_confidence = float(self.params.get("confidence_threshold", 0.4))
        self._keypoint_threshold = float(self.params.get("keypoint_threshold", 0.3))

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        backend_name = str(self.params.get("backend", DEFAULT_BACKEND))
        params = dict(self.params)
        params.setdefault("model", DEFAULT_POSE_MODEL)

        try:
            self._backend = create_pose_backend(backend_name, params)
            self._backend.load()
        except BackendError as exc:
            raise ModuleInitializationError(str(exc)) from exc

        self.log.info(
            "Ready: backend=%s, confidence>=%.2f, keypoint>=%.2f",
            self._backend.description,
            self._min_confidence,
            self._keypoint_threshold,
        )

    def cleanup(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        super().cleanup()

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        if self._backend is None:
            raise RuntimeError("PoseDetectionModule.process() called before initialize()")

        poses: List[Pose] = []
        for raw in self._backend.estimate(context.frame):
            if raw.confidence < self._min_confidence:
                continue

            bbox = BBox.from_xyxy(raw.xyxy).clip(context.width, context.height)
            if bbox.width <= 0 or bbox.height <= 0:
                continue

            poses.append(
                Pose(
                    bbox=bbox,
                    keypoints=self._to_keypoints(raw.keypoints),
                    confidence=raw.confidence,
                    source_module=self.module_name,
                )
            )

        poses.sort(key=lambda pose: pose.confidence, reverse=True)
        context.add_poses(self.module_name, poses)

        # Also publish a plain person box, so a tracker can consume this
        # module without knowing that keypoints exist.
        context.add_detections(
            self.module_name,
            [
                self._as_detection(pose)
                for pose in poses
            ],
        )

        return self.make_result(
            context,
            detections=[self._as_detection(pose).to_dict() for pose in poses],
            data={
                "count": len(poses),
                "poses": [pose.to_dict() for pose in poses],
                "mean_visible_keypoints": self._mean_visible(poses),
            },
        )

    def _to_keypoints(self, array) -> List[Keypoint]:
        points: List[Keypoint] = []
        for index, row in enumerate(array):
            confidence = float(row[2]) if len(row) > 2 else 0.0
            points.append(
                Keypoint(
                    name=keypoint_name(index),
                    x=float(row[0]),
                    y=float(row[1]),
                    # Zeroing low-confidence joints keeps `is_visible`
                    # consistent with the module's own threshold.
                    confidence=confidence if confidence >= self._keypoint_threshold else 0.0,
                )
            )
        return points

    @staticmethod
    def _as_detection(pose: Pose):
        from app.core.context import Detection

        return Detection(
            label="person",
            confidence=pose.confidence,
            bbox=pose.bbox,
            source_module="pose_detection",
        )

    @staticmethod
    def _mean_visible(poses: List[Pose]) -> float:
        if not poses:
            return 0.0
        visible = [sum(1 for k in pose.keypoints if k.is_visible) for pose in poses]
        return round(sum(visible) / len(visible), 2)
