"""Pose estimation backend contract, and the Ultralytics implementation.

A pose model answers "where are this person's joints?". Posture questions -
is someone lying down, has someone fallen - are analysis built on top of that
answer, never part of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np

from computer_vision.core.context import KEYPOINT_NAMES
from computer_vision.inference.base import BackendError, BackendNotAvailableError, ModelLoadError
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_POSE_MODEL = "models/pose/yolo11n-pose.pt"

__all__ = [
    "BackendError",
    "DEFAULT_POSE_MODEL",
    "PoseBackend",
    "RawPose",
    "UltralyticsPoseBackend",
]


@dataclass(frozen=True)
class RawPose:
    """One person's skeleton straight from a pose model.

    ``keypoints`` is ``(N, 3)`` of ``x, y, confidence`` in frame pixels,
    ordered as :data:`computer_vision.core.context.KEYPOINT_NAMES`.
    """

    confidence: float
    xyxy: Tuple[float, float, float, float]
    keypoints: np.ndarray


class PoseBackend(ABC):
    """Estimates body keypoints."""

    name: ClassVar[str] = ""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> List[RawPose]:
        ...

    def unload(self) -> None:
        self._loaded = False

    @property
    def description(self) -> str:
        return type(self).__name__


class UltralyticsPoseBackend(PoseBackend):
    """Wraps an Ultralytics ``-pose`` model.

    Parameters:
        model: Weights. Default ``yolo11n-pose.pt``, downloaded on first use.
        device / imgsz / iou: as for the detection backend.
        model_confidence: model-side score cut; the module applies its own.
    """

    name = "ultralytics"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._model = None
        self._model_ref = str(self.params.get("model") or DEFAULT_POSE_MODEL)
        self._device = str(self.params.get("device", "cpu"))
        self._imgsz = int(self.params.get("imgsz", 640))
        self._conf = float(self.params.get("model_confidence", 0.25))
        self._iou = float(self.params.get("iou", 0.45))
        self._warmup = bool(self.params.get("warmup", True))

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendNotAvailableError(
                "The 'ultralytics' package is not installed, so pose "
                "estimation cannot run.\n    pip install ultralytics"
            ) from exc

        try:
            self._model = YOLO(self._model_ref)
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise ModelLoadError(
                f"Could not load pose model {self._model_ref!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._loaded = True
        logger.info("Loaded pose model %s on '%s'", self._model_ref, self._device)

        if self._warmup:
            try:
                self.estimate(np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8))
            except Exception as exc:  # noqa: BLE001 - warmup is never fatal
                logger.warning("Pose warmup failed (continuing): %s", exc)

    def estimate(self, frame: np.ndarray) -> List[RawPose]:
        if not self._loaded or self._model is None:
            raise ModelLoadError("UltralyticsPoseBackend.estimate() called before load()")

        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []
        return self._to_raw_poses(results[0])

    def _to_raw_poses(self, result: Any) -> List[RawPose]:
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        # `.data` is (people, joints, 3) with x, y, confidence.
        joints = keypoints.data.cpu().numpy()

        poses: List[RawPose] = []
        for index in range(len(xyxy)):
            box = xyxy[index]
            poses.append(
                RawPose(
                    confidence=float(confs[index]),
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    keypoints=np.asarray(joints[index], dtype=np.float32),
                )
            )
        return poses

    def unload(self) -> None:
        self._model = None
        super().unload()

    @property
    def description(self) -> str:
        return f"ultralytics-pose:{Path(self._model_ref).name}@{self._device}"


POSE_BACKENDS: Dict[str, str] = {
    "ultralytics": "computer_vision.inference.pose_backend:UltralyticsPoseBackend",
}


def available_pose_backends() -> List[str]:
    return sorted(POSE_BACKENDS)


def create_pose_backend(name: str, params: Dict[str, Any] | None = None) -> PoseBackend:
    """Build (but do not load) a pose backend."""
    import importlib

    import_path = POSE_BACKENDS.get(name)
    if import_path is None:
        raise BackendError(
            f"Unknown pose backend {name!r}. Available: {available_pose_backends()}"
        )

    module_path, _, class_name = import_path.partition(":")
    try:
        module = importlib.import_module(module_path)
        backend_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise BackendError(
            f"Could not load pose backend {name!r} from {import_path!r}: {exc}"
        ) from exc

    return backend_cls(params)


def keypoint_name(index: int) -> str:
    """Name for a keypoint index, tolerant of models with extra joints."""
    if 0 <= index < len(KEYPOINT_NAMES):
        return KEYPOINT_NAMES[index]
    return f"kp_{index}"
