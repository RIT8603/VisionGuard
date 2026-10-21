"""Pose estimation backend contract, and the Ultralytics implementation.

A pose model answers "where are this person's joints?". Posture questions -
is someone lying down, has someone fallen - are analysis built on top of that
answer, never part of it.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

import numpy as np

from computer_vision.core.context import KEYPOINT_NAMES
from computer_vision.inference.base import (
    BackendError,
    BackendNotAvailableError,
    ModelLoadError,
    verify_model_file,
)
from computer_vision.inference.batching import BatchCombiner
from computer_vision.inference.device import resolve_torch_device
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_POSE_MODEL = "computer_vision/Model/pose/yolo11n-pose.pt"

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

    def estimate_batch(self, frames: Sequence[np.ndarray]) -> List[List[RawPose]]:
        """Poses for several frames, in order. Runtimes that can batch override it."""
        return [self.estimate(frame) for frame in frames]

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
        model: Weights. Default ``yolo11n-pose.pt``. Downloaded only when
            ``allow_download`` is true.
        model_sha256: Optional expected SHA-256; loading is refused on mismatch.
        device / imgsz / iou: as for the detection backend.
        model_confidence: model-side score cut; the module applies its own.
    """

    name = "ultralytics"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._model = None
        self._model_ref = str(self.params.get("model") or DEFAULT_POSE_MODEL)
        self._device = resolve_torch_device(self.params.get("device", "cpu"))
        self._imgsz = int(self.params.get("imgsz", 640))
        self._conf = float(self.params.get("model_confidence", 0.25))
        self._iou = float(self.params.get("iou", 0.45))
        self._warmup = bool(self.params.get("warmup", True))
        self._half = bool(self.params.get("half", False)) and self._device.startswith("cuda")

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

        # Weights are pickles: verify (and only if allowed, fetch) first.
        from computer_vision.inference.ultralytics_backend import _download_only

        model_path = verify_model_file(
            self._model_ref,
            expected_sha256=self.params.get("model_sha256"),
            allow_download=bool(self.params.get("allow_download", False)),
            download=_download_only,
        )

        try:
            self._model = YOLO(model_path)
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
            **({"quantize": 16} if self._half else {}),  # FP16 (the old "half" flag is deprecated)
        )
        if not results:
            return []
        return self._to_raw_poses(results[0])

    def estimate_batch(self, frames: Sequence[np.ndarray]) -> List[List[RawPose]]:
        """One forward pass over several cameras' frames."""
        if not self._loaded or self._model is None:
            raise ModelLoadError("UltralyticsPoseBackend.estimate_batch() called before load()")
        if len(frames) == 1:
            return [self.estimate(frames[0])]
        results = self._model.predict(
            source=list(frames),
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
            **({"quantize": 16} if self._half else {}),  # FP16 (the old "half" flag is deprecated)
        )
        return [self._to_raw_poses(result) for result in results]

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


class SharedPoseBackend(PoseBackend):
    """One pose model for every camera, with their frames batched together.

    Each camera used to load its own copy of the pose model and run it on its
    own thread, so the copies fought over the GPU. Cameras with the same pose
    settings now get this wrapper around one model; reference-counted like the
    shared detector, and dropped when the last camera releases it.
    """

    def __init__(self, inner: PoseBackend, key: Tuple[Any, ...]) -> None:
        super().__init__(inner.params)
        self._inner = inner
        self._key = key
        self._users = 0
        self._lock = threading.Lock()
        self._batcher: BatchCombiner[List[RawPose]] = BatchCombiner(
            inner.estimate_batch, max_batch=int(inner.params.get("max_batch", 8))
        )

    def load(self) -> None:
        with self._lock:
            self._users += 1
            if self._inner.is_loaded:
                self._loaded = True
                return
        try:
            self._inner.load()
        except BaseException:
            # Do not leave a broken model cached for the next camera to reuse.
            with self._lock:
                self._users = max(0, self._users - 1)
                if self._users == 0:
                    _shared_pose.pop(self._key, None)
            raise
        self._loaded = True

    def estimate(self, frame: np.ndarray) -> List[RawPose]:
        return self._batcher.submit(frame)

    def unload(self) -> None:
        with self._lock:
            self._users = max(0, self._users - 1)
            if self._users > 0:
                return
            _shared_pose.pop(self._key, None)
        self._inner.unload()
        self._loaded = False

    @property
    def description(self) -> str:
        return f"{self._inner.description} (shared x{self._users})"


_shared_pose: Dict[Tuple[Any, ...], SharedPoseBackend] = {}
_shared_pose_lock = threading.Lock()


def create_pose_backend(name: str, params: Dict[str, Any] | None = None) -> PoseBackend:
    """Build (but do not load) a pose backend.

    Cameras asking for the same model and settings share one instance (opt out
    with ``share_model: false``).
    """
    params = dict(params or {})
    if params.get("share_model", True):
        key = (
            name,
            str(params.get("model") or DEFAULT_POSE_MODEL),
            resolve_torch_device(params.get("device", "cpu")),
            int(params.get("imgsz", 640)),
            round(float(params.get("model_confidence", 0.25)), 6),
            round(float(params.get("iou", 0.45)), 6),
            bool(params.get("half", False)),
        )
        with _shared_pose_lock:
            shared = _shared_pose.get(key)
            if shared is None:
                shared = SharedPoseBackend(_build_pose_backend(name, params), key)
                _shared_pose[key] = shared
            return shared
    return _build_pose_backend(name, params)


def _build_pose_backend(name: str, params: Dict[str, Any]) -> PoseBackend:
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
