"""Ultralytics YOLO detector backend.

This file is the *only* place in VisionGuard that imports ``ultralytics``, and
it is imported only when a module using this backend is enabled.

Licensing note: the ``ultralytics`` package and its pretrained weights are
AGPL-3.0. That is fine for development and for open-sourced deployments, but it
has obligations for a closed-source product. The :class:`DetectorBackend`
interface exists precisely so this can be swapped for an ONNX Runtime or
OpenCV-DNN backend without touching any module.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.models.base import (
    BackendNotAvailableError,
    DetectorBackend,
    ModelLoadError,
    RawDetection,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "yolo11n.pt"


class UltralyticsDetector(DetectorBackend):
    """Wraps ``ultralytics.YOLO`` behind the VisionGuard backend contract.

    Parameters (from the module's config block):
        model: Weight file path, or a known Ultralytics name that will be
            downloaded on first use. Default ``yolo11n.pt``.
        device: ``cpu``, ``cuda``, ``0`` ... Default ``cpu``.
        imgsz: Inference resolution. Default 640.
        confidence_threshold: Model-side score cut. Default 0.25; the module
            applies its own, stricter threshold afterwards.
        iou: NMS IoU threshold. Default 0.45.
        classes: Class ids to restrict inference to. Restricting at the model
            saves work rather than filtering afterwards.
        half: Use FP16. Only meaningful on CUDA.
    """

    name = "ultralytics"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._model = None
        self._names: Dict[int, str] = {}

        self._model_ref = str(self.params.get("model", DEFAULT_MODEL))
        self._device = str(self.params.get("device", "cpu"))
        self._imgsz = int(self.params.get("imgsz", 640))
        self._conf = float(self.params.get("model_confidence", 0.25))
        self._iou = float(self.params.get("iou", 0.45))
        self._half = bool(self.params.get("half", False))
        self._warmup = bool(self.params.get("warmup", True))

        classes = self.params.get("classes")
        self._classes = [int(c) for c in classes] if classes else None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendNotAvailableError(
                "The 'ultralytics' package is not installed, so the person "
                "detection model cannot be loaded.\n"
                "Install it with:  pip install ultralytics\n"
                "Or select a different backend in config.yaml."
            ) from exc

        local = Path(self._model_ref)
        if local.suffix and not local.exists():
            logger.info(
                "Model '%s' not found locally - Ultralytics will download it on "
                "first use (this happens once).",
                self._model_ref,
            )

        try:
            self._model = YOLO(self._model_ref)
            # Ultralytics resolves class names lazily; touch them now so a bad
            # model surfaces here rather than on the first frame.
            self._names = dict(self._model.names)
        except Exception as exc:  # noqa: BLE001 - surfaced as ModelLoadError
            raise ModelLoadError(
                f"Could not load YOLO model {self._model_ref!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._loaded = True
        logger.info(
            "Loaded %s on device '%s' (imgsz=%d, %d classes)",
            self._model_ref,
            self._device,
            self._imgsz,
            len(self._names),
        )

        if self._warmup:
            self._run_warmup()

    def _run_warmup(self) -> None:
        """Run one throwaway inference so the first real frame is not a spike.

        The first call allocates buffers and traces kernels; measured here it
        cost ~1.9 s against ~45 ms steady state. Paying that at startup keeps
        it out of the live loop.
        """
        blank = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        started = time.perf_counter()
        try:
            self.infer(blank)
        except Exception as exc:  # noqa: BLE001 - warmup must never be fatal
            logger.warning("Model warmup failed (continuing): %s", exc)
            return
        logger.info("Model warmup took %.0f ms", (time.perf_counter() - started) * 1000.0)

    def unload(self) -> None:
        self._model = None
        self._names = {}
        super().unload()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def infer(self, frame: np.ndarray) -> List[RawDetection]:
        if not self._loaded or self._model is None:
            raise ModelLoadError("UltralyticsDetector.infer() called before load()")

        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            classes=self._classes,
            verbose=False,
            # Only passed when explicitly requested: newer Ultralytics warns
            # on every call when 'half' is supplied at all, and FP16 is
            # meaningless on CPU anyway.
            **({"half": True} if self._half else {}),
        )
        if not results:
            return []

        return self._to_raw_detections(results[0])

    def _to_raw_detections(self, result: Any) -> List[RawDetection]:
        """Translate one Ultralytics ``Results`` object into RawDetections."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: List[RawDetection] = []
        for box, conf, class_id in zip(xyxy, confs, class_ids):
            detections.append(
                RawDetection(
                    label=self._names.get(int(class_id), str(class_id)),
                    confidence=float(conf),
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    class_id=int(class_id),
                )
            )
        return detections

    @property
    def description(self) -> str:
        return f"ultralytics:{self._model_ref}@{self._device}"
