"""Detector backend contract.

An AI module must not depend on an inference runtime directly. It depends on
this small interface, and a backend adapter implements it. Swapping Ultralytics
for ONNX Runtime, OpenVINO or TensorRT then means adding one file here and
changing one config value - no module, pipeline or source code changes.

``RawDetection`` is deliberately primitive (no ``BBox``, no ``Detection``):
backends speak model language, modules speak VisionGuard language, and the
module does the translation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

import numpy as np


class BackendError(Exception):
    """Base class for inference-backend failures."""


class BackendNotAvailableError(BackendError):
    """The backend's runtime package is not installed."""


class ModelLoadError(BackendError):
    """The model file is missing, corrupt or incompatible."""


@dataclass(frozen=True)
class RawDetection:
    """One detection straight out of a model, in frame pixel coordinates."""

    label: str
    confidence: float
    xyxy: Tuple[float, float, float, float]
    class_id: int = -1


class DetectorBackend(ABC):
    """An object-detection runtime adapter."""

    #: Value used in ``config.yaml`` to select this backend.
    name: ClassVar[str] = ""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory.

        Called only by an enabled module's ``initialize()``, which is how
        "disabled modules must not load their models" is upheld.

        Raises:
            BackendNotAvailableError: the runtime package is missing.
            ModelLoadError: the model could not be loaded.
        """

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[RawDetection]:
        """Run inference on one BGR frame."""

    def unload(self) -> None:
        """Release the model. Safe to call more than once."""
        self._loaded = False

    @property
    def description(self) -> str:
        return f"{type(self).__name__}"

    def _resolve_labels(self, class_ids: Sequence[int], names: Dict[int, str]) -> List[str]:
        return [names.get(int(cid), str(cid)) for cid in class_ids]

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "unloaded"
        return f"<{type(self).__name__} {state}>"
