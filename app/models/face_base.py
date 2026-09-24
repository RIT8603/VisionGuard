"""Face backend contracts.

Two separate capabilities, kept separate on purpose (Agent.md §12):

* :class:`FaceDetectorBackend` answers **"where is a face?"**
* :class:`FaceEmbedderBackend` answers **"what does this face look like,
  numerically?"** - the input to deciding *whose* it is.

A module depends on these small interfaces, never on a face library directly,
so the runtime can be swapped without touching a module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from app.models.base import BackendError, BackendNotAvailableError, ModelLoadError

__all__ = [
    "BackendError",
    "BackendNotAvailableError",
    "FaceDetectorBackend",
    "FaceEmbedderBackend",
    "ModelLoadError",
    "RawFace",
]


@dataclass(frozen=True)
class RawFace:
    """One detected face in frame pixel coordinates.

    ``landmarks`` are the five points (eyes, nose, mouth corners) that
    alignment needs. Recognition accuracy drops sharply on unaligned crops, so
    a detector that can supply them should.
    """

    confidence: float
    xyxy: Tuple[float, float, float, float]
    landmarks: Optional[np.ndarray] = None
    #: Backend-specific row, kept so an embedder can re-use the detector's own
    #: output format for alignment without re-deriving it.
    raw_row: Optional[np.ndarray] = field(default=None, repr=False)


class FaceDetectorBackend(ABC):
    """Locates faces. Knows nothing about identity."""

    name: ClassVar[str] = ""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model. Called only by an enabled module."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[RawFace]:
        """Find faces in one BGR frame."""

    def unload(self) -> None:
        self._loaded = False

    @property
    def description(self) -> str:
        return type(self).__name__


class FaceEmbedderBackend(ABC):
    """Turns a face into a vector that can be compared to other faces.

    The vector is the only thing VisionGuard keeps. It is not an image and
    cannot be displayed as one, which is what makes an embeddings-only
    identity store meaningfully more private than storing photographs.
    """

    name: ClassVar[str] = ""
    #: Cosine similarity at or above which two embeddings are the same person.
    default_threshold: ClassVar[float] = 0.363

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
    def embed(self, frame: np.ndarray, face: RawFace) -> np.ndarray:
        """Return a 1-D float32 embedding for one face in ``frame``."""

    def unload(self) -> None:
        self._loaded = False

    @property
    def description(self) -> str:
        return type(self).__name__


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors, in [-1, 1]."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)
