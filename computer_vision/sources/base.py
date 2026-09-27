"""Backend-agnostic video source contract.

This module deliberately does not import OpenCV. It defines *what* a video
source is; ``computer_vision.sources.opencv_base`` defines *how* OpenCV implements one.
A future GStreamer or PyAV source would subclass :class:`BaseVideoSource`
directly and the pipeline would not notice.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class VideoSourceError(Exception):
    """Base class for every video-source failure."""


class SourceOpenError(VideoSourceError):
    """The source could not be opened (device busy, missing, bad URL...)."""


class FrameReadError(VideoSourceError):
    """A frame could not be read and the source considers this unrecoverable."""


class SourceNotOpenError(VideoSourceError):
    """``read()`` was called before ``open()``."""


class SourceExhausted(VideoSourceError):
    """A finite source (a file) reached its end. Not an error for the pipeline."""


# --------------------------------------------------------------------------- #
# Data carriers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SourceInfo:
    """Negotiated properties of an opened source.

    These are what the driver actually granted, which can differ from what the
    configuration requested.
    """

    source_id: str
    source_type: str
    width: int
    height: int
    fps: float
    backend: str = "unknown"
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "backend": self.backend,
            "description": self.description,
        }


@dataclass
class FramePacket:
    """A raw captured frame plus its provenance.

    This is the only thing a source hands to the Vision Core. Every source type
    produces exactly this structure, which is what makes webcam and RTSP
    interchangeable.
    """

    frame: np.ndarray
    frame_id: int
    source_id: str
    monotonic_ts: float = field(default_factory=time.perf_counter)
    wall_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    capture_ms: float = 0.0

    @property
    def height(self) -> int:
        return int(self.frame.shape[0])

    @property
    def width(self) -> int:
        return int(self.frame.shape[1])


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #

class BaseVideoSource(ABC):
    """Abstract video source.

    Lifecycle::

        source.open()
        while True:
            packet = source.read()     # None == no frame available right now
            if packet is None:
                continue
        source.release()

    or simply::

        with source:
            for packet in source.frames():
                ...

    Implementations are responsible for their own resilience policy (retries,
    reconnects). They raise :class:`VideoSourceError` only when the situation is
    unrecoverable, and :class:`SourceExhausted` when a finite stream ends.
    """

    source_type: str = "base"

    def __init__(self, source_id: str) -> None:
        self._source_id = source_id
        self._frame_id = 0
        self._is_open = False

    # ---------------- identity ---------------- #

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def frame_id(self) -> int:
        """Number of frames handed out so far."""
        return self._frame_id

    # ---------------- lifecycle ---------------- #

    @abstractmethod
    def open(self) -> SourceInfo:
        """Acquire the underlying device/stream. Idempotent."""

    @abstractmethod
    def read(self) -> Optional[FramePacket]:
        """Return the next frame, or ``None`` if none is available right now."""

    @abstractmethod
    def release(self) -> None:
        """Release the device/stream. Idempotent and safe to call when closed."""

    @property
    @abstractmethod
    def info(self) -> SourceInfo:
        """Properties of the opened source."""

    # ---------------- conveniences ---------------- #

    def frames(self) -> Iterator[FramePacket]:
        """Yield frames until the source is exhausted or released.

        ``None`` reads are skipped rather than terminating the iteration, so a
        transient hiccup on an RTSP stream does not end the pipeline.
        """
        if not self._is_open:
            self.open()
        while self._is_open:
            try:
                packet = self.read()
            except SourceExhausted:
                break
            if packet is None:
                continue
            yield packet

    def _next_frame_id(self) -> int:
        self._frame_id += 1
        return self._frame_id

    def __enter__(self) -> "BaseVideoSource":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "open" if self._is_open else "closed"
        return f"<{type(self).__name__} id={self._source_id!r} {state}>"
