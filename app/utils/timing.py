"""Timing primitives used by the Vision Core for real-time measurement.

These are deliberately allocation-free in the hot path: a bounded ``deque`` of
floats per meter, no locking, no I/O.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


class Stopwatch:
    """Context manager measuring wall-clock duration in milliseconds.

    >>> with Stopwatch() as sw:
    ...     pass
    >>> sw.elapsed_ms >= 0.0
    True
    """

    __slots__ = ("_start", "elapsed_ms")

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


class FpsMeter:
    """Rolling FPS estimate over the last ``window`` frame timestamps."""

    def __init__(self, window: int = 30) -> None:
        if window < 2:
            raise ValueError("FpsMeter window must be >= 2")
        self._samples: Deque[float] = deque(maxlen=window)

    def tick(self, now: float | None = None) -> float:
        """Record a frame and return the current FPS estimate."""
        self._samples.append(time.perf_counter() if now is None else now)
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        span = self._samples[-1] - self._samples[0]
        if span <= 0.0:
            return 0.0
        return (len(self._samples) - 1) / span

    def reset(self) -> None:
        self._samples.clear()


class LatencyMeter:
    """Rolling latency statistics in milliseconds."""

    def __init__(self, window: int = 30) -> None:
        self._samples: Deque[float] = deque(maxlen=window)
        self._peak_ms = 0.0

    def add(self, ms: float) -> None:
        self._samples.append(ms)
        self._peak_ms = max(self._peak_ms, ms)

    @property
    def last_ms(self) -> float:
        return self._samples[-1] if self._samples else 0.0

    @property
    def avg_ms(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def peak_ms(self) -> float:
        return self._peak_ms

    def reset(self) -> None:
        self._samples.clear()
        self._peak_ms = 0.0


@dataclass
class PipelineStats:
    """Live counters the pipeline exposes to output sinks and the HUD."""

    frames_read: int = 0
    frames_processed: int = 0
    frames_failed: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    #: Set on the first processed frame, so opening the camera (which can take
    #: seconds) does not drag down the reported throughput.
    first_frame_at: float | None = None

    fps_meter: FpsMeter = field(default_factory=FpsMeter)
    total_latency: LatencyMeter = field(default_factory=LatencyMeter)
    capture_latency: LatencyMeter = field(default_factory=LatencyMeter)
    module_latency: Dict[str, LatencyMeter] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.fps_meter.fps

    @property
    def uptime_sec(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def capture_sec(self) -> float:
        """Seconds spent actually processing frames."""
        if self.first_frame_at is None:
            return 0.0
        return time.perf_counter() - self.first_frame_at

    @property
    def overall_fps(self) -> float:
        """Average throughput across the whole run, excluding source startup."""
        elapsed = self.capture_sec
        if elapsed <= 0.0 or self.frames_processed < 2:
            return 0.0
        return (self.frames_processed - 1) / elapsed

    def record_module(self, module_name: str, ms: float) -> None:
        meter = self.module_latency.get(module_name)
        if meter is None:
            meter = LatencyMeter()
            self.module_latency[module_name] = meter
        meter.add(ms)

    def as_dict(self) -> dict:
        return {
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_failed": self.frames_failed,
            "uptime_sec": round(self.uptime_sec, 2),
            "fps": round(self.fps, 2),
            "overall_fps": round(self.overall_fps, 2),
            "latency_ms": round(self.total_latency.avg_ms, 2),
            "capture_ms": round(self.capture_latency.avg_ms, 2),
            "module_latency_ms": {
                name: round(meter.avg_ms, 2)
                for name, meter in self.module_latency.items()
            },
        }
