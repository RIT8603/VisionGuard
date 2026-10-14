"""Video file source - testing, evaluation and regression input.

Status: implemented (Phase 4).

Why this matters more than it looks: a webcam gives a different scene every
run, so no two measurements are comparable. A file gives byte-identical frames
every time, which is what makes "did that tracking parameter actually help?"
an answerable question.

Differences from a live source:
    * It ends. EOF raises :class:`SourceExhausted`, which the pipeline treats
      as a clean stop rather than an error.
    * It can loop (``video.loop``).
    * It has no natural pace. By default frames are read as fast as they can
      be processed, which is what you want for benchmarking; set
      ``video.playback_speed`` to watch it at a sensible rate.
    * It can be seeked (``video.start_frame``).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from computer_vision.core.config import CaptureBackend, VideoConfig
from computer_vision.sources.base import FramePacket, SourceExhausted, SourceOpenError
from computer_vision.sources.opencv_base import CaptureTarget, OpenCVVideoSource
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)


class VideoFileSource(OpenCVVideoSource):
    """Reads frames from a video file on disk."""

    source_type = "file"

    def __init__(self, config: VideoConfig, source_id: str | None = None) -> None:
        super().__init__(source_id or config.resolved_source_id(), config)
        self._frame_count = 0
        self._native_fps = 0.0
        self._loops_completed = 0
        self._rewinds_since_read = 0
        self._playback_started_at = 0.0
        self._frames_since_pace_reset = 0

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    @property
    def path(self) -> Path:
        return Path(self._config.path or "")

    @property
    def frame_count(self) -> int:
        """Total frames reported by the container. 0 when unknown."""
        return self._frame_count

    @property
    def duration_sec(self) -> float:
        if self._native_fps <= 0 or self._frame_count <= 0:
            return 0.0
        return self._frame_count / self._native_fps

    @property
    def loops_completed(self) -> int:
        return self._loops_completed

    def _capture_target(self) -> CaptureTarget:
        return str(self.path)

    def _resolve_backend(self) -> CaptureBackend:
        if self._config.backend is not CaptureBackend.AUTO:
            return self._config.backend
        return CaptureBackend.FFMPEG

    def _describe(self) -> str:
        parts = [self.path.name]
        if self._frame_count:
            parts.append(f"{self._frame_count} frames")
        if self.duration_sec:
            parts.append(f"{self.duration_sec:.1f}s")
        return " | ".join(parts)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _on_before_open(self, backend: CaptureBackend) -> None:
        """Fail early and clearly on a missing or unreadable path."""
        if not self._config.path:
            raise SourceOpenError(
                "video.path is not set. Set it in config.yaml, pass "
                "--video-path, or set VIDEO_PATH."
            )
        if not self.path.exists():
            raise SourceOpenError(
                f"Video file not found: {self.path}\n"
                f"(resolved from working directory {Path.cwd()})"
            )
        if self.path.is_dir():
            raise SourceOpenError(f"video.path is a directory, not a file: {self.path}")

    def _configure_capture(self, capture: "cv2.VideoCapture") -> None:
        self._frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self._native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._loops_completed = 0
        self._rewinds_since_read = 0
        self._reset_pacing()

        start_frame = int(self._config.start_frame or 0)
        if start_frame > 0:
            self._seek(capture, start_frame)

        logger.info(
            "Opened %s: %d frames, %.2f fps, %.1fs%s",
            self.path.name,
            self._frame_count,
            self._native_fps,
            self.duration_sec,
            " (looping)" if self._config.loop else "",
        )

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def read(self) -> FramePacket | None:
        self._pace()
        packet = super().read()
        if packet is not None:
            self._rewinds_since_read = 0
        return packet

    def _handle_read_failure(self, attempt: int) -> bool:
        """Distinguish end-of-file from a decode hiccup.

        A file that has run out must not be retried with sleeps the way a live
        camera is - that would just stall for ``read_retries`` before stopping.
        """
        if not self._at_end_of_file():
            return super()._handle_read_failure(attempt)

        if not self._config.loop:
            raise SourceExhausted(
                f"Reached the end of {self.path.name} after {self.frame_id} frames"
            )

        # Guard against an empty or fully undecodable file looping forever.
        if self._rewinds_since_read >= 1:
            raise SourceExhausted(
                f"{self.path.name} produced no decodable frames after rewinding - "
                "the file may be empty or corrupt"
            )

        self._rewind()
        return True

    def _at_end_of_file(self) -> bool:
        if self._capture is None:
            return True
        if self._frame_count <= 0:
            # Some containers do not report a reliable frame count; for those
            # a failed read is assumed to be EOF rather than a decode error.
            return True
        position = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        # Container metadata frame counts can differ by several frames from
        # actual decodable frames (e.g. keyframe index offsets, demuxer padding).
        # Any read failure within 10 frames of reported EOF is considered EOF.
        return position >= max(1, self._frame_count - 10)

    def _rewind(self) -> None:
        if self._capture is None:
            return
        self._loops_completed += 1
        self._rewinds_since_read += 1
        self._seek(self._capture, int(self._config.start_frame or 0))
        self._reset_pacing()
        logger.debug("Looped %s (loop #%d)", self.path.name, self._loops_completed)

    # ------------------------------------------------------------------ #
    # Seeking
    # ------------------------------------------------------------------ #

    def seek(self, frame_index: int) -> None:
        """Jump to ``frame_index``. Useful for evaluating a specific segment."""
        if self._capture is None:
            raise SourceOpenError("seek() called before open()")
        self._seek(self._capture, frame_index)
        self._reset_pacing()

    def _seek(self, capture: "cv2.VideoCapture", frame_index: int) -> None:
        target = max(0, frame_index)
        if self._frame_count and target >= self._frame_count:
            raise SourceOpenError(
                f"Cannot seek to frame {target}: {self.path.name} has "
                f"{self._frame_count} frames"
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)

    # ------------------------------------------------------------------ #
    # Pacing
    # ------------------------------------------------------------------ #

    def _reset_pacing(self) -> None:
        self._playback_started_at = time.perf_counter()
        self._frames_since_pace_reset = 0

    def _pace(self) -> None:
        """Throttle to ``playback_speed`` x the file's native frame rate.

        Paced against a fixed origin rather than the previous frame, so slow
        frames do not accumulate drift.
        """
        speed = float(self._config.playback_speed or 0.0)
        if speed <= 0.0 or self._native_fps <= 0.0:
            return

        interval = 1.0 / (self._native_fps * speed)
        due_at = self._playback_started_at + self._frames_since_pace_reset * interval
        self._frames_since_pace_reset += 1

        delay = due_at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    # ------------------------------------------------------------------ #
    # Progress
    # ------------------------------------------------------------------ #

    @property
    def progress(self) -> float:
        """Fraction of the file consumed, 0.0-1.0. Returns 0.0 when unknown."""
        if self._capture is None or self._frame_count <= 0:
            return 0.0
        position = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        return min(1.0, position / self._frame_count)
