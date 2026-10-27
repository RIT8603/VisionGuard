"""Output sinks - where an analysed frame goes.

The pipeline emits to a sink and does not care what the sink does with the
frame: a live OpenCV window, a log line, or several at once via
:class:`MultiSink`.

The dashboard's sink deliberately does **not** live here. It is
``backend.services.web_sink.WebSink``, because it depends on backend state and
this package must not depend on the backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple

import cv2

from computer_vision.core.config import ProcessingConfig
from computer_vision.core.context import FrameContext
from computer_vision.utils.magic_fx import MagicTrails, magic_enabled
from computer_vision.utils.annotate import (
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_TEXT,
    color_for_label,
    color_for_track_id,
    draw_detections,
    draw_hud,
    draw_overlays,
    draw_tracks,
    tracks_to_draw,
    track_names,
)
from computer_vision.utils.logger import get_logger
from computer_vision.utils.timing import PipelineStats

logger = get_logger(__name__)

_QUIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 == ESC


class DisplayUnavailableError(RuntimeError):
    """The OpenCV build or environment cannot show a window."""


def untracked_detections(context: FrameContext) -> List[Any]:
    """Detections that no tracking module turned into tracks this frame.

    A tracking module records ``data["consumed_from"]`` naming the detector it
    consumed. Reading that keeps this sink module-agnostic: it never needs to
    know that ``person_tracking`` follows ``person_detection``.
    """
    consumed = set()
    for result in context.module_results.values():
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            source = data.get("consumed_from")
            if source:
                consumed.add(source)

    return [
        detection
        for module_name, items in context.detections.items()
        if module_name not in consumed
        for detection in items
    ]


class BaseOutputSink(ABC):
    """Consumes analysed frames."""

    def prepare(self) -> None:
        """Acquire resources up front so failures surface before the loop starts."""

    @abstractmethod
    def emit(self, context: FrameContext, stats: PipelineStats) -> bool:
        """Handle one frame. Return ``False`` to ask the pipeline to stop."""

    def close(self) -> None:
        """Release any resources."""


class NullSink(BaseOutputSink):
    """Headless sink: logs a stats line periodically. Used for benchmarks and CI."""

    def __init__(self, log_every: int = 60) -> None:
        self._log_every = log_every

    def emit(self, context: FrameContext, stats: PipelineStats) -> bool:
        if self._log_every and context.frame_id % self._log_every == 0:
            logger.info(
                "frame=%d fps=%.1f latency=%.1fms detections=%d tracks=%d events=%d",
                context.frame_id,
                stats.fps,
                stats.total_latency.avg_ms,
                len(context.get_detections()),
                len(context.get_tracks()),
                len(context.events),
            )
        return True


class WindowSink(BaseOutputSink):
    """Live annotated preview in an OpenCV window.

    Stops the pipeline when the user presses ``q``/``ESC`` or closes the window.
    """

    def __init__(
        self,
        config: ProcessingConfig,
        app_name: str,
        source_description: str,
        module_names: Sequence[str],
    ) -> None:
        self._config = config
        self._app_name = app_name
        self._source_description = source_description
        self._module_names = list(module_names)
        self._window_ready = False
        self._fx = MagicTrails()

    # ---------------- lifecycle ---------------- #

    def prepare(self) -> None:
        self._ensure_window()

    def _ensure_window(self) -> None:
        if self._window_ready:
            return
        try:
            cv2.namedWindow(self._config.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise DisplayUnavailableError(
                "This OpenCV build cannot open a window (headless install or no "
                "display available). Run with --no-window to continue headless.\n"
                f"OpenCV said: {exc}"
            ) from exc
        self._window_ready = True

    def close(self) -> None:
        if self._window_ready:
            try:
                cv2.destroyWindow(self._config.window_name)
                cv2.waitKey(1)
            except cv2.error:  # pragma: no cover - platform dependent
                pass
            self._window_ready = False

    # ---------------- rendering ---------------- #

    def emit(self, context: FrameContext, stats: PipelineStats) -> bool:
        self._ensure_window()

        frame = context.frame.copy()

        # Zones and other module-published shapes go down first, so boxes and
        # labels stay readable on top of them.
        if context.overlays:
            draw_overlays(frame, context.overlays)

        if magic_enabled():
            self._fx.draw(frame, context.get_tracks(), keep_clear=context.get_detections())

        if self._config.show_boxes:
            # Tracks first, then only the detections nobody turned into tracks,
            # so a tracked person gets one box with an id rather than two.
            draw_detections(frame, untracked_detections(context))
            draw_tracks(frame, tracks_to_draw(context), track_names(context.get_detections("face_recognition")))

        if self._config.show_hud:
            draw_hud(frame, self._hud_lines(context, stats))

        try:
            cv2.imshow(self._config.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            raise DisplayUnavailableError(f"Failed to display frame: {exc}") from exc

        if key in _QUIT_KEYS:
            logger.info("Quit key pressed - stopping")
            return False

        if self._window_closed():
            logger.info("Preview window closed - stopping")
            return False

        return True

    def _detection_lines(
        self, context: FrameContext
    ) -> List[Tuple[str, Tuple[int, int, int], float]]:
        """HUD lines for this frame's results, capped so the panel stays small.

        Tracks are listed by their persistent id where tracking is enabled;
        otherwise raw detections are listed by position.
        """
        tracks = context.get_tracks()
        detections = untracked_detections(context)
        limit = self._config.hud_max_detections
        lines: List[Tuple[str, Tuple[int, int, int], float]] = []

        if tracks:
            lines.append((f"Tracked: {len(tracks)}", COLOR_TEXT, 0.48))
            for track in tracks[:limit]:
                lines.append(
                    (
                        f"  #{track.track_id} {track.label}  conf {track.confidence:.2f}",
                        color_for_track_id(track.track_id, track.source_module),
                        0.46,
                    )
                )
            lines.extend(self._overflow_line(len(tracks), limit))

        if detections:
            lines.append((f"Detections: {len(detections)}", COLOR_TEXT, 0.48))
            for index, detection in enumerate(detections[:limit], start=1):
                lines.append(
                    (
                        f"  {detection.label} #{index}  conf {detection.confidence:.2f}",
                        color_for_label(detection.label),
                        0.46,
                    )
                )
            lines.extend(self._overflow_line(len(detections), limit))

        return lines

    def _event_lines(
        self, context: FrameContext
    ) -> List[Tuple[str, Tuple[int, int, int], float]]:
        """Show this frame's events, coloured by severity."""
        if not context.events:
            return []

        colors = {
            "info": COLOR_TEXT,
            "warning": (0, 210, 255),
            "critical": (80, 80, 255),
        }
        lines: List[Tuple[str, Tuple[int, int, int], float]] = [
            ("Events:", COLOR_TEXT, 0.48)
        ]
        limit = self._config.hud_max_detections
        for event in context.events[:limit]:
            lines.append(
                (f"  {event.describe()}", colors.get(event.severity.value, COLOR_TEXT), 0.46)
            )
        lines.extend(self._overflow_line(len(context.events), limit))
        return lines

    @staticmethod
    def _overflow_line(
        total: int, limit: int
    ) -> List[Tuple[str, Tuple[int, int, int], float]]:
        remaining = total - limit
        if remaining <= 0:
            return []
        return [(f"  (+{remaining} more)", COLOR_MUTED, 0.44)]

    def _window_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(self._config.window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:  # pragma: no cover - platform dependent
            return True

    def _hud_lines(
        self, context: FrameContext, stats: PipelineStats
    ) -> List[Tuple[str, Tuple[int, int, int], float]]:
        lines: List[Tuple[str, Tuple[int, int, int], float]] = [
            (self._app_name, COLOR_ACCENT, 0.62),
            (f"Camera: {self._source_description}", COLOR_TEXT, 0.48),
        ]

        if self._module_names:
            lines.append(("Modules:", COLOR_TEXT, 0.48))
            lines.extend((f"  [x] {name}", COLOR_OK, 0.46) for name in self._module_names)
        else:
            lines.append(("Modules: none enabled (Phase 1)", COLOR_MUTED, 0.46))

        lines.append(
            (
                f"FPS: {stats.fps:5.1f}   Latency: {stats.total_latency.avg_ms:5.1f} ms",
                COLOR_TEXT,
                0.48,
            )
        )
        lines.append(
            (
                f"Frame: {context.frame_id}   {context.width}x{context.height}",
                COLOR_MUTED,
                0.46,
            )
        )

        lines.extend(self._detection_lines(context))
        lines.extend(self._event_lines(context))

        if context.errors:
            failing = ", ".join(sorted(context.errors))
            lines.append((f"Module errors: {failing}", (80, 80, 255), 0.46))

        return lines


class MultiSink(BaseOutputSink):
    """Fans one frame out to several sinks, e.g. a window *and* the dashboard.

    Stops the pipeline if any sink asks to stop, so closing the preview window
    still ends the run while a browser is watching.
    """

    def __init__(self, sinks: Sequence[BaseOutputSink]) -> None:
        self._sinks = list(sinks)

    @property
    def sinks(self) -> List[BaseOutputSink]:
        return list(self._sinks)

    def prepare(self) -> None:
        for sink in self._sinks:
            sink.prepare()

    def emit(self, context: FrameContext, stats: PipelineStats) -> bool:
        keep_going = True
        for sink in self._sinks:
            if not sink.emit(context, stats):
                keep_going = False
        return keep_going

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:  # noqa: BLE001 - one bad close must not block the rest
                logger.warning("Sink %s failed to close cleanly", type(sink).__name__)


def create_output_sink(
    config: ProcessingConfig,
    app_name: str,
    source_description: str,
    module_names: Sequence[str],
) -> BaseOutputSink:
    """Build the sink described by ``processing.show_window``."""
    if config.show_window:
        return WindowSink(config, app_name, source_description, module_names)
    return NullSink(log_every=config.log_stats_every)
