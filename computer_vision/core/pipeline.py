"""The VisionGuard processing pipeline — the Vision Core orchestrator.

    VideoSource -> FrameProcessor -> FrameContext -> enabled modules -> OutputSink

This file contains no module-specific logic and no reference to any concrete
AI module or model. It iterates whatever list of :class:`BaseAIModule` it was
given, in whatever order the registry resolved. Adding fire detection later
changes configuration and the registry catalogue — not this file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence

from computer_vision.core.config import VisionGuardConfig
from computer_vision.core.context import FrameContext
from computer_vision.core.frame_processor import FrameProcessor
from computer_vision.core.output import BaseOutputSink
from computer_vision.modules.base import BaseAIModule, ModuleStatus
from computer_vision.sources.base import BaseVideoSource, SourceExhausted, VideoSourceError
from computer_vision.utils.logger import get_logger
from computer_vision.utils.timing import FpsMeter, LatencyMeter, PipelineStats, Stopwatch

logger = get_logger(__name__)


@dataclass
class RunLimits:
    """Optional stopping conditions. Used by smoke tests and benchmarks."""

    max_frames: int | None = None
    max_seconds: float | None = None

    def reached(self, frames: int, elapsed: float) -> bool:
        if self.max_frames is not None and frames >= self.max_frames:
            return True
        if self.max_seconds is not None and elapsed >= self.max_seconds:
            return True
        return False


class VisionPipeline:
    """Drives frames from a source through the enabled modules to a sink."""

    def __init__(
        self,
        config: VisionGuardConfig,
        source: BaseVideoSource,
        modules: Sequence[BaseAIModule],
        output: BaseOutputSink,
        processor: FrameProcessor | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.modules: List[BaseAIModule] = list(modules)
        self.output = output
        self.processor = processor or FrameProcessor(config.processing)

        self.stats = PipelineStats(
            fps_meter=FpsMeter(config.processing.stats_window),
            total_latency=LatencyMeter(config.processing.stats_window),
            capture_latency=LatencyMeter(config.processing.stats_window),
        )

        self._running = False
        self._stop_requested = False
        #: True when the last run ended because of Ctrl+C. The run loop swallows
        #: the interrupt (so the pipeline can shut down cleanly); this lets a
        #: caller that supervises several pipelines still tell it happened.
        self.interrupted = False
        self._max_consecutive_failures = 10
        self._min_frame_interval = (
            1.0 / config.processing.max_fps if config.processing.max_fps > 0 else 0.0
        )
        self._next_frame_at = 0.0
        self._last_frame_at = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """Ask the loop to finish after the current frame."""
        self._stop_requested = True

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self, limits: RunLimits | None = None) -> PipelineStats:
        """Run until the source ends, a limit is hit, or a stop is requested."""
        limits = limits or RunLimits()
        self._running = True
        self._stop_requested = False
        self.interrupted = False

        try:
            # Inside the try: if opening fails, modules, the output sink and
            # any half-opened source are still shut down by `finally`.
            info = self.source.open()
            logger.info("Pipeline started on %s", info.as_dict())

            # Start the clock after the source is up: opening a camera can
            # take seconds, and that must not eat into a --duration budget.
            started = time.perf_counter()
            self.stats.started_at = started
            self._last_frame_at = started

            while not self._stop_requested:
                if limits.reached(self.stats.frames_processed, time.perf_counter() - started):
                    logger.info("Run limit reached")
                    break

                self._throttle()

                packet, stream_ended = self._read_frame()
                if stream_ended:
                    break
                if packet is None:
                    # No frame available right now. That is normal on a live
                    # network stream, so keep going - unless the source has
                    # closed itself (it can produce nothing more) or nothing
                    # has arrived for a long time.
                    if not self.source.is_open:
                        logger.info("Video source closed")
                        break
                    if self._stalled():
                        break
                    continue

                self._last_frame_at = time.perf_counter()
                self.stats.frames_read += 1

                context = self._process(packet)

                if not self.output.emit(context, self.stats):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.interrupted = True
        finally:
            self._running = False
            self._shutdown()

        logger.info("Pipeline stats: %s", self.stats.as_dict())
        return self.stats

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    def _read_frame(self):
        """Return ``(packet, stream_ended)``.

        ``(None, False)`` means "nothing right now, try again" - distinct from
        ``(None, True)`` which means the source is finished or broken.
        """
        try:
            return self.source.read(), False
        except SourceExhausted:
            logger.info("Video source exhausted")
            return None, True
        except VideoSourceError as exc:
            logger.error("Video source failure: %s", exc)
            return None, True

    def _stalled(self) -> bool:
        """True when no frame has arrived for longer than the stall timeout."""
        timeout = self.config.processing.stall_timeout_sec
        if timeout <= 0:
            return False

        if self.source.is_recovering:
            # The source is working through its own reconnect budget; it
            # raises when that is exhausted. Restart the idle clock so the
            # stall check applies again once recovery ends.
            self._last_frame_at = time.perf_counter()
            return False

        idle_for = time.perf_counter() - self._last_frame_at
        if idle_for < timeout:
            return False

        logger.error(
            "No frame received for %.1fs (stall_timeout_sec=%.1f) - stopping",
            idle_for,
            timeout,
        )
        return True

    def _process(self, packet) -> FrameContext:
        with Stopwatch() as watch:
            context = self.processor.build_context(packet)
            self._run_modules(context)

        if self.stats.first_frame_at is None:
            self.stats.first_frame_at = time.perf_counter()

        self.stats.frames_processed += 1
        self.stats.total_latency.add(watch.elapsed_ms)
        self.stats.capture_latency.add(packet.capture_ms)
        self.stats.fps_meter.tick()
        if context.errors:
            self.stats.frames_failed += 1

        context.metadata["latency_ms"] = round(watch.elapsed_ms, 2)
        context.metadata["fps"] = round(self.stats.fps, 2)
        return context

    def _run_modules(self, context: FrameContext) -> None:
        """Execute enabled modules in registry-resolved order.

        A module that raises is contained by ``safe_process``; one that keeps
        raising is dropped so a single broken optional module cannot degrade
        the whole platform.
        """
        for module in self.modules:
            if module.status is not ModuleStatus.READY:
                continue

            result = module.safe_process(context)
            self.stats.record_module(module.module_name, result.latency_ms)

            if module.consecutive_failures >= self._max_consecutive_failures:
                logger.error(
                    "Disabling module '%s' after %d consecutive failures",
                    module.module_name,
                    module.consecutive_failures,
                )
                module.status = ModuleStatus.FAILED

    def _throttle(self) -> None:
        if self._min_frame_interval <= 0.0:
            return
        now = time.perf_counter()
        if self._next_frame_at and now < self._next_frame_at:
            time.sleep(self._next_frame_at - now)
            now = time.perf_counter()
        self._next_frame_at = now + self._min_frame_interval

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #

    def _shutdown(self) -> None:
        for module in self.modules:
            try:
                module.cleanup()
            except Exception:  # noqa: BLE001 - one bad cleanup must not block the rest
                logger.warning("cleanup() failed for module '%s'", module.module_name)

        try:
            self.output.close()
        except Exception:  # noqa: BLE001
            logger.warning("Output sink failed to close cleanly")

        try:
            self.source.release()
        except Exception:  # noqa: BLE001
            logger.warning("Video source failed to release cleanly")
