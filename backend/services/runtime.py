"""Building and running the cameras of one VisionGuard run.

Each camera is a complete, independent :class:`VisionPipeline`: its own video
source, its own module instances (so trackers, zones and dwell timers never
mix between cameras) and its own output sink. What they share is the detector
models - the factory hands every camera the same loaded model, and
:class:`~computer_vision.inference.shared.SharedDetector` serialises the calls -
so ten cameras do not mean ten copies of YOLO in memory.

A single camera runs on the calling thread exactly as it always did (including
the preview window and Ctrl+C handling). Several cameras each get a thread, and
one camera failing - an unreachable RTSP URL, a pulled USB cable - stops only
that camera.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from backend.services.cameras import mask_url
from backend.services.hub import PHASE_ENDED, PHASE_FAILED, PHASE_RUNNING, CameraHub
from backend.services.reload import ReloadController
from backend.services.web_sink import WebSink
from computer_vision.core.config import CameraConfig, ModuleConfig, VideoConfig, VisionGuardConfig
from computer_vision.core.frame_processor import FrameProcessor
from computer_vision.core.output import (
    BaseOutputSink,
    DisplayUnavailableError,
    MultiSink,
    NullSink,
    create_output_sink,
)
from computer_vision.core.pipeline import RunLimits, VisionPipeline
from computer_vision.modules.base import BaseAIModule
from computer_vision.modules.registry import ModuleRegistry
from computer_vision.sources.base import BaseVideoSource, VideoSourceError
from computer_vision.sources.factory import create_video_source
from computer_vision.utils.logger import get_logger
from computer_vision.utils.timing import PipelineStats

logger = get_logger(__name__)

#: How often the supervising thread looks at its workers (and at Ctrl+C).
_POLL_SEC = 0.25

FAILURE_SOURCE = "source"
FAILURE_FATAL = "fatal"


# --------------------------------------------------------------------------- #
# Specs: what to run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraSpec:
    """One camera to run: an id, a label, and its complete video settings."""

    id: str
    name: str
    video: VideoConfig
    #: The stored entry, or None for the single camera described by ``video:``.
    camera: Optional[CameraConfig] = None


def specs_from_video(config: VisionGuardConfig) -> List[CameraSpec]:
    """The one camera described by the ``video:`` section (the pre-multi-camera way)."""
    cam_id = config.video.resolved_source_id()
    return [CameraSpec(id=cam_id, name=cam_id, video=config.video)]


def specs_from_cameras(config: VisionGuardConfig, cameras: Sequence[CameraConfig]) -> List[CameraSpec]:
    return [
        CameraSpec(id=cam.id, name=cam.label, video=cam.to_video(config.video), camera=cam)
        for cam in cameras
    ]


def describe_specs(specs: Sequence[CameraSpec]) -> Tuple[List[CameraConfig], dict]:
    """``(cameras, details)`` for the camera manager: what is running, safely described.

    A ``video:`` camera is expressed as a :class:`CameraConfig` when it fits
    one (so a hidden password can be restored when the dashboard sends the
    list back); otherwise only its masked public description is kept.
    """
    cameras: List[CameraConfig] = []
    details: dict = {}
    for spec in specs:
        if spec.camera is not None:
            cameras.append(spec.camera)
            continue
        video = spec.video
        try:
            cameras.append(
                CameraConfig(
                    id=spec.id,
                    source=video.source,
                    device_id=video.device_id,
                    rtsp_url=video.rtsp_url,
                    path=video.path,
                    loop=video.loop,
                )
            )
        except ValueError:
            details[spec.id] = {
                "id": spec.id,
                "name": spec.name,
                "source": video.source.value,
                "device_id": video.device_id,
                "rtsp_url": mask_url(video.rtsp_url),
                "path": video.path,
                "loop": video.loop,
            }
    return cameras, details


# --------------------------------------------------------------------------- #
# Runtime: one built camera
# --------------------------------------------------------------------------- #


@dataclass
class CameraRuntime:
    spec: CameraSpec
    config: VisionGuardConfig
    modules: List[BaseAIModule]
    source: BaseVideoSource
    sink: BaseOutputSink
    pipeline: Optional[VisionPipeline] = None
    stats: Optional[PipelineStats] = None
    #: ``(FAILURE_SOURCE | FAILURE_FATAL, exception)`` when the pipeline raised.
    failure: Optional[Tuple[str, BaseException]] = field(default=None)

    @property
    def id(self) -> str:
        return self.spec.id


def with_enabled_modules(config: VisionGuardConfig, enabled: Iterable[str]) -> VisionGuardConfig:
    """A copy of ``config`` with exactly ``enabled`` turned on, nothing else touched.

    Built with ``model_copy`` rather than a dump-and-revalidate round trip so
    the dashboard's ``auth_token`` (a ``SecretStr``) is carried over as-is
    instead of risking its masked ``"**********"`` JSON form leaking in.
    """
    wanted = set(enabled)
    new_modules = {
        name: module_cfg.model_copy(update={"enabled": name in wanted})
        for name, module_cfg in config.modules.items()
    }
    # The API only accepts names ModuleRegistry already knows, but a name the
    # config file never listed at all has no existing ModuleConfig to copy.
    for name in wanted - set(new_modules):
        new_modules[name] = ModuleConfig(enabled=True)
    return config.model_copy(update={"modules": new_modules})


def _config_for(config: VisionGuardConfig, spec: CameraSpec, multi: bool) -> VisionGuardConfig:
    # A camera with its own module list runs exactly that; the others follow
    # the shared selection already in ``config``. Parameters stay shared.
    if spec.camera is not None and spec.camera.modules is not None:
        config = with_enabled_modules(config, spec.camera.modules)
    update = {"video": spec.video}
    if multi and config.processing.show_window:
        # One OpenCV window per thread is not workable (GUI calls belong on the
        # main thread), and the dashboard is the view for several cameras anyway.
        update["processing"] = config.processing.model_copy(update={"show_window": False})
    return config.model_copy(update=update)


def _make_sink(
    cam_config: VisionGuardConfig,
    spec: CameraSpec,
    modules: Sequence[BaseAIModule],
    hub: Optional[CameraHub],
) -> BaseOutputSink:
    sink = create_output_sink(
        cam_config.processing,
        app_name=cam_config.app.name,
        source_description=spec.id,
        module_names=[module.module_name for module in modules],
    )
    try:
        sink.prepare()
    except DisplayUnavailableError as exc:
        logger.warning("%s", exc)
        logger.warning("Falling back to headless output")
        sink = NullSink(log_every=cam_config.processing.log_stats_every)

    state = hub.get(spec.id) if hub is not None else None
    if state is None:
        return sink

    web = WebSink(
        state,
        cam_config.processing,
        stream_fps=cam_config.dashboard.stream_fps,
        jpeg_quality=cam_config.dashboard.jpeg_quality,
        sync=cam_config.dashboard.live_view_sync,
    )
    # A NullSink alongside the dashboard is just duplicate logging.
    return web if isinstance(sink, NullSink) else MultiSink([sink, web])


def _attach_live_view(sink: BaseOutputSink, source: Any, cam_config: VisionGuardConfig) -> None:
    """Let the dashboard show the camera at its own rate (see ``WebSink``).

    Only for a source that can be read independently of the pipeline; the
    others keep showing analysed frames, with no extra thread.
    """
    peek = getattr(type(source), "peek", None)
    if peek is None or peek is BaseVideoSource.peek:
        return
    sinks = sink.sinks if isinstance(sink, MultiSink) else [sink]
    for candidate in sinks:
        if isinstance(candidate, WebSink):
            candidate.attach_live_view(
                source.peek, FrameProcessor(cam_config.processing).prepare
            )


def build_runtimes(
    config: VisionGuardConfig,
    specs: Sequence[CameraSpec],
    hub: Optional[CameraHub] = None,
) -> List[CameraRuntime]:
    """Build modules, source and sink for every camera.

    Raises whatever building raises (``ModuleError``, ``NotImplementedError``,
    ``ValueError``, ``VideoSourceError``), after releasing everything already
    built, so a failed attempt leaves nothing loaded.
    """
    multi = len(specs) > 1
    runtimes: List[CameraRuntime] = []
    try:
        for spec in specs:
            cam_config = _config_for(config, spec, multi)
            modules = ModuleRegistry.build_enabled(cam_config)
            try:
                source = create_video_source(spec.video)
            except BaseException:
                _release(modules)
                raise
            sink = _make_sink(cam_config, spec, modules, hub)
            if cam_config.dashboard.live_view:
                _attach_live_view(sink, source, cam_config)
            runtimes.append(CameraRuntime(spec, cam_config, modules, source, sink))
    except BaseException:
        cleanup_runtimes(runtimes)
        raise
    return runtimes


def _release(modules: Sequence[BaseAIModule]) -> None:
    for module in modules:
        try:
            module.cleanup()
        except Exception:  # noqa: BLE001 - best effort on the way out
            logger.warning("cleanup() failed for module '%s'", module.module_name)


def cleanup_runtimes(runtimes: Sequence[CameraRuntime]) -> None:
    """Release what was built when startup stops before the pipelines run."""
    for rt in runtimes:
        _release(rt.modules)
        try:
            rt.sink.close()
        except Exception:  # noqa: BLE001
            logger.warning("Output sink for camera '%s' failed to close cleanly", rt.id)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def _run_one(rt: CameraRuntime, limits: RunLimits, hub: Optional[CameraHub]) -> None:
    """Run one camera's pipeline to its end and record how it ended."""

    def phase(value: str, error: str | None = None) -> None:
        if hub is not None:
            hub.set_phase(rt.id, value, error)

    phase(PHASE_RUNNING)
    assert rt.pipeline is not None
    try:
        rt.stats = rt.pipeline.run(limits)
    except VideoSourceError as exc:
        # The message already has any password masked (see RTSPSource).
        rt.failure = (FAILURE_SOURCE, exc)
        phase(PHASE_FAILED, str(exc))
    except Exception as exc:  # noqa: BLE001 - one camera must not take the others down
        logger.exception("Unhandled pipeline failure in camera '%s'", rt.id)
        rt.failure = (FAILURE_FATAL, exc)
        phase(PHASE_FAILED, "internal error - see the log")
    else:
        # A source that died mid-run is contained by the pipeline (so it can
        # shut down cleanly); its reason is kept on the pipeline, not raised.
        reason = rt.pipeline.end_reason
        if rt.pipeline.interrupted or rt.pipeline.stop_requested:
            phase(PHASE_ENDED)
        elif reason:
            phase(PHASE_FAILED, reason)
        elif rt.stats is not None and rt.stats.frames_processed == 0:
            phase(PHASE_FAILED, "no frames received - check the camera and its address")
        else:
            phase(PHASE_ENDED)


def _watch_for_reload(
    reload: ReloadController, pipelines: Sequence[VisionPipeline], done: threading.Event
) -> None:
    """Stop every pipeline as soon as a change is requested, or exit once ``done``.

    ``pipeline.stop()`` only flips a flag the run loop checks between frames,
    so calling it from here is safe.
    """
    while not done.is_set():
        if reload.wait_for(_POLL_SEC):
            for pipeline in pipelines:
                pipeline.stop()
            return


def run_group(
    runtimes: Sequence[CameraRuntime],
    limits: RunLimits,
    reload: Optional[ReloadController] = None,
    hub: Optional[CameraHub] = None,
) -> bool:
    """Run every camera until all have stopped. Returns True on Ctrl+C.

    One camera runs on this thread. Several run on their own threads while this
    thread supervises them, so Ctrl+C reaches it and stops them all.
    """
    for rt in runtimes:
        rt.failure = None
        rt.stats = None
        rt.pipeline = VisionPipeline(
            config=rt.config,
            source=rt.source,
            modules=rt.modules,
            output=rt.sink,
            processor=FrameProcessor(rt.config.processing),
        )

    done = threading.Event()
    if reload is not None:
        threading.Thread(
            target=_watch_for_reload,
            args=(reload, [rt.pipeline for rt in runtimes], done),
            name="visionguard-reload-watch",
            daemon=True,
        ).start()

    try:
        if len(runtimes) == 1:
            _run_one(runtimes[0], limits, hub)
            return bool(runtimes[0].pipeline and runtimes[0].pipeline.interrupted)

        threads = [
            threading.Thread(
                target=_run_one, args=(rt, limits, hub), name=f"camera-{rt.id}", daemon=True
            )
            for rt in runtimes
        ]
        for thread in threads:
            thread.start()

        try:
            while any(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(_POLL_SEC)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            for rt in runtimes:
                rt.pipeline.stop()
            for thread in threads:
                thread.join(timeout=10.0)
            return True
        return False
    finally:
        done.set()
