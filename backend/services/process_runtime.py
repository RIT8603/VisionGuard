"""Run each camera in its own process, so cameras stop competing for one GIL.

With every camera on a thread of one Python process, only one thread can run
Python code at a time. The model calls release the lock, but everything around
them - decoding, resizing, tracking, drawing, JPEG encoding - does not, so a
4-camera run used about 2.5 of 12 CPU cores and left the GPU mostly waiting.

Here each camera is a worker process with its own source, modules and models.
The dashboard, alerts and face registry stay in the main process:

    worker (one per camera)                        main process
    -----------------------                        ------------
    VisionPipeline -> WebSink -> _ProxyState  ==>  out queue  -> DashboardState
                                                   (frames, results, events, phase)
    control thread <== ctrl queue  <==  identities (Face Registry), magic toggle
    viewer demand  <== shared value <== who is watching (skip unseen encoding)
    pipeline.stop  <== stop event   <== camera / module change, Ctrl+C

``WebSink`` is unchanged: it is handed a proxy that implements the four calls it
makes on a ``DashboardState`` (``publish``, ``publish_frame``,
``viewer_demand``, ``stop``) by sending them to the main process.

The cost is memory: every worker loads its own PyTorch, CUDA context and
models (roughly 1 GB of RAM each). Enabled with
``dashboard.process_per_camera``; with a single camera it is never used.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.services.hub import PHASE_ENDED, PHASE_FAILED, PHASE_RUNNING, CameraHub
from backend.services.reload import ReloadController
from computer_vision.core.config import VisionGuardConfig
from computer_vision.core.output import NullSink
from computer_vision.core.pipeline import RunLimits
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

_DEMAND = {"idle": 0, "snapshot": 1, "stream": 2}
_DEMAND_NAMES = {v: k for k, v in _DEMAND.items()}

#: How long a worker may take to load its models and open its source.
BUILD_TIMEOUT_SEC = 180.0
#: How often the main process pumps worker messages and pushes updates.
_PUMP_SEC = 0.02
#: How long a worker gets to stop after being asked before it is terminated.
STOP_GRACE_SEC = 20.0

FAILURE_SOURCE = "source"
FAILURE_FATAL = "fatal"


# --------------------------------------------------------------------------- #
# Worker side
# --------------------------------------------------------------------------- #


class _ProxyState:
    """Stands in for a ``DashboardState`` inside a worker process."""

    def __init__(self, out_q: Any, demand: Any) -> None:
        self._out = out_q
        self._demand = demand

    def publish(self, jpeg=None, summary=None, stats=None, events=None) -> None:
        self._out.put(("publish", jpeg, summary, stats, events))

    def publish_frame(self, jpeg: bytes) -> None:
        self._out.put(("frame", jpeg))

    def viewer_demand(self) -> str:
        return _DEMAND_NAMES.get(int(self._demand.value), "idle")

    def stop(self) -> None:
        self._out.put(("stopped",))


class _ProxyHub:
    """Gives ``_make_sink`` the proxy state for this worker's camera."""

    def __init__(self, cam_id: str, state: _ProxyState) -> None:
        self._id, self._state = cam_id, state

    def get(self, cam_id):
        return self._state if cam_id == self._id else None


def _error_kind(exc: BaseException) -> str:
    from computer_vision.modules.base import ModuleError
    from computer_vision.sources.base import VideoSourceError

    if isinstance(exc, ModuleError):
        return "module"
    if isinstance(exc, VideoSourceError):
        return "source"
    if isinstance(exc, NotImplementedError):
        return "not_implemented"
    if isinstance(exc, ValueError):
        return "value"
    return "fatal"


def _apply_control(message: Tuple) -> None:
    kind = message[0]
    if kind == "identities":
        from computer_vision.identity.shared import replace_shared_store

        _, pairs, threshold = message
        replace_shared_store(pairs, threshold)
    elif kind == "magic":
        from computer_vision.utils.magic_fx import set_magic_enabled

        set_magic_enabled(bool(message[1]))


def worker_main(
    spec: Any,
    cam_config: VisionGuardConfig,
    out_q: Any,
    ctrl_q: Any,
    stop: Any,
    demand: Any,
    identities: Tuple[List, float],
    magic: bool,
) -> None:
    """Entry point of one camera's process. Module-level so it can be spawned."""
    from computer_vision.utils.logger import setup_logging

    setup_logging(cam_config.app.log_level, cam_config.app.log_file)
    _apply_control(("identities", identities[0], identities[1]))
    _apply_control(("magic", magic))

    from backend.services import runtime as rt_mod
    from computer_vision.core.frame_processor import FrameProcessor
    from computer_vision.core.pipeline import VisionPipeline
    from computer_vision.modules.registry import ModuleRegistry
    from computer_vision.sources.base import VideoSourceError
    from computer_vision.sources.factory import create_video_source

    proxy = _ProxyState(out_q, demand)
    try:
        modules = ModuleRegistry.build_enabled(cam_config)
        try:
            source = create_video_source(spec.video)
        except BaseException:
            rt_mod._release(modules)
            raise
        sink = rt_mod._make_sink(cam_config, spec, modules, _ProxyHub(spec.id, proxy))
        if cam_config.dashboard.live_view:
            rt_mod._attach_live_view(sink, source, cam_config)
    except BaseException as exc:  # noqa: BLE001 - reported to the main process
        out_q.put(("build_failed", _error_kind(exc), f"{exc}"))
        return

    out_q.put((
        "ready",
        [(m.module_name, m.module_type.value, m.version) for m in modules],
    ))

    # Wait for the go-ahead (every camera built) with the run limits, applying
    # updates meanwhile - or for being told to give up.
    limits: Optional[RunLimits] = None
    while limits is None:
        if stop.value:
            rt_mod._release(modules)
            try:
                sink.close()
            except Exception:  # noqa: BLE001
                pass
            out_q.put(("done", None, None, None))
            return
        try:
            message = ctrl_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if message[0] == "go":
            limits = message[1]
        else:
            _apply_control(message)

    pipeline = VisionPipeline(
        config=cam_config,
        source=source,
        modules=modules,
        output=sink,
        processor=FrameProcessor(cam_config.processing),
    )

    def control_loop() -> None:
        while True:
            try:
                message = ctrl_q.get(timeout=0.25)
            except queue.Empty:
                if stop.value:
                    return
                continue
            if message[0] == "quit":
                return
            try:
                _apply_control(message)
            except Exception:  # noqa: BLE001 - a bad update must not stop the camera
                logger.exception("Could not apply %s update", message[0])

    def stop_watch() -> None:
        while not stop.value:
            time.sleep(0.25)
        # Keep asking until the run loop is gone: a source stuck reconnecting
        # only looks at the flag between reads.
        while True:
            pipeline.stop()
            time.sleep(0.5)

    threading.Thread(target=control_loop, name="camera-control", daemon=True).start()
    threading.Thread(target=stop_watch, name="camera-stop", daemon=True).start()

    out_q.put(("phase", PHASE_RUNNING, None))
    stats = failure = None
    try:
        stats = pipeline.run(limits)
    except VideoSourceError as exc:
        failure = (FAILURE_SOURCE, str(exc))
        out_q.put(("phase", PHASE_FAILED, str(exc)))
    except KeyboardInterrupt:
        out_q.put(("phase", PHASE_ENDED, None))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        logger.exception("Unhandled pipeline failure in camera '%s'", spec.id)
        failure = (FAILURE_FATAL, f"{type(exc).__name__}: {exc}")
        out_q.put(("phase", PHASE_FAILED, "internal error - see the log"))
    else:
        reason = pipeline.end_reason
        if pipeline.interrupted or pipeline.stop_requested:
            out_q.put(("phase", PHASE_ENDED, None))
        elif reason:
            out_q.put(("phase", PHASE_FAILED, reason))
        elif stats is not None and stats.frames_processed == 0:
            out_q.put(("phase", PHASE_FAILED, "no frames received - check the camera and its address"))
        else:
            out_q.put(("phase", PHASE_ENDED, None))

    summary = None
    if stats is not None:
        summary = {
            "frames_processed": stats.frames_processed,
            "capture_sec": stats.capture_sec,
            "overall_fps": stats.overall_fps,
            "latency_ms": stats.total_latency.avg_ms,
            "capture_ms": stats.capture_latency.avg_ms,
        }
    out_q.put(("done", summary, failure, bool(pipeline.interrupted)))
    _flush_then_let_go(out_q)


def _flush_then_let_go(out_q: Any, timeout: float = 5.0) -> None:
    """Give the main process a few seconds to receive what was sent, no more.

    Normally the queue drains at once. If the main process is busy or gone,
    waiting for it would keep this process - and its camera and GPU memory -
    alive forever, so after ``timeout`` unsent messages are abandoned.
    """
    out_q.close()
    flusher = threading.Thread(target=out_q.join_thread, daemon=True)
    flusher.start()
    flusher.join(timeout)
    if flusher.is_alive():
        out_q.cancel_join_thread()


# --------------------------------------------------------------------------- #
# Main side
# --------------------------------------------------------------------------- #


def _stats_from(summary: Optional[Dict[str, float]]):
    """A stand-in with the attributes ``backend.main._finish`` reads."""
    if summary is None:
        return None
    return SimpleNamespace(
        frames_processed=int(summary["frames_processed"]),
        capture_sec=float(summary["capture_sec"]),
        overall_fps=float(summary["overall_fps"]),
        total_latency=SimpleNamespace(avg_ms=float(summary["latency_ms"])),
        capture_latency=SimpleNamespace(avg_ms=float(summary["capture_ms"])),
    )


@dataclass
class ProcessRuntime:
    """One camera running in a worker process; quacks like ``CameraRuntime``."""

    spec: Any
    config: VisionGuardConfig
    process: Any
    out_q: Any
    ctrl_q: Any
    stop: Any
    demand: Any
    modules: List[Any] = field(default_factory=list)
    #: Never a window: several cameras always use the dashboard.
    sink: Any = field(default_factory=NullSink)
    stats: Any = None
    failure: Optional[Tuple[str, BaseException]] = None
    finished: bool = False
    interrupted: bool = False

    @property
    def id(self) -> str:
        return self.spec.id

    def request_stop(self) -> None:
        self.stop.value = 1

    def join(self, timeout: float = 10.0) -> None:
        # Our end of its queues must not block this process's exit either.
        for q in (self.out_q, self.ctrl_q):
            try:
                q.cancel_join_thread()
            except Exception:  # noqa: BLE001
                pass
        self.process.join(timeout)
        if self.process.is_alive():
            logger.warning("Camera '%s' worker did not exit in time; terminating it", self.id)
            self.process.terminate()
            self.process.join(5.0)


def _rebuild_error(kind: str, message: str) -> BaseException:
    from computer_vision.modules.base import ModuleError
    from computer_vision.sources.base import VideoSourceError

    return {
        "module": ModuleError,
        "source": VideoSourceError,
        "not_implemented": NotImplementedError,
        "value": ValueError,
    }.get(kind, ModuleError)(message)


def build_process_runtimes(
    config: VisionGuardConfig,
    specs: Sequence[Any],
    hub: CameraHub,
) -> List[ProcessRuntime]:
    """Start one worker per camera and wait until every one has built.

    Raises the same kinds of error as the in-process build (``ModuleError``,
    ``VideoSourceError``, ``ValueError``, ``NotImplementedError``) when any
    camera fails to build, after stopping the workers already started.
    """
    from backend.services.runtime import _config_for
    from computer_vision.identity.shared import snapshot_identities
    from computer_vision.utils.magic_fx import magic_enabled

    ctx = mp.get_context("spawn")
    identities = snapshot_identities()
    runtimes: List[ProcessRuntime] = []
    for spec in specs:
        cam_config = _config_for(config, spec, True)
        out_q, ctrl_q = ctx.Queue(), ctx.Queue()
        # A plain shared flag, polled by the worker - not an Event. Setting a
        # multiprocessing Event waits for every process blocked in wait() to
        # wake up; a worker that had already died (a YouTube stream refused,
        # for one) never does, and the main process hung there for good, so no
        # camera change could ever be applied again.
        stop = ctx.Value("b", 0, lock=False)
        # No lock: a worker that exits while one of its threads is reading a
        # locked Value leaves the lock held forever, and the main process then
        # blocks on its next write - every camera freezes. One int needs none.
        demand = ctx.Value("i", 0, lock=False)
        process = ctx.Process(
            target=worker_main,
            args=(spec, cam_config, out_q, ctrl_q, stop, demand, identities, magic_enabled()),
            name=f"camera-{spec.id}",
            daemon=True,
        )
        process.start()
        runtimes.append(ProcessRuntime(spec, cam_config, process, out_q, ctrl_q, stop, demand))
        logger.info("Camera '%s' started in process %s", spec.id, process.pid)

    error: Optional[BaseException] = None
    deadline = time.monotonic() + BUILD_TIMEOUT_SEC
    pending = {rt.id: rt for rt in runtimes}
    while pending and error is None:
        for cam_id, rt in list(pending.items()):
            try:
                message = rt.out_q.get(timeout=0.05)
            except queue.Empty:
                if not rt.process.is_alive():
                    error = _rebuild_error("fatal", f"Camera '{cam_id}' worker exited while starting - see the log")
                continue
            if message[0] == "ready":
                rt.modules = [
                    SimpleNamespace(module_name=n, module_type=SimpleNamespace(value=t), version=v)
                    for n, t, v in message[1]
                ]
                del pending[cam_id]
            elif message[0] == "build_failed":
                error = _rebuild_error(message[1], f"[{cam_id}] {message[2]}")
        if time.monotonic() > deadline:
            error = _rebuild_error(
                "fatal", f"Cameras did not start within {BUILD_TIMEOUT_SEC:.0f} s: {sorted(pending)}"
            )

    if error is not None:
        cleanup_process_runtimes(runtimes)
        raise error
    return runtimes


def cleanup_process_runtimes(runtimes: Sequence[ProcessRuntime]) -> None:
    """Stop workers that were started but will not be run."""
    for rt in runtimes:
        rt.request_stop()
    for rt in runtimes:
        rt.join(15.0)


def _dispatch(rt: ProcessRuntime, message: Tuple, hub: CameraHub) -> None:
    kind = message[0]
    state = hub.get(rt.id)
    if kind == "publish" and state is not None:
        _, jpeg, summary, stats, events = message
        state.publish(jpeg=jpeg, summary=summary, stats=stats, events=events)
    elif kind == "frame" and state is not None:
        state.publish_frame(message[1])
    elif kind == "stopped" and state is not None:
        state.stop()
    elif kind == "phase":
        hub.set_phase(rt.id, message[1], message[2])
    elif kind == "done":
        _, summary, failure, interrupted = message
        rt.stats = _stats_from(summary)
        if failure is not None:
            rt.failure = (failure[0], RuntimeError(failure[1]))
        rt.interrupted = bool(interrupted)
        rt.finished = True


def _start_pump_watchdog(state: Dict[str, Any], pump_thread: threading.Thread) -> None:
    """Log where the relay loop is stuck if it stops ticking (it should never)."""
    import sys
    import traceback

    def watch() -> None:
        reported = False
        while not state["done"]:
            time.sleep(2.0)
            stalled = time.monotonic() - state["tick"]
            if stalled > 10.0 and not reported:
                reported = True
                frame = sys._current_frames().get(pump_thread.ident)  # noqa: SLF001
                stack = "".join(traceback.format_stack(frame)) if frame is not None else "(no frame)"
                logger.error("Camera relay loop has not run for %.0f s. It is at:\n%s", stalled, stack)
            elif stalled < 10.0:
                reported = False

    threading.Thread(target=watch, name="camera-relay-watchdog", daemon=True).start()


def run_process_group(
    runtimes: Sequence[ProcessRuntime],
    limits: RunLimits,
    reload: Optional[ReloadController] = None,
    hub: Optional[CameraHub] = None,
) -> bool:
    """Let every worker run, relay what they publish, return True on Ctrl+C."""
    from computer_vision.identity.shared import snapshot_identities, store_version
    from computer_vision.utils.magic_fx import magic_enabled

    assert hub is not None, "process-per-camera needs the dashboard hub"
    for rt in runtimes:
        rt.failure, rt.stats, rt.finished, rt.interrupted = None, None, False, False
        rt.ctrl_q.put(("go", limits))

    version, magic = store_version(), magic_enabled()
    interrupted = False
    stop_asked_at: Optional[float] = None
    pump_state = {"tick": time.monotonic(), "done": False}
    _start_pump_watchdog(pump_state, threading.current_thread())
    try:
        while not all(rt.finished for rt in runtimes):
            pump_state["tick"] = time.monotonic()
            if reload is not None and stop_asked_at is None and reload.wait_for(0):
                stop_asked_at = time.monotonic()
                for rt in runtimes:
                    rt.request_stop()
            if stop_asked_at is not None and time.monotonic() - stop_asked_at > STOP_GRACE_SEC:
                # A camera that will not stop must not hold every other change hostage.
                for rt in runtimes:
                    if not rt.finished:
                        logger.warning("Camera '%s' did not stop in %.0f s; terminating its worker",
                                       rt.id, STOP_GRACE_SEC)
                        rt.process.terminate()
                        hub.set_phase(rt.id, PHASE_ENDED, None)
                        rt.finished = True
                break

            # Push what changed in the main process to every camera.
            if store_version() != version:
                version = store_version()
                pairs, threshold = snapshot_identities()
                for rt in runtimes:
                    rt.ctrl_q.put(("identities", pairs, threshold))
            if magic_enabled() != magic:
                magic = magic_enabled()
                for rt in runtimes:
                    rt.ctrl_q.put(("magic", magic))

            busy = False
            for rt in runtimes:
                state = hub.get(rt.id)
                if state is not None:
                    rt.demand.value = _DEMAND.get(state.viewer_demand(), 0)
                # Drain everything waiting, so frames never queue up behind.
                for _ in range(64):
                    try:
                        message = rt.out_q.get_nowait()
                    except queue.Empty:
                        break
                    busy = True
                    _dispatch(rt, message, hub)
                if not rt.finished and not rt.process.is_alive():
                    # Drain anything it managed to send before deciding it crashed.
                    try:
                        while True:
                            _dispatch(rt, rt.out_q.get_nowait(), hub)
                    except queue.Empty:
                        pass
                    if not rt.finished:
                        rt.failure = (FAILURE_FATAL, RuntimeError(
                            f"worker process exited unexpectedly (code {rt.process.exitcode})"))
                        hub.set_phase(rt.id, PHASE_FAILED, "worker process crashed - see the log")
                        rt.finished = True
            if not busy:
                time.sleep(_PUMP_SEC)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        interrupted = True
        for rt in runtimes:
            rt.request_stop()
    finally:
        pump_state["done"] = True
        for rt in runtimes:
            try:
                rt.ctrl_q.put(("quit",))
            except Exception:  # noqa: BLE001
                pass
        for rt in runtimes:
            rt.join(15.0)
    return interrupted or any(rt.interrupted for rt in runtimes)
