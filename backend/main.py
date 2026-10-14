"""VisionGuard AI entry point.

    python -m backend.main
    python -m backend.main --source webcam --device-id 1
    python -m backend.main --no-window --max-frames 120

Responsibilities: parse flags, load configuration, wire the components, run the
pipeline, and turn failures into clear messages and exit codes. No vision logic
lives here.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from computer_vision import __phase__, __version__
from computer_vision.core.config import (
    ConfigError,
    VisionGuardConfig,
    load_config,
    resolve_config_path,
)
from computer_vision.core.output import BaseOutputSink, MultiSink, WindowSink
from backend.api.server import DashboardServer, DashboardUnavailableError
from backend.services.alerts import AlertCenter
from backend.services.camera_manager import CameraManager
from backend.services.cameras import cameras_file_for, load_camera_file, save_camera_file
from backend.services.hub import CameraHub
from backend.services.reload import ReloadController
from backend.services.settings import SettingsManager, with_module_settings
from backend.services.runtime import (
    FAILURE_FATAL,
    CameraRuntime,
    CameraSpec,
    build_runtimes,
    cleanup_runtimes,
    describe_specs,
    run_group,
    specs_from_cameras,
    specs_from_video,
    with_enabled_modules,
)
from computer_vision.core.pipeline import RunLimits
from computer_vision.modules.base import ModuleError
from computer_vision.modules.registry import ModuleRegistry
from computer_vision.sources.base import VideoSourceError
from computer_vision.sources.factory import available_source_types
from computer_vision.utils.logger import get_logger, setup_logging

logger = get_logger("visionguard")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_SOURCE = 3
EXIT_MODULE = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionguard",
        description="VisionGuard AI - modular real-time computer vision platform",
    )
    parser.add_argument("--config", help="Path to config.yaml (default: configs/config.yaml)")
    parser.add_argument("--source", choices=["webcam", "file", "rtsp", "youtube"], help="Override video.source")
    parser.add_argument("--device-id", type=int, help="Override video.device_id")
    parser.add_argument("--video-path", help="Override video.path (file source)")
    parser.add_argument("--loop", action="store_true", help="Loop the video file")
    parser.add_argument(
        "--playback-speed",
        type=float,
        help="File playback rate vs native fps (0 = unthrottled, 1.0 = real time)",
    )
    parser.add_argument("--start-frame", type=int, help="Seek here before the first frame")
    parser.add_argument("--rtsp-url", help="Override video.rtsp_url (rtsp source)")
    parser.add_argument("--youtube-url", help="Override video.youtube_url (youtube source)")
    parser.add_argument("--width", type=int, help="Override requested capture width")
    parser.add_argument("--height", type=int, help="Override requested capture height")
    parser.add_argument("--target-width", type=int, help="Override processing.target_width")
    parser.add_argument("--mirror", action="store_true", help="Horizontally flip the frame")
    parser.add_argument(
        "--no-window", action="store_true", help="Run headless (no preview window)"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Serve the web dashboard"
    )
    parser.add_argument("--dashboard-port", type=int, help="Dashboard port")
    parser.add_argument("--dashboard-host", help="Dashboard bind address")
    parser.add_argument("--log-level", help="DEBUG / INFO / WARNING / ERROR / CRITICAL")
    parser.add_argument("--max-frames", type=int, help="Stop after N processed frames")
    parser.add_argument("--duration", type=float, help="Stop after N seconds")
    parser.add_argument(
        "--list-modules", action="store_true", help="Show module catalogue and exit"
    )
    parser.add_argument(
        "--list-sources", action="store_true", help="Show source catalogue and exit"
    )
    parser.add_argument("--version", action="version", version=f"VisionGuard AI {__version__}")
    return parser


def infer_source(args: argparse.Namespace) -> str | None:
    """Pick the source implied by the flags, when it was not stated.

    ``--video-path clip.mp4`` obviously means "use the file source", so the
    user should not also have to type ``--source file``.
    """
    if args.source:
        return args.source
    if args.video_path:
        return "file"
    if args.rtsp_url:
        return "rtsp"
    if args.youtube_url:
        return "youtube"
    return None


def cli_overrides(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Translate CLI flags into nested config overrides (None values ignored)."""
    return {
        "app": {"log_level": args.log_level},
        "video": {
            "source": infer_source(args),
            "device_id": args.device_id,
            "path": args.video_path,
            "rtsp_url": args.rtsp_url,
            "youtube_url": args.youtube_url,
            "width": args.width,
            "height": args.height,
            "loop": True if args.loop else None,
            "playback_speed": args.playback_speed,
            "start_frame": args.start_frame,
        },
        "processing": {
            "target_width": args.target_width,
            "show_window": False if args.no_window else None,
            "mirror": True if args.mirror else None,
        },
        "dashboard": {
            "enabled": True if args.dashboard else None,
            "host": args.dashboard_host,
            "port": args.dashboard_port,
        },
    }


def _publish_info(
    hub: CameraHub, config: VisionGuardConfig, runtimes: List[CameraRuntime]
) -> None:
    """Describe each camera's run to its dashboard state (modules, source, size)."""
    resolution = (
        f"{config.processing.target_width}px wide" if config.processing.target_width else "native"
    )
    for rt in runtimes:
        state = hub.get(rt.id)
        if state is None:
            continue
        state.update_info({
            "app": config.app.name,
            "version": __version__,
            "source_type": rt.spec.video.source.value,
            "source_id": rt.spec.id,
            "camera_id": rt.spec.id,
            "camera_name": rt.spec.name,
            "resolution": resolution,
            "modules": [
                {"name": m.module_name, "type": m.module_type.value, "version": m.version}
                for m in rt.modules
            ],
        })


def start_dashboard(
    config: VisionGuardConfig,
    hub: CameraHub,
    reload: ReloadController,
    settings: SettingsManager | None = None,
) -> Tuple["DashboardServer", CameraManager]:
    """Start the dashboard server over ``hub``'s cameras."""
    from backend.api.server import AccessPolicy, DashboardServer

    manager = CameraManager(
        hub,
        reload,
        max_cameras=config.dashboard.max_cameras,
        editable=config.dashboard.allow_camera_edit,
    )

    token = config.dashboard.auth_token
    policy = AccessPolicy.for_bind(
        config.dashboard.host,
        token=token.get_secret_value() if token is not None else None,
        extra_hosts=config.dashboard.allowed_hosts,
    )
    default_state = hub.get(None)
    assert default_state is not None  # the hub always has at least one camera
    server = DashboardServer(
        default_state,
        config.dashboard.host,
        config.dashboard.port,
        policy=policy,
        max_streams=config.dashboard.max_streams,
        stream_fps=config.dashboard.stream_fps,
        reload=reload,
        known_modules=frozenset(ModuleRegistry.known_names()),
        hub=hub,
        cameras=manager,
        settings=settings,
    )
    server.start()  # raises OSError if the port is taken

    if policy.token_generated:
        # Printed to the console only - never to the log file - because the
        # URL itself grants access for this run.
        print(
            "\n[dashboard] Network access requires a token. Open:\n"
            f"    {server.url_with_token()}\n"
            "Set VISIONGUARD_DASHBOARD_TOKEN to use a fixed token instead.\n",
            file=sys.stderr,
        )
    elif policy.token:
        print(
            f"\n[dashboard] Token required. Open {server.url}/?token=<your token>\n",
            file=sys.stderr,
        )

    return server, manager


def initial_cameras(
    args: argparse.Namespace, config: VisionGuardConfig, config_path: Path
) -> Tuple[List[CameraSpec], str]:
    """Which cameras to start with, and where they came from.

    Returns ``(specs, origin)`` with origin ``"cli"`` (a camera named on the
    command line), ``"file"`` (``configs/cameras.yaml``) or ``"video"`` (the
    ``video:`` section). A source given on the command line always wins, so an
    existing ``--rtsp-url ...`` command keeps doing exactly what it did.

    Raises:
        ConfigError: ``cameras.yaml`` exists but is unusable.
    """
    explicit = infer_source(args) is not None or args.device_id is not None
    stored = load_camera_file(
        cameras_file_for(config_path), max_cameras=config.dashboard.max_cameras
    )
    if explicit:
        if stored is not None:
            logger.info(
                "A camera was given on the command line; ignoring %s",
                cameras_file_for(config_path).name,
            )
        return specs_from_video(config), "cli"
    if stored is not None:
        return specs_from_cameras(config, stored), "file"
    return specs_from_video(config), "video"


# Kept under its old name: the module-reload path and its tests use it.
_with_enabled_modules = with_enabled_modules


def _persist_enabled_modules(config_path: Path, enabled: Iterable[str]) -> None:
    """Best-effort: flip ``enabled:`` for each known module block in the YAML file.

    Comments and formatting are left untouched - this only ever replaces the
    one boolean already on the line directly under a ``  <name>:`` block, so
    a hand-edited config keeps its shape after a change made from the
    dashboard. A module with no such block in this particular file (not every
    config lists every known module) is simply left alone.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist module selection to %s: %s", config_path, exc)
        return

    wanted = set(enabled)
    for name in ModuleRegistry.known_names():
        value = "true" if name in wanted else "false"
        pattern = re.compile(rf"(?m)^(  {re.escape(name)}:\n {{4}}enabled:\s*)\S+")
        text = pattern.sub(lambda m: m.group(1) + value, text, count=1)

    try:
        config_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist module selection to %s: %s", config_path, exc)


def _persist_module_settings(config_path: Path, changes: Dict[str, Dict[str, Any]]) -> None:
    """Best-effort: rewrite changed settings in place in the YAML file.

    Like :func:`_persist_enabled_modules` this only replaces the value on a
    ``    <key>: <value>`` line that already exists inside the module's block,
    keeping any trailing comment, so the file keeps its shape. A setting the
    file does not list stays in effect for this run only.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist module settings to %s: %s", config_path, exc)
        return

    for name, values in changes.items():
        block = re.search(rf"(?m)^  {re.escape(name)}:[ \t]*\n((?:(?:    .*|[ \t]*)\n)*)", text)
        if block is None:
            continue
        body = block.group(1)
        for key, value in values.items():
            shown = ("true" if value else "false") if isinstance(value, bool) else str(value)
            line = re.compile(rf"(?m)^(    {re.escape(key)}:[ \t]*)[^#\n]*?([ \t]*(?:#.*)?)$")
            body = line.sub(lambda m: m.group(1) + shown + m.group(2), body, count=1)
        text = text[: block.start(1)] + body + text[block.end(1):]

    try:
        config_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist module settings to %s: %s", config_path, exc)


def _has_window(sink: BaseOutputSink) -> bool:
    """True when a live preview window is part of ``sink``."""
    if isinstance(sink, WindowSink):
        return True
    if isinstance(sink, MultiSink):
        return any(_has_window(inner) for inner in sink.sinks)
    return False


def _warn_if_cli_secret(args: argparse.Namespace) -> None:
    """A password on the command line leaks via the process list and history."""
    if not args.rtsp_url:
        return
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(args.rtsp_url)
        has_secret = bool(parts.password) or "pass" in parts.query.lower() or "token" in parts.query.lower()
    except ValueError:
        has_secret = "@" in args.rtsp_url
    if has_secret:
        logger.warning(
            "--rtsp-url contains credentials; command lines are visible to other "
            "users (process list) and saved in shell history. Prefer RTSP_URL "
            "in .env or the environment."
        )


def print_catalogue(config: VisionGuardConfig | None) -> None:
    print(f"\nVisionGuard AI {__version__} - {__phase__}\n")

    print("Video sources:")
    for name, implemented in available_source_types().items():
        mark = "available" if implemented else "not implemented yet"
        print(f"  {name:<10} {mark}")

    print("\nAI modules:")
    known = ModuleRegistry.known_names()
    if known:
        for name in known:
            print(f"  {name}")
    else:
        print("  (none implemented yet - Phase 1 ships the interface only)")

    if config and config.modules:
        print("\nConfigured in config.yaml:")
        for name, module_cfg in config.modules.items():
            state = "enabled" if module_cfg.enabled else "disabled"
            known_mark = "" if ModuleRegistry.is_known(name) else "  (not implemented yet)"
            print(f"  {name:<20} {state}{known_mark}")
    print()


def _fail_message(prefix: str, exc: BaseException) -> None:
    print(f"\n[{prefix}] {exc}\n", file=sys.stderr)


def _build_or_report(
    config: VisionGuardConfig, specs: List[CameraSpec], hub: CameraHub | None
) -> Tuple[List[CameraRuntime] | None, int]:
    """``(runtimes, EXIT_OK)`` or ``(None, exit_code)`` with the error already printed."""
    try:
        return build_runtimes(config, specs, hub), EXIT_OK
    except ModuleError as exc:
        _fail_message("module error", exc)
        return None, EXIT_MODULE
    except NotImplementedError as exc:
        _fail_message("source not available", exc)
        return None, EXIT_SOURCE
    except (ValueError, VideoSourceError) as exc:
        _fail_message("source error", exc)
        return None, EXIT_SOURCE


def run(args: argparse.Namespace) -> int:
    # ---- configuration ----
    try:
        config = load_config(args.config, overrides=cli_overrides(args))
    except ConfigError as exc:
        print(f"\n[config error] {exc}\n", file=sys.stderr)
        return EXIT_CONFIG

    setup_logging(config.app.log_level, config.app.log_file)

    if args.list_modules or args.list_sources:
        print_catalogue(config)
        return EXIT_OK

    logger.info("%s %s - %s", config.app.name, __version__, __phase__)
    _warn_if_cli_secret(args)

    # ---- cameras ----
    config_path = resolve_config_path(args.config)
    try:
        specs, origin = initial_cameras(args, config, config_path)
    except ConfigError as exc:
        print(f"\n[config error] {exc}\n", file=sys.stderr)
        return EXIT_CONFIG
    if len(specs) > 1:
        logger.info("Running %d cameras: %s", len(specs), ", ".join(s.id for s in specs))

    hub = CameraHub(
        max_events=config.dashboard.max_events,
        alerts=AlertCenter.from_config(config.dashboard.alerts),
    )
    hub.configure([(s.id, s.name, s.video.source.value) for s in specs])
    use_dashboard = config.dashboard.enabled

    # ---- modules, sources, outputs (per camera) ----
    runtimes, code = _build_or_report(config, specs, hub if use_dashboard else None)
    if runtimes is None:
        return code

    dashboard: DashboardServer | None = None
    reload: ReloadController | None = None
    manager: CameraManager | None = None
    settings: SettingsManager | None = None
    if use_dashboard:
        reload = ReloadController()
        settings = SettingsManager(reload, editable=config.dashboard.allow_settings_edit)
        try:
            dashboard, manager = start_dashboard(config, hub, reload, settings)
        except DashboardUnavailableError as exc:
            cleanup_runtimes(runtimes)
            print(f"\n[dashboard error] {exc}\n", file=sys.stderr)
            return EXIT_CONFIG
        except OSError as exc:
            cleanup_runtimes(runtimes)
            print(
                f"\n[dashboard error] Could not bind "
                f"{config.dashboard.host}:{config.dashboard.port}: {exc}\n"
                "Another process may already be using that port.\n",
                file=sys.stderr,
            )
            return EXIT_CONFIG

    # ---- run ----
    # A loop, not one call: a change requested from the dashboard
    # (POST /api/modules or /api/cameras) stops the running pipelines between
    # frames, and the loop below rebuilds the cameras, leaving the dashboard
    # server running throughout. Any other reason the pipelines stop (the
    # sources ended, a limit was hit, Ctrl+C, a fatal error) leaves no pending
    # change, so the loop exits for good - exactly as before this loop existed.
    limits = RunLimits(max_frames=args.max_frames, max_seconds=args.duration)
    has_limits = limits.max_frames is not None or limits.max_seconds is not None
    cameras_file = cameras_file_for(config_path)
    exit_code = EXIT_OK

    while True:
        if use_dashboard:
            _publish_info(hub, config, runtimes)
            hub.resume_all()
        if manager is not None:
            current, details = describe_specs(specs)
            manager.set_current(
                current, details, shared_modules=config.enabled_module_names()
            )
        if settings is not None:
            settings.set_current(config)

        if _has_window(runtimes[0].sink):
            logger.info("Press 'q' or ESC in the preview window to stop")
        if dashboard is not None:
            logger.info("Open %s to watch. Ctrl+C to stop.", dashboard.url)

        interrupted = run_group(runtimes, limits, reload, hub if use_dashboard else None)

        modules_req = reload.pop() if reload is not None else None
        cameras_req = reload.pop_cameras() if reload is not None else None
        settings_req = reload.pop_settings() if reload is not None else None

        if modules_req is None and cameras_req is None and settings_req is None:
            # Nothing asked for a change. With several cameras and a dashboard,
            # staying up lets the operator fix a bad camera from the page
            # instead of the whole program exiting under them.
            can_idle = (
                dashboard is not None
                and reload is not None
                and (len(specs) > 1 or origin == "file")
                and not has_limits
                and not interrupted
            )
            if not can_idle:
                break
            logger.warning(
                "All cameras have stopped. Waiting for a new camera or module selection "
                "from the dashboard (Ctrl+C to quit)."
            )
            try:
                while not reload.wait_for(0.5):
                    pass
            except KeyboardInterrupt:
                break
            modules_req = reload.pop()
            cameras_req = reload.pop_cameras()
            settings_req = reload.pop_settings()
            if modules_req is None and cameras_req is None and settings_req is None:
                break

        # ---- apply the requested change ----
        if modules_req is not None:
            logger.info(
                "Applying module change from the dashboard: %s", modules_req or "(none enabled)"
            )
        if cameras_req is not None:
            logger.info(
                "Applying camera change from the dashboard: %s",
                ", ".join(cam.id for cam in cameras_req),
            )

        if settings_req is not None:
            logger.info("Applying settings change from the dashboard: %s", settings_req)

        new_config = (
            _with_enabled_modules(config, modules_req) if modules_req is not None else config
        )
        if settings_req is not None:
            new_config = with_module_settings(new_config, settings_req)
        new_specs = (
            specs_from_cameras(config, cameras_req) if cameras_req is not None else specs
        )
        hub.configure([(s.id, s.name, s.video.source.value) for s in new_specs])
        try:
            runtimes = build_runtimes(new_config, new_specs, hub)
        except (ModuleError, NotImplementedError, ValueError, VideoSourceError) as exc:
            logger.error("Change rejected, keeping the previous setup: %s", exc)
            reload.fail(str(exc))
            # The previous modules were cleaned up when their pipelines ended,
            # so start fresh ones rather than reuse dead objects.
            hub.configure([(s.id, s.name, s.video.source.value) for s in specs])
            runtimes, code = _build_or_report(config, specs, hub)
            if runtimes is None:
                exit_code = code
                break
            continue

        if modules_req is not None:
            _persist_enabled_modules(config_path, modules_req)
        if settings_req is not None:
            _persist_module_settings(config_path, settings_req)
        if cameras_req is not None:
            try:
                save_camera_file(cameras_file, cameras_req)
            except OSError as exc:
                logger.warning(
                    "Could not save the camera list to %s (it stays in effect for this run): %s",
                    cameras_file,
                    exc,
                )
            origin = "file"
        config, specs = new_config, new_specs

    if dashboard is not None:
        dashboard.stop()

    return _finish(exit_code, specs, runtimes)


def _finish(exit_code: int, specs: List[CameraSpec], runtimes: List[CameraRuntime]) -> int:
    """Report how the last run of each camera went, and pick the exit code."""
    if exit_code != EXIT_OK:
        return exit_code

    failures = [rt for rt in runtimes if rt.failure is not None]
    if len(runtimes) == 1 and failures:
        kind, exc = failures[0].failure
        _fail_message("fatal" if kind == FAILURE_FATAL else "source error",
                      exc if kind != FAILURE_FATAL else f"{type(exc).__name__}: {exc}")
        return EXIT_ERROR if kind == FAILURE_FATAL else EXIT_SOURCE

    processed = 0
    for rt in runtimes:
        stats = rt.stats
        if rt.failure is not None:
            logger.error("Camera '%s' stopped with an error: %s", rt.id, rt.failure[1])
        if stats is None or stats.frames_processed == 0:
            continue
        processed += stats.frames_processed
        logger.info(
            "Done%s: %d frames in %.1f s | avg %.1f FPS | avg processing latency %.1f ms "
            "| avg capture wait %.1f ms",
            f" [{rt.id}]" if len(runtimes) > 1 else "",
            stats.frames_processed,
            stats.capture_sec,
            stats.overall_fps,
            stats.total_latency.avg_ms,
            stats.capture_latency.avg_ms,
        )

    if processed == 0:
        logger.error("No frames were processed - check the camera and configuration")
        return EXIT_SOURCE
    return EXIT_OK


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
