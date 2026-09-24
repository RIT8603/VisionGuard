"""VisionGuard AI entry point.

    python -m app.main
    python -m app.main --source webcam --device-id 1
    python -m app.main --no-window --max-frames 120

Responsibilities: parse flags, load configuration, wire the components, run the
pipeline, and turn failures into clear messages and exit codes. No vision logic
lives here.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from app import __phase__, __version__
from app.core.config import ConfigError, VisionGuardConfig, load_config
from app.core.frame_processor import FrameProcessor
from app.core.output import (
    BaseOutputSink,
    DisplayUnavailableError,
    NullSink,
    create_output_sink,
)
from app.dashboard.server import DashboardServer, DashboardUnavailableError
from app.core.pipeline import RunLimits, VisionPipeline
from app.modules.base import BaseAIModule, ModuleError
from app.modules.registry import ModuleRegistry
from app.sources.base import VideoSourceError
from app.sources.factory import available_source_types, create_video_source
from app.utils.logger import get_logger, setup_logging

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
    parser.add_argument("--source", choices=["webcam", "file", "rtsp"], help="Override video.source")
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


def start_dashboard(
    config: VisionGuardConfig,
    modules: List[BaseAIModule],
    sink: BaseOutputSink,
) -> tuple["DashboardServer", BaseOutputSink]:
    """Start the dashboard and fan output to it alongside the existing sink."""
    from app.core.output import MultiSink, WebSink
    from app.dashboard.server import DashboardServer
    from app.dashboard.state import DashboardState

    state = DashboardState(max_events=config.dashboard.max_events)
    state.info = {
        "app": config.app.name,
        "version": __version__,
        "source_type": config.video.source.value,
        "source_id": config.video.resolved_source_id(),
        "resolution": (
            f"{config.processing.target_width}px wide"
            if config.processing.target_width
            else "native"
        ),
        "modules": [
            {"name": m.module_name, "type": m.module_type.value, "version": m.version}
            for m in modules
        ],
    }

    web = WebSink(
        state,
        config.processing,
        stream_fps=config.dashboard.stream_fps,
        jpeg_quality=config.dashboard.jpeg_quality,
    )

    server = DashboardServer(state, config.dashboard.host, config.dashboard.port)
    server.start()

    # A NullSink alongside the dashboard is just duplicate logging.
    combined = web if isinstance(sink, NullSink) else MultiSink([sink, web])
    return server, combined


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

    # ---- AI modules (only the enabled ones are imported or constructed) ----
    try:
        modules: List[BaseAIModule] = ModuleRegistry.build_enabled(config)
    except ModuleError as exc:
        print(f"\n[module error] {exc}\n", file=sys.stderr)
        return EXIT_MODULE

    # ---- video source ----
    try:
        source = create_video_source(config.video)
    except NotImplementedError as exc:
        print(f"\n[source not available] {exc}\n", file=sys.stderr)
        return EXIT_SOURCE
    except (ValueError, VideoSourceError) as exc:
        print(f"\n[source error] {exc}\n", file=sys.stderr)
        return EXIT_SOURCE

    # ---- output ----
    module_names = [module.module_name for module in modules]
    sink = create_output_sink(
        config.processing,
        app_name=config.app.name,
        source_description=config.video.resolved_source_id(),
        module_names=module_names,
    )
    try:
        sink.prepare()
    except DisplayUnavailableError as exc:
        logger.warning("%s", exc)
        logger.warning("Falling back to headless output")
        sink = NullSink(log_every=config.processing.log_stats_every)

    dashboard: DashboardServer | None = None
    if config.dashboard.enabled:
        try:
            dashboard, sink = start_dashboard(config, modules, sink)
        except DashboardUnavailableError as exc:
            print(f"\n[dashboard error] {exc}\n", file=sys.stderr)
            return EXIT_CONFIG
        except OSError as exc:
            print(
                f"\n[dashboard error] Could not bind "
                f"{config.dashboard.host}:{config.dashboard.port}: {exc}\n"
                "Another process may already be using that port.\n",
                file=sys.stderr,
            )
            return EXIT_CONFIG

    # ---- run ----
    pipeline = VisionPipeline(
        config=config,
        source=source,
        modules=modules,
        output=sink,
        processor=FrameProcessor(config.processing),
    )
    limits = RunLimits(max_frames=args.max_frames, max_seconds=args.duration)

    if config.processing.show_window and isinstance(sink, NullSink) is False:
        logger.info("Press 'q' or ESC in the preview window to stop")
    if dashboard is not None:
        logger.info("Open %s to watch. Ctrl+C to stop.", dashboard.url)

    try:
        stats = pipeline.run(limits)
    except VideoSourceError as exc:
        print(f"\n[source error] {exc}\n", file=sys.stderr)
        return EXIT_SOURCE
    except Exception as exc:  # noqa: BLE001 - top-level guard
        logger.exception("Unhandled pipeline failure")
        print(f"\n[fatal] {type(exc).__name__}: {exc}\n", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if dashboard is not None:
            dashboard.stop()

    if stats.frames_processed == 0:
        logger.error("No frames were processed - check the camera and configuration")
        return EXIT_SOURCE

    logger.info(
        "Done: %d frames in %.1f s | avg %.1f FPS | avg processing latency %.1f ms "
        "| avg capture wait %.1f ms",
        stats.frames_processed,
        stats.capture_sec,
        stats.overall_fps,
        stats.total_latency.avg_ms,
        stats.capture_latency.avg_ms,
    )
    return EXIT_OK


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
