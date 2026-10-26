"""CLI:  python -m findface_lite <video file | camera index | rtsp url> [options]"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields

from .config import Config
from .pipeline import FacePipeline, run_video
from .recognition import Recognizer
from .sinks import DirectorySink, WebhookSink


def _coerce(cfg: Config, key: str, raw: str):
    current = getattr(cfg, key)
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="findface_lite", description="Face detection + tracking + best-snapshot posting + watchlist matching")
    ap.add_argument("source", help="video file, camera index (0) or stream URL")
    ap.add_argument("--config", help="JSON file with Config fields")
    ap.add_argument("--mode", choices=["offline", "realtime"], help="offline = one best face per track")
    ap.add_argument("--out", default="output", help="directory for events.jsonl and face crops")
    ap.add_argument("--webhook", help="POST every event to this URL")
    ap.add_argument("--watchlist", metavar="DIR",
                    help="folder of enrolled photos (name = file name); adds a match to each event")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--loop", action="store_true", help="loop video playback continuously")
    ap.add_argument("--preview", help="write annotated video to this .mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any Config field, e.g. --set roi=640x360+0+0")
    args = ap.parse_args(argv)

    data = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            data = json.load(fh)
    cfg = Config.from_dict(data)
    if args.mode:
        cfg.overall_only = args.mode == "offline"
    names = {f.name for f in fields(Config)}
    for item in args.set:
        key, _, raw = item.partition("=")
        if key not in names:
            ap.error(f"unknown config key: {key}")
        if key in ("roi", "rot"):
            cfg = Config.from_dict({**{f.name: getattr(cfg, f.name) for f in fields(Config)}, key: raw})
        else:
            setattr(cfg, key, _coerce(cfg, key, raw))

    sinks = [DirectorySink(args.out, cfg.jpeg_quality)]
    if args.webhook:
        sinks.append(WebhookSink(args.webhook, cfg.jpeg_quality))
    hooks = []
    if args.watchlist:
        rec = Recognizer.from_config(cfg, args.watchlist)
        print(f"watchlist: {len(rec.watchlist)} enrolled ({', '.join(sorted(set(rec.watchlist.names)))})")
        hooks.append(rec)
    pipe = FacePipeline(cfg, sinks, source_name=str(args.source), hooks=hooks)
    events = run_video(pipe, args.source, show=args.show, max_frames=args.max_frames,
                       write_preview=args.preview, loop=args.loop)
    s = pipe.stats
    print(f"frames={s['frames']} detector_runs={s['detector_runs']} detections={s['detections']} "
          f"tracks={s['tracks']} events={len(events)} rejected={s['rejected']}")
    print(f"events written to {args.out}/events.jsonl")


if __name__ == "__main__":
    main()
