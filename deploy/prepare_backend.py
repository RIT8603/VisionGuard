"""Prepare/check a separate Linux deployment config; never edit the source config.

First boot deliberately enables person detection/tracking only. Saved camera
selections survive reruns; enable other modules after the first smoke test.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import re
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.cameras import load_camera_file  # noqa: E402
from computer_vision.core.config import load_config  # noqa: E402
from computer_vision.inference.base import verify_model_file  # noqa: E402
from computer_vision.modules.registry import ModuleRegistry  # noqa: E402


def linux_path(value: str, description: str) -> Path:
    if re.match(r"^[A-Za-z]:", value) or "\\" in value:
        raise ValueError(f"{description}: replace the Windows path with a Linux path")
    return Path(value)


def configure_devices(modules: dict, mode: str) -> None:
    for params in modules.values():
        backend = params.get("backend", "ultralytics")
        if backend in {"ultralytics", "yoloface"} and ("backend" in params or "model" in params):
            params["device"] = "cuda:0" if mode == "gpu" else "cpu"
        elif backend == "openvino":
            params["device"] = "cpu"
        # Re-ID/ANPR use the existing OpenVINO adapter, not NVIDIA CUDA.
        if "gallery_path" in params:
            params["device"] = "cpu"
        for extra in params.get("models") or []:
            extra_backend = extra.get("backend", backend)
            if extra_backend in {"ultralytics", "yoloface"}:
                extra["device"] = "cuda:0" if mode == "gpu" else "cpu"
            elif extra_backend == "openvino":
                extra["device"] = "cpu"


def validate_runtime(config_path: Path) -> dict:
    config = load_config(config_path, use_env=False)
    cameras = load_camera_file(config_path.with_name("cameras.yaml"), max_cameras=config.dashboard.max_cameras)
    selected = set(config.enabled_module_names())
    selections = [selected]
    videos = [config.video]
    if cameras is not None:
        selected = set()
        selections = []
        videos = []
        for camera in cameras:
            own_selection = set(config.enabled_module_names() if camera.modules is None else camera.modules)
            selected.update(own_selection)
            selections.append(own_selection)
            videos.append(camera.to_video(config.video))
    for video in videos:
        if video.source.value == "webcam":
            raise ValueError("EC2 has no local webcam: configure a file, RTSP or YouTube source")
        if video.source.value == "file":
            path = linux_path(video.path or "", "Video")
            if not path.is_file():
                raise ValueError("A configured video is missing; upload it and correct its Linux path")
            if not video.loop:
                raise ValueError("Use loop: true for a persistent file-video demo; EOF stops processing")
    for selection in selections:
        for name in sorted(selection):
            cls = ModuleRegistry.resolve(name)
            missing = set(cls.required_modules(config.module_config(name).params)) - selection
            if missing:
                raise ValueError(f"{name}: missing enabled dependencies on a camera: {', '.join(sorted(missing))}")
    for name in sorted(selected):
        params = config.module_config(name).params
        for spec in [params, *(params.get("models") or [])]:
            model = spec.get("model")
            if model:
                path = linux_path(str(model), f"{name} model")
                verify_model_file(str(path), expected_sha256=spec.get("model_sha256"), allow_download=False)
        if name in {"ppe_detection", "fire_detection", "smoke_detection"} and not params.get("model"):
            raise ValueError(f"{name}: provide custom model weights before enabling")
        if params.get("identity_store"):
            if not linux_path(str(params["identity_store"]), "Identity store").is_file():
                raise ValueError(f"{name}: configured identity store is missing")
    for value in (config.dashboard.face_db, config.dashboard.event_db, config.dashboard.snapshot_dir):
        linux_path(str(value), "Persistent data")
    return {"cameras": len(videos), "modules": sorted(selected), "max_streams": config.dashboard.max_streams}


def prepare(mode: str, source: Path, output: Path, video: str | None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "config.yaml"
    existed = target.is_file()
    raw = yaml.safe_load((target if existed else source).read_text(encoding="utf-8-sig"))
    raw = copy.deepcopy(raw)
    if not existed:
        if not video:
            raise ValueError("First preparation requires --video with an uploaded Linux video path")
        for name, params in raw.get("modules", {}).items():
            params["enabled"] = name in {"person_detection", "person_tracking"}
        raw.setdefault("dashboard", {}).update(enabled=True, host="0.0.0.0", port=8080, auth_token=None,
                                               allowed_hosts=[], max_streams=8)
        raw.setdefault("processing", {})["show_window"] = False
    if video:
        path = linux_path(video, "Bootstrap video")
        if not path.is_file():
            raise ValueError("Bootstrap video does not exist")
        raw.setdefault("video", {}).update(source="file", path=path.as_posix(), loop=True, source_id="cam-1")
    configure_devices(raw.get("modules", {}), mode)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output,
                                         suffix=".yaml", delete=False) as handle:
            temporary = Path(handle.name)
            yaml.safe_dump(raw, handle, sort_keys=False)
        # Same directory as the eventual config: validate persisted cameras too.
        result = validate_runtime(temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"config": str(target), "mode": mode, "preserved_existing_selection": existed, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--video")
    parser.add_argument("--source-config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("deploy/runtime"))
    parser.add_argument("--check", action="store_true", help="Validate existing runtime config without writing")
    args = parser.parse_args()
    try:
        result = validate_runtime(args.output_dir / "config.yaml") if args.check else prepare(
            args.mode, args.source_config, args.output_dir, args.video)
    except Exception as exc:
        print(f"Deployment preflight failed: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
