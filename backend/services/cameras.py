"""The stored camera list, and turning an API payload into one.

Where cameras come from, highest priority first (decided in ``backend.main``):

1. ``--source`` / ``--rtsp-url`` / ``--video-path`` / ``--device-id`` on the
   command line - a single camera, exactly as before multi-camera existed.
2. ``configs/cameras.yaml`` - written by the dashboard, or by hand.
3. The ``video:`` section of ``config.yaml`` - a single camera.

The list is kept in its own file, next to ``config.yaml``, because an RTSP URL
usually embeds a password and ``config.yaml`` is meant to be committed. The file
is git-ignored and written owner-only where the OS supports that.

Nothing here logs a URL and nothing here returns one unmasked: every message
that leaves this module has had its credentials stripped.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import yaml
from pydantic import ValidationError

from computer_vision.core.config import CameraConfig, ConfigError, VideoSourceType
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

CAMERA_FILE_NAME = "cameras.yaml"
MASK = "***"

_SENSITIVE_QUERY_KEYS = frozenset({"password", "pass", "pwd", "token", "key", "secret"})
_AUTO_ID = re.compile(r"^cam-(\d+)$")


def cameras_file_for(config_path: Path) -> Path:
    """``cameras.yaml`` next to the config file in use."""
    return Path(config_path).with_name(CAMERA_FILE_NAME)


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #


def mask_url(url: str | None) -> str | None:
    """``url`` with any password (and secret-looking query value) replaced by ``***``."""
    if not url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return MASK
    netloc = parts.netloc
    if parts.password is not None:
        user = parts.username or ""
        host = netloc.rsplit("@", 1)[1]
        netloc = f"{user}:{MASK}@{host}"
    query = parts.query
    if query:
        pairs = [
            (k, MASK if k.lower() in _SENSITIVE_QUERY_KEYS else v)
            for k, v in parse_qsl(query, keep_blank_values=True)
        ]
        query = "&".join(f"{k}={v}" for k, v in pairs)
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def public_view(camera: CameraConfig) -> Dict[str, Any]:
    """What the API may say about a camera: everything except secrets."""
    return {
        "id": camera.id,
        "name": camera.label,
        "source": camera.source.value,
        "device_id": camera.device_id,
        "rtsp_url": mask_url(camera.rtsp_url),
        "youtube_url": camera.youtube_url,
        "path": camera.path,
        "loop": camera.loop,
        # None = follows the shared selection; a list = this camera's own.
        "modules": None if camera.modules is None else list(camera.modules),
    }


def describe_validation_error(exc: ValidationError) -> str:
    """Field-by-field message that never echoes the rejected value.

    pydantic's default text quotes the input, which for ``rtsp_url`` would put
    the password into an HTTP response and the logs.
    """
    parts = []
    for error in exc.errors(include_input=False, include_url=False):
        where = ".".join(str(p) for p in error.get("loc", ()))
        message = str(error.get("msg", "invalid")).removeprefix("Value error, ")
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts) or "invalid camera"


# --------------------------------------------------------------------------- #
# File
# --------------------------------------------------------------------------- #


def load_camera_file(path: Path, *, max_cameras: int = 32) -> List[CameraConfig] | None:
    """The cameras stored at ``path``, or None when there is no such file.

    Raises:
        ConfigError: the file exists but cannot be used. A broken file must not
            silently fall back to another camera - the operator would be
            watching the wrong place without knowing it.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    items = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise ConfigError(f"{path} must contain a non-empty 'cameras:' list")

    cameras, errors = parse_cameras(items, max_cameras=max_cameras)
    if errors:
        raise ConfigError(f"{path}: " + "; ".join(errors))
    return cameras


def save_camera_file(path: Path, cameras: Sequence[CameraConfig]) -> None:
    """Write ``cameras`` to ``path`` atomically, owner-only where possible."""
    path = Path(path)
    entries: List[Dict[str, Any]] = []
    for cam in cameras:
        entry: Dict[str, Any] = {"id": cam.id}
        if cam.name:
            entry["name"] = cam.name
        entry["source"] = cam.source.value
        if cam.source is VideoSourceType.RTSP:
            entry["rtsp_url"] = cam.rtsp_url
        elif cam.source is VideoSourceType.FILE:
            entry["path"] = cam.path
            if cam.loop:
                entry["loop"] = True
        elif cam.source is VideoSourceType.YOUTUBE:
            entry["youtube_url"] = cam.youtube_url
        else:
            entry["device_id"] = cam.device_id
        if cam.modules is not None:
            entry["modules"] = list(cam.modules)
        entries.append(entry)

    header = (
        "# Camera list for VisionGuard. Written by the dashboard; safe to edit by hand.\n"
        "# RTSP URLs may contain passwords - keep this file out of version control.\n"
    )
    body = yaml.safe_dump({"cameras": entries}, sort_keys=False, allow_unicode=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".cameras-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(header + body)
        try:
            os.chmod(tmp_name, 0o600)  # best effort; Windows only honours the read-only bit
        except OSError:
            pass
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# Payload -> cameras
# --------------------------------------------------------------------------- #


def _restore_masked_url(new_url: str | None, old: CameraConfig | None) -> Tuple[str | None, str | None]:
    """Put a hidden password back, or say why that is impossible.

    The API only ever shows ``rtsp://user:***@host/...``. A form that sends that
    back unchanged means "keep the stored password". Returns ``(url, error)``.
    """
    if not new_url or MASK not in new_url:
        return new_url, None
    if old is None or not old.rtsp_url:
        return None, "the password is hidden (***); type the full URL again"
    if mask_url(old.rtsp_url) == new_url:
        return old.rtsp_url, None
    try:
        old_parts, new_parts = urlsplit(old.rtsp_url), urlsplit(new_url)
    except ValueError:
        return None, "the URL could not be read; type it again"
    if (
        old_parts.password
        and new_parts.password == MASK
        and (old_parts.username or "") == (new_parts.username or "")
    ):
        restored = new_url.replace(f":{MASK}@", f":{old_parts.password}@", 1)
        if MASK not in restored:
            return restored, None
    return None, "the password is hidden (***); type the full URL again"


def parse_cameras(
    items: Iterable[Mapping[str, Any]],
    *,
    max_cameras: int,
    existing: Sequence[CameraConfig] = (),
    known_modules: Iterable[str] | None = None,
) -> Tuple[List[CameraConfig], List[str]]:
    """Validate a list of camera dicts. Returns ``(cameras, errors)``.

    ``existing`` is the running list, used to restore a masked password and to
    keep ids stable. Entries without an ``id`` get ``cam-N``. All problems are
    reported at once, each prefixed with the camera's position.

    ``known_modules`` (when given) is the catalogue a camera's own ``modules``
    list is checked against, so a misspelled name is refused here rather than
    when the pipeline is rebuilt.
    """
    items = list(items)
    known = None if known_modules is None else set(known_modules)
    errors: List[str] = []
    if not items:
        return [], ["at least one camera is required"]
    if len(items) > max_cameras:
        return [], [f"at most {max_cameras} cameras are allowed (got {len(items)})"]

    old_by_id = {cam.id: cam for cam in existing}
    taken = {str(i.get("id", "")).strip().lower() for i in items if isinstance(i, Mapping)}
    next_auto = 1

    def auto_id() -> str:
        nonlocal next_auto
        while True:
            candidate = f"cam-{next_auto}"
            next_auto += 1
            if candidate not in taken:
                taken.add(candidate)
                return candidate

    cameras: List[CameraConfig] = []
    seen_ids: set[str] = set()
    seen_devices: set[int] = set()
    for position, raw in enumerate(items, start=1):
        label = f"camera {position}"
        if not isinstance(raw, Mapping):
            errors.append(f"{label}: must be an object")
            continue
        data = dict(raw)
        if not str(data.get("id", "")).strip():
            data["id"] = auto_id()

        old = old_by_id.get(str(data["id"]).strip().lower())
        if data.get("rtsp_url"):
            url, problem = _restore_masked_url(str(data["rtsp_url"]), old)
            if problem:
                errors.append(f"{label}: {problem}")
                continue
            data["rtsp_url"] = url

        try:
            camera = CameraConfig(**data)
        except ValidationError as exc:
            errors.append(f"{label}: {describe_validation_error(exc)}")
            continue
        except TypeError:
            errors.append(f"{label}: unreadable fields")
            continue

        if known is not None and camera.modules:
            unknown = sorted(set(camera.modules) - known)
            if unknown:
                errors.append(f"{label}: unknown module(s): {', '.join(unknown)}")
                continue

        if camera.id in seen_ids:
            errors.append(f"{label}: duplicate id '{camera.id}'")
            continue
        seen_ids.add(camera.id)
        if camera.source is VideoSourceType.WEBCAM:
            if camera.device_id in seen_devices:
                errors.append(f"{label}: webcam device {camera.device_id} is used twice")
                continue
            seen_devices.add(camera.device_id)
        cameras.append(camera)

    return ([] if errors else cameras), errors
