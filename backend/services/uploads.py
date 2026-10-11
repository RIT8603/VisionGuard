"""Video files sent from the browser, kept on the machine running VisionGuard.

A ``file`` camera reads a path on this machine. Uploading puts the file where
such a path can point: ``data/videos/`` next to the project. Only the file
name the browser gives is used, reduced to safe characters, so an upload can
never write anywhere else. Only common video extensions are accepted.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import AsyncIterable, Dict, List, Tuple

VIDEO_DIR = Path(__file__).resolve().parents[2] / "data" / "videos"
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".ts"})
#: Large enough for a long CCTV recording, small enough to stop a runaway upload.
MAX_UPLOAD_BYTES = 4 * 1024**3

_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


class UploadError(ValueError):
    """The upload was refused; the message is safe to show the operator."""


def safe_video_name(name: str | None) -> str:
    """A plain file name for ``name``, or :class:`UploadError`."""
    base = Path(str(name or "").replace("\\", "/")).name
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise UploadError(f"Only video files can be uploaded ({allowed}).")
    stem = _UNSAFE.sub("_", stem).strip(" ._") or "video"
    return f"{stem[:80]}{ext}"


def _unique(directory: Path, name: str) -> Path:
    target = directory / name
    stem, ext = os.path.splitext(name)
    n = 2
    while target.exists():
        target = directory / f"{stem}-{n}{ext}"
        n += 1
    return target


async def save_upload(
    name: str | None,
    chunks: AsyncIterable[bytes],
    *,
    directory: Path = VIDEO_DIR,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Dict[str, object]:
    """Stream ``chunks`` into ``directory`` and describe the saved file.

    Written to a temporary file first and renamed at the end, so a cut-off
    upload never leaves a half file that looks like a real video.
    """
    clean = safe_video_name(name)
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".upload-", suffix=".part", dir=str(directory))
    size = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            async for chunk in chunks:
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise UploadError(
                        f"The file is larger than {max_bytes // 1024**3} GB."
                    )
                handle.write(chunk)
        if size == 0:
            raise UploadError("The file is empty.")
        target = _unique(directory, clean)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return {"name": target.name, "path": str(target), "size": size}


def list_videos(directory: Path = VIDEO_DIR) -> List[Dict[str, object]]:
    """Uploaded videos, newest first."""
    if not directory.is_dir():
        return []
    found: List[Tuple[float, Dict[str, object]]] = []
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
            stat = entry.stat()
            found.append(
                (stat.st_mtime, {"name": entry.name, "path": str(entry), "size": stat.st_size})
            )
    return [item for _, item in sorted(found, key=lambda pair: pair[0], reverse=True)]
