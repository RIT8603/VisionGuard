"""Download the two face models VisionGuard needs (one-off).

OpenCV 5 ships no face model data, so face detection and recognition need two
small ONNX files from the OpenCV Zoo:

* ``face_detection_yunet_2023mar.onnx``      (~340 KB)  - finds faces
* ``face_recognition_sface_2021dec.onnx``    (~37 MB)   - turns a face into an embedding

Run from the project root::

    python -m tools.download_face_models

Files that already exist are skipped. A download is written to a temporary file
and moved into place only when complete, so an interrupted run never leaves a
truncated model behind.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
TARGET_DIR = Path("computer_vision/Model/face")

MODELS = (
    (
        "face_detection_yunet_2023mar.onnx",
        f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        100_000,
    ),
    (
        "face_recognition_sface_2021dec.onnx",
        f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        10_000_000,
    ),
)


def download(name: str, url: str, min_bytes: int) -> bool:
    target = TARGET_DIR / name
    if target.is_file() and target.stat().st_size >= min_bytes:
        print(f"  {name}: already present")
        return True

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    print(f"  {name}: downloading from {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except Exception as exc:  # noqa: BLE001 - reported, then non-zero exit
        tmp.unlink(missing_ok=True)
        print(f"  {name}: FAILED - {exc}", file=sys.stderr)
        return False

    if tmp.stat().st_size < min_bytes:
        # A Git LFS pointer or an HTML error page is a few hundred bytes.
        tmp.unlink(missing_ok=True)
        print(f"  {name}: FAILED - the download is too small to be the model", file=sys.stderr)
        return False

    tmp.replace(target)
    print(f"  {name}: saved ({target.stat().st_size / 1_000_000:.1f} MB)")
    return True


def main() -> int:
    print(f"Face models -> {TARGET_DIR.resolve()}")
    results = [download(*model) for model in MODELS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
