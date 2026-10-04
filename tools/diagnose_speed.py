"""Find out how fast detection can run on *this* machine, and what is slowing it.

The dashboard shows how long a frame took, but not why. A slow frame can be a
slow model, a busy CPU, a machine that has run out of memory, or several
cameras waiting for the one model. This separates them: it times the detector
alone, with no cameras, no video decoding and no browser.

Usage (from the repository root, with the backend stopped for a clean number,
or running to see what it costs)::

    python -m tools.diagnose_speed
    python -m tools.diagnose_speed --video rtsp-test/video/Walking_area.mp4
    python -m tools.diagnose_speed --cameras 2 --frames 50

What it prints:

* the machine - cores, memory free, how busy the CPU already is;
* which OpenVINO devices exist (CPU, and GPU if the Intel graphics driver
  provides one);
* for every detection model in ``computer_vision/Model/detection`` on each
  device: time per frame, and the frame rate that leaves for each camera;
* a plain reading of those numbers.

How to read it. YOLO11s 8-bit at 544x960 measures about 50 ms per frame on a
2-core server CPU. If this machine reports several times that with the backend
stopped, the model is not the problem - the machine is busy or short of
memory, and a smaller model would only hide it. If the number here is good
and the dashboard's is not, the difference is everything else the running
system does: decoding the camera streams, the other cameras, the browser.

Exit codes: 0 success, 1 nothing to measure (OpenVINO or the models missing).
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

MODEL_ROOT = Path("computer_vision/Model/detection")
FRAME_SIZE = (960, 540)  # width, height: what target_width 960 gives a 16:9 camera

#: Per-frame time on an idle 2-core server CPU, for the reading at the end.
_REFERENCE_MS = {(544, 960): 50.0, (384, 640): 25.0}


def machine_report() -> Tuple[Optional[float], Optional[float]]:
    """Print the machine; return (available memory GB, CPU busy percent)."""
    print("Machine")
    print(f"  Python      {platform.python_version()}  ({sys.executable})")
    print(f"  System      {platform.system()} {platform.release()}  {platform.machine()}")
    print(f"  Processor   {platform.processor() or 'unknown'}")
    print(f"  Cores       {os.cpu_count()} logical")

    available_gb: Optional[float] = None
    busy: Optional[float] = None
    try:
        import psutil

        memory = psutil.virtual_memory()
        available_gb = memory.available / 2**30
        print(
            f"  Memory      {memory.total / 2**30:.1f} GB total, "
            f"{available_gb:.1f} GB free ({memory.percent:.0f}% used)"
        )
        busy = psutil.cpu_percent(interval=1.0)
        print(f"  CPU busy    {busy:.0f}% right now, before this test starts")
        # The first reading only starts each process's counter; the second,
        # half a second later, is the usage in between.
        processes = list(psutil.process_iter(["name", "cpu_percent"]))
        time.sleep(0.5)
        usage = []
        for process in processes:
            try:
                usage.append((process.cpu_percent(interval=None), process.info["name"]))
            except Exception:  # noqa: BLE001 - a process may have ended
                continue
        top = [u for u in sorted(usage, reverse=True)[:5] if u[0] >= 5.0]
        if top:
            # Per-core percent: 100% is one whole core.
            print("  Busiest     " + ", ".join(f"{name} {cpu:.0f}%" for cpu, name in top))
    except ImportError:
        print("  Memory/CPU  (pip install psutil to see memory and CPU load)")
    print()
    return available_gb, busy


def load_frame(video: Optional[str]) -> np.ndarray:
    """One frame to run on: from ``video`` if given, else a synthetic one."""
    import cv2

    if video:
        capture = cv2.VideoCapture(video)
        frame = None
        for _ in range(30):  # a little way in, past any black leader
            ok, candidate = capture.read()
            if not ok:
                break
            frame = candidate
        capture.release()
        if frame is not None:
            return cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        print(f"  (could not read {video}; using a synthetic frame)\n")

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    return cv2.GaussianBlur(frame, (0, 0), 3)


def find_models() -> List[Path]:
    if not MODEL_ROOT.is_dir():
        return []
    return sorted(
        path for path in MODEL_ROOT.iterdir() if path.is_dir() and any(path.glob("*.xml"))
    )


def time_model(model: Path, device: str, frame: np.ndarray, frames: int):
    """Median ms per frame for one model on one device, or an error string."""
    from computer_vision.inference.openvino_backend import OpenVinoDetector

    detector = OpenVinoDetector({"model": str(model), "device": device})
    started = time.perf_counter()
    try:
        detector.load()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    load_seconds = time.perf_counter() - started

    if detector.description.rsplit("@", 1)[-1] != device:
        detector.unload()
        return None, "device not available"

    times = []
    count = 0
    for _ in range(frames):
        begin = time.perf_counter()
        count = len(detector.infer(frame))
        times.append((time.perf_counter() - begin) * 1000.0)
    size = (detector._height, detector._width)  # noqa: SLF001 - a diagnostic
    detector.unload()
    return (statistics.median(times), max(times), load_seconds, count, size), None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", help="a video file to take the test frame from")
    parser.add_argument("--frames", type=int, default=30, help="inferences per test")
    parser.add_argument("--cameras", type=int, default=1, help="cameras you plan to run")
    args = parser.parse_args(argv)
    cameras = max(1, args.cameras)

    available_gb, busy = machine_report()

    try:
        import openvino as ov
    except ImportError:
        print("OpenVINO is not installed in this Python:  pip install openvino")
        return 1

    core = ov.Core()
    devices = [name for name in core.available_devices if name.split(".")[0] in ("CPU", "GPU")]
    print(f"OpenVINO {ov.__version__.split('-')[0]}")
    for name in devices:
        try:
            full = core.get_property(name, "FULL_DEVICE_NAME")
        except Exception:  # noqa: BLE001
            full = ""
        print(f"  {name:<6} {full}")
    if not any(name.startswith("GPU") for name in devices):
        print("  (no GPU device: Intel graphics driver missing or not supported)")
    print()

    models = find_models()
    if not models:
        print(f"No OpenVINO models found in {MODEL_ROOT}. Run this from the repository root.")
        return 1

    frame = load_frame(args.video)
    print(f"Detector alone, {args.frames} frames each, frame {FRAME_SIZE[0]}x{FRAME_SIZE[1]}")
    print(f"  {'model':<40} {'device':<6} {'ms/frame':>9} {'worst':>7} {'fps':>6}"
          + (f" {'fps/camera':>11}" if cameras > 1 else ""))

    results = []
    for model in models:
        for device in ("CPU", "GPU"):
            if not any(name.split(".")[0] == device for name in devices):
                continue
            outcome, error = time_model(model, device, frame, args.frames)
            if outcome is None:
                print(f"  {model.name:<40} {device:<6} failed: {error}")
                continue
            median, worst, load_seconds, _, size = outcome
            fps = 1000.0 / median
            line = f"  {model.name:<40} {device:<6} {median:>9.0f} {worst:>7.0f} {fps:>6.1f}"
            if cameras > 1:
                line += f" {fps / cameras:>11.1f}"
            if device == "GPU" and load_seconds > 5:
                line += f"   (first load {load_seconds:.0f} s, cached afterwards)"
            print(line)
            results.append((model, device, median, size))
    print()

    if not results:
        return 1
    reading(results, available_gb, busy, cameras)
    return 0


def reading(results, available_gb, busy, cameras: int) -> None:
    print("Reading")
    cpu = [r for r in results if r[1] == "CPU"]
    for model, _, median, size in cpu:
        reference = _REFERENCE_MS.get(size)
        if reference and median > 3 * reference:
            print(
                f"  - {model.name} takes {median:.0f} ms on the CPU; an idle 2-core "
                f"server does it in about {reference:.0f} ms. The model is not what "
                "is slow - something else is using this machine."
            )
            break
    if busy is not None and busy >= 50:
        print(
            f"  - The CPU was already {busy:.0f}% busy before the test. Close what "
            "is using it (see 'Busiest' above); a video encoder re-encoding the "
            "test streams is a common one - publish them with '-c copy' instead."
        )
    if available_gb is not None and available_gb < 1.5:
        print(
            f"  - Only {available_gb:.1f} GB of memory is free. A machine that is "
            "swapping makes every frame slow and uneven; close other programs."
        )

    best = min(results, key=lambda r: r[2])
    same_size = [r for r in results if r[3] == (544, 960)]
    if same_size:
        best_full = min(same_size, key=lambda r: r[2])
        device = "intel:gpu" if best_full[1] == "GPU" else "intel:cpu"
        print(
            f"  - Full accuracy (544x960): {best_full[0].name} on {best_full[1]} at "
            f"{best_full[2]:.0f} ms = {1000 / best_full[2] / cameras:.1f} fps per camera"
            f" with {cameras} camera(s). Set 'device: {device}' on every detector."
        )
        if best is not best_full and best[2] < 0.7 * best_full[2]:
            print(
                f"  - {best[0].name} on {best[1]} is faster ({best[2]:.0f} ms) but "
                "misses more distant people; see the comment in configs/config.yaml."
            )
    print(
        "  - These are the detector alone. The running system also decodes every "
        "camera stream; a 1080p stream costs far more than a 720p or sub-stream."
    )


if __name__ == "__main__":
    sys.exit(main())
