"""Process-per-camera: real worker processes, synthetic video, no models."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from backend.services.hub import PHASE_ENDED, CameraHub
from backend.services.runtime import CameraSpec, build_runtimes, cleanup_runtimes, run_group, uses_processes
from computer_vision.core.config import VideoSourceType, build_config
from computer_vision.core.pipeline import RunLimits


def make_video(path, frames=30, size=(160, 120)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.rectangle(img, (5 + i * 3, 30), (45 + i * 3, 90), (220, 220, 220), -1)
        writer.write(img)
    writer.release()


@pytest.fixture
def setup(tmp_path):
    videos = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.avi"
        make_video(path)
        videos.append(path)
    config = build_config({
        "processing": {"show_window": False, "target_width": None},
        "dashboard": {"enabled": True, "process_per_camera": True, "stream_fps": 0, "live_view": False},
        "modules": {},
    })
    specs = [
        CameraSpec(id=f"cam-{i}", name=f"cam-{i}",
                   video=config.video.model_copy(update={"source": VideoSourceType.FILE, "path": str(p), "loop": False}))
        for i, p in enumerate(videos)
    ]
    hub = CameraHub()
    hub.configure([(s.id, s.name, "file") for s in specs])
    return config, specs, hub


def test_each_camera_runs_in_its_own_process_and_reports_back(setup):
    config, specs, hub = setup
    assert uses_processes(config, specs, hub)

    runtimes = build_runtimes(config, specs, hub)
    pids = {rt.process.pid for rt in runtimes}
    assert len(pids) == 2
    # The viewer-demand value must be lock-free: a worker dying while one of
    # its threads held a lock froze the whole dashboard.
    assert all(not hasattr(rt.demand, "get_lock") for rt in runtimes)

    interrupted = run_group(runtimes, RunLimits(), None, hub)
    assert interrupted is False
    for rt in runtimes:
        assert rt.failure is None
        assert rt.stats is not None and rt.stats.frames_processed == 30
        state = hub.get(rt.id)
        assert state.latest_jpeg() is not None          # frames reached the dashboard
        assert state.snapshot()["stats"]["frames_processed"] == 30
    assert {row["status"] for row in hub.describe()} == {"offline"}  # both videos ended
    assert all(not rt.process.is_alive() for rt in runtimes)


def test_a_broken_camera_fails_alone_while_the_other_keeps_running(setup, tmp_path):
    config, specs, hub = setup
    broken = specs[0].video.model_copy(update={"path": str(tmp_path / "missing.avi")})
    cams = [specs[0].__class__(id="cam-x", name="cam-x", video=broken), specs[1]]
    hub.configure([(s.id, s.name, "file") for s in cams])

    runtimes = build_runtimes(config, cams, hub)
    run_group(runtimes, RunLimits(), None, hub)

    status = {row["id"]: row["status"] for row in hub.describe()}
    assert status["cam-x"] == "failed"
    assert runtimes[1].stats.frames_processed == 30 and runtimes[1].failure is None


def test_stop_request_ends_every_worker(setup, tmp_path):
    config, specs, hub = setup
    looping = [
        s.__class__(id=s.id, name=s.name, video=s.video.model_copy(update={"loop": True})) for s in specs
    ]
    runtimes = build_runtimes(config, looping, hub)

    class Reload:
        def __init__(self):
            self.at = time.monotonic() + 2.0

        def wait_for(self, timeout):
            return time.monotonic() > self.at

    started = time.monotonic()
    run_group(runtimes, RunLimits(), Reload(), hub)
    assert time.monotonic() - started < 30
    assert all(not rt.process.is_alive() for rt in runtimes)
    cleanup_runtimes(runtimes)  # safe after a run


def test_a_dead_worker_does_not_stop_a_change_from_applying(setup):
    """A worker that died mid-run (a refused YouTube stream did this) used to
    make the main process hang forever when it asked the cameras to stop."""
    config, specs, hub = setup
    looping = [
        s.__class__(id=s.id, name=s.name, video=s.video.model_copy(update={"loop": True})) for s in specs
    ]
    runtimes = build_runtimes(config, looping, hub)

    class Reload:
        def __init__(self):
            self.at = time.monotonic() + 3.0
            self.killed = False

        def wait_for(self, timeout):
            if not self.killed and time.monotonic() > self.at - 1.5:
                runtimes[0].process.kill()  # dies without a word
                self.killed = True
            return time.monotonic() > self.at

    started = time.monotonic()
    run_group(runtimes, RunLimits(), Reload(), hub)
    assert time.monotonic() - started < 30
    assert not runtimes[1].process.is_alive()  # the healthy camera was stopped too
