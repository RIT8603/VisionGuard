"""Synthetic video: a face crosses the frame twice, with an empty gap between passes.

Pass 1 is sharp except the middle frames (detector drops out), pass 2 is softly
blurred except a few frames,
so we can check that the *sharp* frame is the one that gets posted.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from findface_lite import Config, DirectorySink, FacePipeline, run_video

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
FACE = ROOT / "tests" / "data" / "face.jpg"
FPS, W, H = 25, 960, 540

pytestmark = pytest.mark.skipif(not (MODEL.exists() and FACE.exists()), reason="model/sample missing")


def make_video(path: Path) -> dict:
    face = cv2.resize(cv2.imread(str(FACE)), (260, 260))
    bg = np.full((H, W, 3), 60, np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    sharp_frames = {1: set(range(0, 75)) - set(range(30, 45)), 2: {50, 51, 52}}
    frame_idx = 0
    expected = {}
    for pass_no in (1, 2):
        for i in range(75):                       # 3 s crossing
            f = bg.copy()
            x = 20 + i * 9
            if i in sharp_frames[pass_no]:
                patch = face
            elif pass_no == 1:                     # heavy blur: detector drops out for a while
                patch = cv2.GaussianBlur(face, (21, 21), 8)
            else:                                  # mild blur: still detected, lower quality
                patch = cv2.GaussianBlur(face, (7, 7), 2.5)
            f[140:400, x:x + 260] = patch
            writer.write(f)
            if i in sharp_frames[pass_no]:
                expected.setdefault(pass_no, []).append(frame_idx)
            frame_idx += 1
        for _ in range(50):                        # 2 s of empty scene
            writer.write(bg)
            frame_idx += 1
    writer.release()
    return expected


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "two_passes.mp4"
    return path, make_video(path)


def run(video_path, out, **overrides):
    cfg = Config(detector_model=str(MODEL), **overrides)
    pipe = FacePipeline(cfg, [DirectorySink(out)])
    events = run_video(pipe, str(video_path))
    return events, pipe


def test_offline_one_best_face_per_track(video, tmp_path):
    path, sharp = video
    events, pipe = run(path, tmp_path, filter_min_quality=0.3)
    assert pipe.stats["tracks"] == 2, pipe.stats
    assert [e["reason"] for e in events] == ["overall_best", "overall_best"]
    assert events[0]["frame_idx"] in sharp[1]
    assert events[1]["frame_idx"] in sharp[2]          # picked one of the 3 sharp frames
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    assert (tmp_path / rec["face_path"]).exists() and (tmp_path / rec["full_frame_path"]).exists()
    # motion gate skipped the static empty frames
    assert pipe.stats["detector_runs"] < pipe.stats["frames"]


def test_realtime_posts_more_than_offline(video, tmp_path):
    path, _ = video
    events, _ = run(path, tmp_path, filter_min_quality=0.3, overall_only=False,
                    realtime_post_every_interval=True, realtime_post_first_immediately=True)
    reasons = [e["reason"] for e in events]
    assert reasons.count("first_immediately") == 2
    assert len(events) > 2


def test_roi_and_history(video, tmp_path):
    path, _ = video
    # ROI on the right half: faces are only posted once they reach it
    events, pipe = run(path, tmp_path, filter_min_quality=0.0, roi=(W // 2, H, W // 2, 0),
                       track_send_history=True)
    assert len(events) == 2
    for e in events:
        cx = (e["bbox"][0] + e["bbox"][2]) / 2
        assert cx >= W // 2
        assert len(e["track"]["history"]) > 10
    assert pipe.stats["rejected"].get("outside_roi", 0) > 0
