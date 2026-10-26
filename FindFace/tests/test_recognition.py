"""Recognition stage: SFace embeddings + watchlist matching on posted best faces."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from findface_lite import Config, DirectorySink, FacePipeline, Recognizer, Watchlist, run_video
from findface_lite.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
DET_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
REC_MODEL = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
FACE = ROOT / "tests" / "data" / "face.jpg"
DEMO = ROOT / "samples" / "demo.mp4"


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_watchlist_threshold_and_best_match():
    wl = Watchlist(threshold=0.5)
    assert wl.match(_unit(np.ones(128))) is None          # empty watchlist
    a, b = np.zeros(128), np.zeros(128)
    a[0], b[1] = 1, 1
    wl.add("alice", _unit(a))
    wl.add("bob", _unit(b))
    m = wl.match(_unit(a + 0.3 * b))
    assert m.name == "alice" and m.similarity == pytest.approx(1 / np.sqrt(1.09), abs=1e-4)
    assert wl.match(_unit(a + b + 2 * np.eye(128)[2])) is None   # cos 0.41 < 0.5


needs_models = pytest.mark.skipif(
    not all(p.exists() for p in (DET_MODEL, REC_MODEL, FACE, DEMO)), reason="models/samples missing")


@pytest.fixture
def watchlist_dir(tmp_path):
    folder = tmp_path / "watchlist"
    folder.mkdir()
    shutil.copy(FACE, folder / "face.jpg")
    cv2.imwrite(str(folder / "nobody.png"), np.full((200, 200, 3), 90, np.uint8))  # no face: skipped
    (folder / "notes.txt").write_text("ignored")
    return folder


@needs_models
def test_demo_video_events_match_enrolled_face(watchlist_dir, tmp_path):
    cfg = Config(detector_model=str(DET_MODEL), recognizer_model=str(REC_MODEL))
    rec = Recognizer.from_config(cfg, watchlist_dir)
    assert rec.watchlist.names == ["face"]
    pipe = FacePipeline(cfg, [DirectorySink(tmp_path / "out")], hooks=[rec])
    events = run_video(pipe, str(DEMO))
    assert len(events) >= 1
    for e in events:
        assert e["match"]["name"] == "face"
        assert e["match"]["similarity"] >= cfg.match_threshold
    logged = [json.loads(l) for l in (tmp_path / "out" / "events.jsonl").read_text().splitlines()]
    assert [e["match"] for e in logged] == [e["match"] for e in events]


@needs_models
def test_threshold_above_similarity_gives_null_match(watchlist_dir, tmp_path):
    cfg = Config(detector_model=str(DET_MODEL), recognizer_model=str(REC_MODEL),
                 match_threshold=0.999, post_embedding=True)
    pipe = FacePipeline(cfg, [], hooks=[Recognizer.from_config(cfg, watchlist_dir)])
    events = run_video(pipe, str(DEMO))
    assert events and all(e["match"] is None for e in events)
    assert len(events[0]["embedding"]) == 128


@needs_models
def test_cli_watchlist_flag(watchlist_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)                   # default model paths are relative to the repo
    out = tmp_path / "cli"
    main([str(DEMO), "--out", str(out), "--watchlist", str(watchlist_dir)])
    events = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
    assert events and all(e["match"]["name"] == "face" for e in events)


@needs_models
def test_no_watchlist_leaves_events_unchanged(tmp_path):
    pipe = FacePipeline(Config(detector_model=str(DET_MODEL)), [])
    events = run_video(pipe, str(DEMO), max_frames=100)
    assert events and all("match" not in e for e in events)
