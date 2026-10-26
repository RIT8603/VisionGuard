import numpy as np

from findface_lite.config import Config, parse_geometry
from findface_lite.detector import Detection
from findface_lite.snapshot import Candidate, SnapshotSelector
from findface_lite.tracker import Tracker, iou


def det(x, y, s=50):
    return Detection(np.array([x, y, x + s, y + s], dtype=np.float32), 0.9, np.zeros((5, 2), np.float32))


def test_parse_geometry():
    assert parse_geometry("640x360+10+20") == (640, 360, 10, 20)
    assert Config.from_dict({"roi": "100x50+1+2", "unknown": 3}).roi == (100, 50, 1, 2)


def test_iou():
    a = np.array([0, 0, 10, 10.0])
    assert iou(a, a) == 1.0
    assert iou(a, np.array([20, 20, 30, 30.0])) == 0.0


def test_track_continues_and_closes_after_miss_interval():
    tr = Tracker(overlap_threshold=0.25, miss_interval=1.0)
    for i in range(10):
        matched, closed = tr.update([det(100 + 5 * i, 100)], i, i / 10)
        assert not closed
    assert len(tr.active) == 1 and next(iter(tr.active.values())).hits == 10
    _, closed = tr.update([], 30, 3.0)          # > 1 s without detection
    assert len(closed) == 1 and closed[0].close_reason == "miss_interval"


def test_two_faces_get_two_tracks():
    tr = Tracker()
    tr.update([det(0, 0), det(300, 300)], 0, 0.0)
    tr.update([det(4, 0), det(304, 300)], 1, 0.04)
    assert sorted(t.hits for t in tr.active.values()) == [2, 2]


def test_interpolation_fills_gaps():
    tr = Tracker(interpolate=True)
    tr.update([det(0, 0)], 0, 0.0)
    tr.update([det(10, 0)], 4, 0.16)
    hist = next(iter(tr.active.values())).history
    assert [h[0] for h in hist] == [0, 1, 2, 3, 4]
    assert [h[2] for h in hist] == [True, False, False, False, True]
    assert abs(hist[2][1][0] - 5.0) < 1e-4


def test_max_duration_force_closes():
    tr = Tracker(max_duration_frames=5)
    closed_all = []
    for i in range(8):
        _, closed = tr.update([det(100, 100)], i, i / 25)
        closed_all += closed
    assert closed_all and closed_all[0].close_reason == "max_duration"


def test_offline_posts_single_best_on_close():
    sel = SnapshotSelector(overall_only=True)
    for i, q in enumerate([0.5, 0.9, 0.7]):
        assert sel.offer(1, Candidate(q, i, i * 0.1)) == []
    posts, _ = sel.close(1)
    assert len(posts) == 1 and posts[0].candidate.quality == 0.9


def test_realtime_increasing_quality_only():
    sel = SnapshotSelector(overall_only=False, interval=1.0)
    posts = []
    for t, q in [(0.0, 0.5), (0.5, 0.6), (1.1, 0.4), (1.5, 0.45), (2.2, 0.8)]:
        posts += sel.offer(1, Candidate(q, int(t * 10), t))
    posts += sel.close(1)[0]
    # window0 best 0.6 posted; window1 best 0.45 < 0.6 skipped; window2 0.8 posted at close
    assert [p.candidate.quality for p in posts] == [0.6, 0.8]


def test_realtime_every_interval_and_first_immediately():
    sel = SnapshotSelector(overall_only=False, interval=1.0, every_interval=True, first_immediately=True)
    posts = []
    for t, q in [(0.0, 0.5), (0.5, 0.6), (1.1, 0.4), (1.5, 0.45)]:
        posts += sel.offer(1, Candidate(q, int(t * 10), t))
    posts += sel.tick(2.05)
    assert [(p.reason, p.candidate.quality) for p in posts] == [
        ("first_immediately", 0.5), ("interval_best", 0.6), ("interval_best", 0.45)]
