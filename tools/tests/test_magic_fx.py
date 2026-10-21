"""The cosmetic 'magic' effect that follows tracked people."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from computer_vision.core.context import BBox
from computer_vision.utils import magic_fx
from computer_vision.utils.magic_fx import MAX_SPARKLES, MagicTrails


def person(tid, x, label="person"):
    return SimpleNamespace(
        track_id=tid, label=label, bbox=BBox(x, 100, x + 80, 300), source_module="person_tracking"
    )


def blank():
    return np.zeros((400, 600, 3), np.uint8)


def test_effect_lights_up_around_a_walking_person():
    fx = MagicTrails(seed=1)
    frame = blank()
    for i in range(12):
        frame = blank()
        fx.draw(frame, [person(1, 50 + 15 * i)], now=i * 0.1)
    assert frame.sum() > 0
    # The trail is behind the person: pixels lit to the left of where they are now.
    assert frame[:, :180].sum() > 0


def test_a_person_standing_still_gets_no_effect():
    fx = MagicTrails(seed=1)
    frame = blank()
    for i in range(12):
        frame = blank()
        fx.draw(frame, [person(1, 200)], now=i * 0.1)
    assert frame.sum() == 0
    assert fx._sparkles == []


def test_nothing_is_drawn_on_the_person_or_kept_clear_boxes():
    fx = MagicTrails(seed=1)
    face = SimpleNamespace(label="face", bbox=BBox(60, 120, 100, 160))
    for i in range(12):
        frame = blank()
        walker = person(1, 50 + 15 * i)
        fx.draw(frame, [walker], now=i * 0.1, keep_clear=[face])
    box = walker.bbox
    assert frame[int(box.y1):int(box.y2), int(box.x1):int(box.x2)].sum() == 0
    assert frame[120:160, 60:100].sum() == 0
    assert frame.sum() > 0  # the trail behind is still there


def test_other_tracks_are_ignored():
    fx = MagicTrails(seed=1)
    frame = blank()
    car = SimpleNamespace(track_id=1, label="car", bbox=BBox(50, 100, 130, 300), source_module="vehicle_tracking")
    fx.draw(frame, [car], now=0.0)
    assert frame.sum() == 0


def test_a_person_who_leaves_is_forgotten():
    fx = MagicTrails(seed=1)
    fx.draw(blank(), [person(7, 100)], now=0.0)
    fx.draw(blank(), [], now=5.0)
    assert fx._paths == {} and fx._sparkles == []


def test_sparkles_are_capped_in_a_crowd():
    fx = MagicTrails(seed=1)
    crowd = [person(i, (i * 37) % 500) for i in range(60)]
    for step in range(40):
        moved = [person(p.track_id, p.bbox.x1 + step * 9) for p in crowd]
        fx.draw(blank(), moved, now=step * 0.2)
    assert len(fx._sparkles) <= MAX_SPARKLES


def test_draws_in_place_and_keeps_the_frame_shape():
    fx = MagicTrails(seed=1)
    frame = blank()
    out = fx.draw(frame, [person(1, 100)], now=0.0)
    assert out is frame and out.shape == (400, 600, 3)


fastapi = pytest.importorskip("fastapi", reason="dashboard extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.server import build_app  # noqa: E402
from backend.services.state import DashboardState  # noqa: E402


def test_toggle_over_http():
    magic_fx.set_magic_enabled(False)
    client = TestClient(build_app(DashboardState(max_events=5)), base_url="http://127.0.0.1")
    assert client.get("/api/view").json() == {"magic_effect": False}
    assert client.post("/api/view", json={"magic_effect": True}).json() == {"magic_effect": True}
    assert magic_fx.magic_enabled() is True
    assert client.post("/api/view", json={"magic_effect": "yes"}).status_code == 400
    magic_fx.set_magic_enabled(False)
