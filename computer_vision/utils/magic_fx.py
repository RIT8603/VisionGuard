"""A "magic" visual effect that traces how tracked people move.

Purely cosmetic - it reads tracks and draws, and never changes what was
detected. The effect shows **movement only**:

* **Motion trail** - a glowing ribbon along the path a person walked over the
  last ~1.5 s, bright where they just were and fading behind them.
* **Sparkles** - small stars that fall off the trail while the person is
  moving (more the faster they go), drift upwards and twinkle out.

A person standing still leaves no trail and sheds nothing, so the effect mimics
their motion rather than decorating them. And nothing is drawn *on* anyone: the
current box of every tracked person, and every other detection passed in
``keep_clear`` (faces, for instance), is masked out, so people and their face
and person boxes look exactly as they do without the effect. The trail appears
to stream out from behind each person.

Everything is driven by wall-clock time, not frame count, so the trail length
and the particle motion look the same at 4 fps or 30 fps. The glow comes from
drawing on a black layer at half resolution, blurring it and adding it to the
frame - cheap enough for a live stream.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from computer_vision.utils.annotate import color_for_track_id

#: Seconds of path kept in a trail.
TRAIL_SECONDS = 1.5
#: Seconds a sparkle lives.
SPARKLE_LIFE = 1.1
#: Sparkles shed per second by a person walking at ``REFERENCE_SPEED`` px/s.
SPARKLE_RATE = 24.0
REFERENCE_SPEED = 120.0
#: Slower than this (px/s) counts as standing still: no sparkles.
MIN_SPEED = 15.0
#: A track not seen for this long has its trail dropped.
FORGET_AFTER = 2.0
#: Hard cap, so a crowd cannot turn the effect into a slideshow.
MAX_SPARKLES = 900

Key = Tuple[str, int]

#: Process-wide switch: every video output checks it on each frame, so it can
#: be flipped from the dashboard while running. Seeded from
#: ``processing.magic_effect`` at start-up.
_enabled = False


def magic_enabled() -> bool:
    return _enabled


def set_magic_enabled(value: bool) -> None:
    global _enabled
    _enabled = bool(value)


@dataclass
class _Sparkle:
    x: float
    y: float
    vx: float
    vy: float
    born: float
    size: float
    color: Tuple[int, int, int]


def _brighten(color: Tuple[int, int, int], amount: float = 0.45) -> Tuple[int, int, int]:
    """Mix towards white, so glows read as light rather than paint."""
    return tuple(int(c + (255 - c) * amount) for c in color)  # type: ignore[return-value]


def _is_person(track: Any) -> bool:
    return str(getattr(track, "label", "")).lower() == "person" or (
        getattr(track, "source_module", "") == "person_tracking"
    )


def _box_of(item: Any) -> Optional[Tuple[int, int, int, int]]:
    box = getattr(item, "bbox", None)
    if box is None:
        return None
    return int(box.x1), int(box.y1), int(math.ceil(box.x2)), int(math.ceil(box.y2))


class MagicTrails:
    """Stateful renderer: remembers where each person has been."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._paths: Dict[Key, Deque[Tuple[float, float, float]]] = {}
        self._last_seen: Dict[Key, float] = {}
        self._sparkles: List[_Sparkle] = []
        self._last_draw: Optional[float] = None
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #

    def draw(
        self,
        frame: np.ndarray,
        tracks: Iterable[Any],
        now: Optional[float] = None,
        keep_clear: Iterable[Any] = (),
    ) -> np.ndarray:
        """Draw the movement effect for ``tracks`` onto ``frame`` (in place).

        ``keep_clear`` are further detections (faces, ...) whose boxes must stay
        untouched; every person track's own box is always kept clear.
        """
        now = time.perf_counter() if now is None else now
        tracks = list(tracks)
        people = [t for t in tracks if _is_person(t)]
        clear = [b for b in (_box_of(x) for x in [*tracks, *keep_clear]) if b is not None]
        with self._lock:
            dt = 0.0 if self._last_draw is None else min(0.25, max(0.0, now - self._last_draw))
            self._last_draw = now
            self._update(people, now, dt)
            self._render(frame, now, clear)
        return frame

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    def _update(self, people: List[Any], now: float, dt: float) -> None:
        for track in people:
            key = (getattr(track, "source_module", ""), int(track.track_id))
            box = track.bbox
            # The path is traced at the feet, where a walking person's motion
            # reads most naturally; it is hidden inside their own box anyway.
            point = ((box.x1 + box.x2) / 2.0, box.y1 + 0.9 * (box.y2 - box.y1))
            path = self._paths.setdefault(key, deque())
            speed = 0.0
            if path:
                pt, px, py = path[-1]
                if now - pt > 1e-3:
                    speed = math.hypot(point[0] - px, point[1] - py) / (now - pt)
            path.append((now, point[0], point[1]))
            while path and now - path[0][0] > TRAIL_SECONDS:
                path.popleft()
            self._last_seen[key] = now
            self._shed(key, path, speed, dt, max(8.0, box.x2 - box.x1))

        for key in [k for k, seen in self._last_seen.items() if now - seen > FORGET_AFTER]:
            self._last_seen.pop(key, None)
            self._paths.pop(key, None)

        # Move and age the sparkles: they float up and slow down.
        alive: List[_Sparkle] = []
        for s in self._sparkles:
            if now - s.born >= SPARKLE_LIFE:
                continue
            s.x += s.vx * dt
            s.y += s.vy * dt
            s.vx *= 0.96
            s.vy = s.vy * 0.96 - 18.0 * dt
            alive.append(s)
        self._sparkles = alive[-MAX_SPARKLES:]

    def _shed(self, key: Key, path: Deque, speed: float, dt: float, width: float) -> None:
        """Sparkles fall off the trail behind a moving person - never while still."""
        if dt <= 0 or speed < MIN_SPEED or len(path) < 2:
            return
        expected = SPARKLE_RATE * min(2.0, speed / REFERENCE_SPEED) * dt
        count = int(expected) + (1 if self._rng.random() < expected - int(expected) else 0)
        color = _brighten(color_for_track_id(key[1], key[0]))
        points = list(path)
        for _ in range(count):
            # Somewhere along the older part of the trail, not at the person.
            t, x, y = points[self._rng.randrange(0, max(1, len(points) - 1))]
            self._sparkles.append(
                _Sparkle(
                    x=x + self._rng.uniform(-0.15, 0.15) * width,
                    y=y + self._rng.uniform(-0.1, 0.05) * width,
                    vx=self._rng.uniform(-20.0, 20.0),
                    vy=self._rng.uniform(-60.0, -15.0),
                    born=self._last_draw or 0.0,
                    size=self._rng.uniform(1.2, 2.6) * max(0.7, width / 120.0),
                    color=color,
                )
            )

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def _render(self, frame: np.ndarray, now: float, clear: List[Tuple[int, int, int, int]]) -> None:
        height, width = frame.shape[:2]
        scale = 0.5
        glow = np.zeros((max(1, int(height * scale)), max(1, int(width * scale)), 3), np.uint8)

        trails = []
        for key, path in self._paths.items():
            # Only the segments where the person actually moved: someone
            # standing still has no trail at all, not a glowing dot.
            points: List[Tuple[float, float, float]] = []
            for p in path:
                if not points or math.hypot(p[1] - points[-1][1], p[2] - points[-1][2]) >= 1.0:
                    points.append(p)
                else:
                    points[-1] = (p[0], points[-1][1], points[-1][2])  # still here: refresh its age
            if len(points) >= 2:
                trails.append((_brighten(color_for_track_id(key[1], key[0])), points))

        for color, points in trails:
            for (_, x0, y0), (t1, x1, y1) in zip(points, points[1:]):
                life = 1.0 - (now - t1) / TRAIL_SECONDS
                if life <= 0:
                    continue
                c = tuple(int(v * life) for v in color)
                cv2.line(
                    glow,
                    (int(x0 * scale), int(y0 * scale)),
                    (int(x1 * scale), int(y1 * scale)),
                    c,
                    max(1, int((3 + 12 * life) * scale)),
                    cv2.LINE_AA,
                )
        for s in self._sparkles:
            life = 1.0 - (now - s.born) / SPARKLE_LIFE
            if life <= 0:
                continue
            c = tuple(int(v * life) for v in s.color)
            cv2.circle(glow, (int(s.x * scale), int(s.y * scale)), max(1, int(s.size * scale)), c, -1, cv2.LINE_AA)

        blurred = cv2.GaussianBlur(glow, (0, 0), sigmaX=5)
        layer = cv2.addWeighted(blurred, 1.3, glow, 0.35, 0)
        layer = cv2.resize(layer, (width, height), interpolation=cv2.INTER_LINEAR)

        # Crisp cores on the same layer: the trail's spine and twinkling stars.
        for color, points in trails:
            core = _brighten(color, 0.75)
            for (_, x0, y0), (t1, x1, y1) in zip(points, points[1:]):
                life = 1.0 - (now - t1) / TRAIL_SECONDS
                if life <= 0.35:
                    continue
                cv2.line(layer, (int(x0), int(y0)), (int(x1), int(y1)), core, max(1, int(2 * life)), cv2.LINE_AA)
        for s in self._sparkles:
            life = 1.0 - (now - s.born) / SPARKLE_LIFE
            twinkle = 0.6 + 0.4 * math.sin((now - s.born) * 18.0 + s.x)
            if life * twinkle < 0.35:
                continue
            r = max(1, int(s.size * life))
            x, y = int(s.x), int(s.y)
            cv2.line(layer, (x - r * 2, y), (x + r * 2, y), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(layer, (x, y - r * 2), (x, y + r * 2), (255, 255, 255), 1, cv2.LINE_AA)

        # People, and the face/person boxes on them, stay exactly as they are.
        for x1, y1, x2, y2 in clear:
            layer[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = 0

        cv2.add(frame, layer, dst=frame)

    def reset(self) -> None:
        with self._lock:
            self._paths.clear()
            self._last_seen.clear()
            self._sparkles.clear()
            self._last_draw = None
