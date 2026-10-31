"""The output sink that feeds the dashboard.

Lives in the backend, not in ``computer_vision``, because it is the bridge
between the two and the dependency must point one way: **backend knows about
the vision engine; the vision engine knows nothing about the backend.**

Keeping it here means the pipeline still runs with no server, no API and no
browser - it is just another :class:`BaseOutputSink` as far as the core is
concerned.

Live view
---------
The pipeline hands this sink one frame per *analysed* frame. When analysis is
slow - one inference can take a large part of a second on a busy CPU - a view
built only from those frames is a slideshow, and it freezes whenever a single
frame takes long.

So when the source can be read independently (``BaseVideoSource.peek``), the
picture comes from a small thread here instead: it takes the camera's frames
at the stream rate, draws analysis results over them and publishes that.
Detection speed then no longer decides whether the video moves. Results,
statistics and events are untouched - they still come from analysed frames
only, exactly once each.

Where the boxes go is the hard part: a result describes a frame from a moment
ago, and people have moved since. Two ways of drawing them:

* **synced** (default): the picture is held back by about one analysis
  interval, so every frame shown lies *between* two analysed frames and each
  box is placed between its two known positions. Boxes sit on what they mark;
  the price is that the view runs that far behind the camera.
* **live**: the newest frame, with each box moved along its track's velocity
  for the time that has passed. No delay, but a guess: measured on a street
  clip the boxes overlapped their targets by 0.72 (IoU) at 5 analysed frames
  per second, 0.56 at 2 and 0.40 at 1.

Synced falls back to live while analysis is more than a few seconds behind,
because a view that far in the past is worse than one with loose boxes.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.services.state import DashboardState
from computer_vision.core.config import ProcessingConfig
from computer_vision.core.context import BBox, FrameContext, json_safe
from computer_vision.core.output import BaseOutputSink, untracked_detections
from computer_vision.utils.annotate import draw_detections, draw_overlays, draw_tracks, track_names, tracks_to_draw
from computer_vision.utils.magic_fx import MagicTrails, magic_enabled
from computer_vision.utils.logger import get_logger
from computer_vision.utils.timing import PipelineStats

logger = get_logger(__name__)

#: ``peek`` of a video source: newest frame (or None) and its sequence number.
Peek = Callable[[], Tuple[Optional[np.ndarray], int]]
#: Brings a raw frame to the geometry analysis used (mirror, resize).
Prepare = Callable[[np.ndarray], np.ndarray]

#: Boxes older than this are not drawn on a live frame: by then they say
#: where things were, not where they are. Zones are drawn regardless.
STALE_RESULTS_SEC = 3.0
#: The live view counts as working while it published within this long;
#: otherwise analysed frames are shown, as they are without a live view.
_LIVE_GRACE_SEC = 1.0
#: Frame rates when nobody has a stream open: enough for a thumbnail that is
#: fetched about once a second, and almost nothing when no one is looking.
_SNAPSHOT_FPS = 2.0
_IDLE_FPS = 0.5
#: Live-view rate when ``stream_fps`` is 0 ("every frame").
_UNCAPPED_FPS = 30.0
#: Synced view: the furthest behind the camera the picture may run. Past
#: this, the newest frame is shown instead, with boxes moved by velocity.
MAX_SYNC_DELAY_SEC = 3.0
#: Synced view: how far the playhead may trail the ideal position before it
#: jumps forward (analysis sped up, or a frame took unusually long).
_SYNC_SLACK_SEC = 0.5
#: Live view: a box is moved along its velocity for at most this long, and
#: never further than its own size - beyond that the guess is worth nothing.
_MAX_EXTRAPOLATION_SEC = 1.5
#: Synced view: how many analysis intervals the playhead aims to stay behind
#: the newest result right after it arrives. 1.0 would be just enough if
#: results came evenly; the extra half absorbs one arriving late.
_SYNC_LAG_INTERVALS = 1.5
#: Analysed results kept for the synced view.
_HISTORY = 5


@dataclass(frozen=True)
class _Results:
    """What the last analysed frame found, kept for drawing on live frames."""

    overlays: List[Any] = field(default_factory=list)
    detections: List[Any] = field(default_factory=list)
    tracks: List[Any] = field(default_factory=list)
    #: When the results were published.
    at: float = 0.0
    #: When the analysed frame was captured (same clock as ``time.perf_counter``).
    ts: float = 0.0


def _track_key(track: Any) -> Tuple[str, int]:
    return getattr(track, "source_module", ""), track.track_id


def _blend(a: BBox, b: BBox, alpha: float) -> BBox:
    return BBox(
        a.x1 + (b.x1 - a.x1) * alpha,
        a.y1 + (b.y1 - a.y1) * alpha,
        a.x2 + (b.x2 - a.x2) * alpha,
        a.y2 + (b.y2 - a.y2) * alpha,
    )


def tracks_between(before: _Results, after: _Results, alpha: float) -> List[Any]:
    """Tracks as they stood ``alpha`` of the way from ``before`` to ``after``.

    A track present at both ends is placed between its two boxes. One that
    exists at only one end is shown for the half of the interval nearer to
    that end - it appeared or disappeared somewhere in between.
    """
    later: Dict[Tuple[str, int], Any] = {_track_key(t): t for t in after.tracks}
    shown: List[Any] = []
    for track in before.tracks:
        match = later.pop(_track_key(track), None)
        if match is not None:
            base = track if alpha < 0.5 else match
            shown.append(replace(base, bbox=_blend(track.bbox, match.bbox, alpha)))
        elif alpha < 0.5:
            shown.append(track)
    if alpha >= 0.5:
        shown.extend(later.values())
    return shown


def tracks_moved_on(tracks: List[Any], seconds: float) -> List[Any]:
    """Tracks moved along their own velocity for ``seconds`` (a bounded guess)."""
    seconds = max(0.0, min(seconds, _MAX_EXTRAPOLATION_SEC))
    moved: List[Any] = []
    for track in tracks:
        velocity = getattr(track, "extra", {}).get("velocity")
        if not velocity or seconds <= 0:
            moved.append(track)
            continue
        dx, dy = velocity[0] * seconds, velocity[1] * seconds
        limit = max(track.bbox.width, track.bbox.height)
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > limit > 0:
            dx, dy = dx * limit / distance, dy * limit / distance
        box = track.bbox
        moved.append(
            replace(track, bbox=BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy))
        )
    return moved


class WebSink(BaseOutputSink):
    """Publishes annotated frames and results to the dashboard.

    Encoding every frame to JPEG costs real time, so the stream is throttled
    independently of the pipeline: analysis still runs at full rate, only the
    *view* is capped. Results and events are published every frame regardless,
    because dropping an event to save bandwidth would be a bug.
    """

    def __init__(
        self,
        state: DashboardState,
        config: ProcessingConfig,
        stream_fps: float = 15.0,
        jpeg_quality: int = 75,
        sync: bool = True,
    ) -> None:
        self._state = state
        self._config = config
        self._min_interval = 1.0 / stream_fps if stream_fps > 0 else 0.0
        self._quality = int(jpeg_quality)
        self._next_encode_at = 0.0

        self._peek: Optional[Peek] = None
        self._prepare: Optional[Prepare] = None
        self._results = _Results()
        self._sync = bool(sync)
        #: Recent results, oldest first. Replaced whole, never mutated, so the
        #: live thread can read it without a lock.
        self._history: Tuple[_Results, ...] = ()
        #: Synced view only: frames waiting to be shown, as (captured, frame).
        self._frames: Deque[Tuple[float, np.ndarray]] = deque()
        self._playhead: Optional[float] = None
        self._playhead_moved_at = 0.0
        self._shown_ts: Optional[float] = None
        self._live_thread: Optional[threading.Thread] = None
        self._live_stop = threading.Event()
        self._live_published_at = 0.0
        #: The magic effect remembers paths across frames, so one per camera.
        self._fx = MagicTrails()

    # ------------------------------------------------------------------ #
    # Live view
    # ------------------------------------------------------------------ #

    def attach_live_view(self, peek: Peek, prepare: Prepare) -> None:
        """Show the camera at its own rate, independent of analysis speed.

        Call before the pipeline runs. ``peek`` returning ``(None, 0)`` (a
        source that cannot be read independently, or one that is
        reconnecting) is fine: analysed frames are shown meanwhile.
        """
        self._peek = peek
        self._prepare = prepare

    @property
    def live_view_active(self) -> bool:
        """True while live frames, not analysed ones, are what viewers see."""
        return (
            self._live_thread is not None
            and (time.perf_counter() - self._live_published_at) < _LIVE_GRACE_SEC
        )

    def _start_live_view(self) -> None:
        if self._peek is None or self._live_thread is not None:
            return
        self._live_stop.clear()
        self._live_thread = threading.Thread(
            target=self._live_loop, name="visionguard-live-view", daemon=True
        )
        self._live_thread.start()

    def _live_interval(self) -> float:
        demand = self._state.viewer_demand()
        if demand == "stream":
            return self._min_interval or 1.0 / _UNCAPPED_FPS
        return 1.0 / (_SNAPSHOT_FPS if demand == "snapshot" else _IDLE_FPS)

    def _live_loop(self) -> None:
        assert self._peek is not None and self._prepare is not None
        last_sequence = -1
        last_attempt = 0.0
        while not self._live_stop.is_set():
            # The interval is re-read on every pass (at most 0.1 s apart), so
            # a viewer opening the stream is not kept waiting by an idle rate.
            due = last_attempt + self._live_interval()
            now = time.perf_counter()
            if now < due:
                self._live_stop.wait(min(due - now, 0.1))
                continue
            try:
                frame, sequence = self._peek()
                if frame is None:
                    # Nothing to show (camera reconnecting): look again at
                    # the normal rate rather than spinning.
                    last_attempt = now
                    continue
                if sequence == last_sequence:
                    self._live_stop.wait(0.01)  # the next frame is due any moment
                    continue
                last_sequence = sequence
                last_attempt = now
                jpeg = self._encode_live(frame)
            except Exception:  # noqa: BLE001 - the view must never stop analysis
                logger.debug("Live view frame failed", exc_info=True)
                last_attempt = now
                continue
            if jpeg is not None and not self._live_stop.is_set():
                self._state.publish_frame(jpeg)
                self._live_published_at = time.perf_counter()

    def _encode_live(self, raw: np.ndarray) -> Optional[bytes]:
        assert self._prepare is not None
        frame = self._prepare(raw)
        now = time.perf_counter()

        if self._sync and self._state.viewer_demand() == "stream":
            synced = self._synced_frame(frame, now)
            if synced is not None:
                picture, results, tracks = synced
                if picture is None:
                    return None  # the playhead is waiting on analysis
                return self._draw(picture.copy(), results, tracks, fresh=True)
        else:
            self._frames.clear()
            self._playhead = None

        results = self._results
        if frame is raw:
            frame = raw.copy()  # the capture's own array: never draw on it
        fresh = (now - results.at) <= STALE_RESULTS_SEC
        return self._draw(frame, results, tracks_moved_on(results.tracks, now - results.ts), fresh)

    def _draw(self, frame: np.ndarray, results: _Results, tracks: List[Any], fresh: bool):
        if results.overlays:
            # Dynamic alert boxes belong to the analysed person, unlike static
            # zone geometry: do not keep them on an unrelated stale live frame.
            draw_overlays(frame, [o for o in results.overlays if o.kind != "alert"])
        if fresh and magic_enabled():
            self._fx.draw(frame, tracks, keep_clear=results.detections)
        if self._config.show_boxes and fresh:
            draw_detections(frame, results.detections)
            draw_tracks(frame, tracks, track_names(results.detections))
        if fresh:
            draw_overlays(frame, [o for o in results.overlays if o.kind == "alert"])
        return self._jpeg(frame)

    def _synced_frame(self, frame: np.ndarray, now: float):
        """The delayed frame to show, with the results that belong on it.

        Returns ``None`` when the synced view cannot be used right now (the
        caller then shows the newest frame), or ``(picture, results, tracks)``
        - where ``picture`` is ``None`` if nothing new is due yet.
        """
        frames = self._frames
        frames.append((now, frame))
        while frames and now - frames[0][0] > MAX_SYNC_DELAY_SEC + 1.0:
            frames.popleft()

        history = self._history
        if len(history) < 2 or now - history[-1].ts > MAX_SYNC_DELAY_SEC:
            self._playhead = None
            return None

        newest = history[-1].ts
        # Results do not arrive evenly (several cameras take turns at the
        # model, and a busy CPU stretches some frames). Running a little more
        # than one interval behind leaves room for a late one, so the picture
        # does not have to stop and wait for it.
        interval = max(b.ts - a.ts for a, b in zip(history, history[1:]))
        target = max(history[0].ts, newest - _SYNC_LAG_INTERVALS * interval)
        if self._playhead is None:
            self._playhead = target
        else:
            self._playhead += now - self._playhead_moved_at
        self._playhead_moved_at = now
        # Never ahead of what has been analysed, and never much further
        # behind than intended.
        if self._playhead > newest:
            self._playhead = newest
        elif self._playhead < target - interval - _SYNC_SLACK_SEC:
            self._playhead = target
        playhead = self._playhead

        while len(frames) > 1 and frames[1][0] <= playhead:
            frames.popleft()
        shown_ts, picture = frames[0]
        if shown_ts > playhead + _SYNC_SLACK_SEC:
            # The viewer has only just connected: nothing was kept from the
            # moment the playhead is at. Show the newest frame until what has
            # been kept reaches back far enough, rather than a still picture.
            return None
        if shown_ts == self._shown_ts:
            return None, history[-1], []
        self._shown_ts = shown_ts

        before = next((r for r in reversed(history) if r.ts <= playhead), history[0])
        after = next((r for r in history if r.ts >= playhead), history[-1])
        span = after.ts - before.ts
        alpha = min(1.0, max(0.0, (playhead - before.ts) / span)) if span > 0 else 1.0
        nearer = before if alpha < 0.5 else after
        return picture, nearer, tracks_between(before, after, alpha)

    # ------------------------------------------------------------------ #
    # Analysed frames
    # ------------------------------------------------------------------ #

    def emit(self, context: FrameContext, stats: PipelineStats) -> bool:
        if self._peek is not None:
            self._results = _Results(
                overlays=list(context.overlays),
                detections=untracked_detections(context),
                tracks=context.get_tracks(),
                at=time.perf_counter(),
                ts=context.monotonic_ts,
            )
            self._history = (self._history + (self._results,))[-_HISTORY:]
            self._start_live_view()

        self._state.publish(
            # While the live view is delivering, an analysed frame would be
            # an older picture arriving after newer ones.
            jpeg=None if self.live_view_active else self._encode(context),
            summary=context.summary(),
            # Everything crossing into the HTTP layer must be JSON-encodable;
            # one NumPy value or backend object would 500 every status call.
            stats=json_safe(stats.as_dict()),
            events=json_safe([event.to_dict() for event in context.events]),
            captured_at=context.wall_time.timestamp(),
        )
        return True

    def _encode(self, context: FrameContext) -> Optional[bytes]:
        now = time.perf_counter()
        if self._min_interval:
            # A fixed cadence, not "interval since the last encode": frames
            # arrive on the source's own clock, so the one due at 66.7 ms often
            # lands at 66.6 ms - skipping it turned 15 fps into ~11 on a 30 fps
            # video. A quarter interval of slack absorbs that jitter; deadlines
            # still advance by a full interval, so the average never exceeds
            # stream_fps.
            if now < self._next_encode_at - self._min_interval / 4:
                return None
            on_time = now - self._next_encode_at < self._min_interval
            self._next_encode_at = (self._next_encode_at if on_time else now) + self._min_interval

        frame = context.frame.copy()
        if context.overlays:
            draw_overlays(frame, [o for o in context.overlays if o.kind != "alert"])
        if magic_enabled():
            self._fx.draw(frame, context.get_tracks(), keep_clear=context.get_detections())
        if self._config.show_boxes:
            draw_detections(frame, untracked_detections(context))
            draw_tracks(frame, tracks_to_draw(context), track_names(context.get_detections("face_recognition")))
        draw_overlays(frame, [o for o in context.overlays if o.kind == "alert"])
        return self._jpeg(frame)

    def _jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
        )
        return buffer.tobytes() if ok else None

    def close(self) -> None:
        self._live_stop.set()
        thread = self._live_thread
        self._live_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._state.stop()
