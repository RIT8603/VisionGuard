"""Best-snapshot selection: when, and which, face of a track gets posted.

Mirrors FindFace's two documented modes:

offline (overall_only=True)
    Buffer the best candidate while the face is visible; post exactly one
    snapshot -- the best -- when the track closes.

real-time (overall_only=False)
    Time is cut into windows of realtime_post_interval seconds.
    * realtime_post_first_immediately: post the first candidate that passes
      the filters without waiting for the first window to end.
    * realtime_post_every_interval=True: post the best candidate of every window.
    * realtime_post_every_interval=False: post a window's best only if it beats
      everything posted so far ("in order of increasing quality").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Candidate:
    quality: float
    frame_idx: int
    t: float
    payload: Any = None          # crop / full frame / bbox -- opaque here


@dataclass
class Post:
    track_id: int
    reason: str                  # overall_best | first_immediately | interval_best | quality_improved
    candidate: Candidate


@dataclass
class _State:
    overall_best: Candidate | None = None
    window_start: float | None = None
    window_best: Candidate | None = None
    last_posted_quality: float = float("-inf")
    posts: int = 0


class SnapshotSelector:
    def __init__(self, overall_only: bool = True, interval: float = 1.0,
                 every_interval: bool = False, first_immediately: bool = False):
        self.overall_only = overall_only
        self.interval = interval
        self.every_interval = every_interval
        self.first_immediately = first_immediately
        self._states: dict[int, _State] = {}

    def state(self, track_id: int) -> _State:
        return self._states.setdefault(track_id, _State())

    def offer(self, track_id: int, cand: Candidate) -> list[Post]:
        """Feed a candidate that already passed all filters."""
        st = self.state(track_id)
        if st.overall_best is None or cand.quality > st.overall_best.quality:
            st.overall_best = cand
        if self.overall_only:
            return []

        posts = self._roll_window(track_id, st, cand.t)
        if st.window_start is None:
            st.window_start = cand.t
            if self.first_immediately and st.posts == 0:
                posts.append(self._post(track_id, st, "first_immediately", cand))
                return posts
        if st.window_best is None or cand.quality > st.window_best.quality:
            st.window_best = cand
        return posts

    def tick(self, t: float) -> list[Post]:
        """Close real-time windows that ended even if no candidate arrived."""
        if self.overall_only:
            return []
        posts = []
        for tid, st in self._states.items():
            posts += self._roll_window(tid, st, t)
        return posts

    def close(self, track_id: int) -> tuple[list[Post], _State | None]:
        st = self._states.pop(track_id, None)
        if st is None:
            return [], None
        if self.overall_only:
            posts = [Post(track_id, "overall_best", st.overall_best)] if st.overall_best else []
        else:
            posts = self._flush_window(track_id, st)
        return posts, st

    # -- internals ---------------------------------------------------------------
    def _roll_window(self, tid: int, st: _State, t: float) -> list[Post]:
        posts = []
        while st.window_start is not None and t - st.window_start >= self.interval:
            posts += self._flush_window(tid, st)
            st.window_start += self.interval
        return posts

    def _flush_window(self, tid: int, st: _State) -> list[Post]:
        best, st.window_best = st.window_best, None
        if best is None:
            return []
        if self.every_interval:
            return [self._post(tid, st, "interval_best", best)]
        if best.quality > st.last_posted_quality:
            return [self._post(tid, st, "quality_improved", best)]
        return []

    def _post(self, tid: int, st: _State, reason: str, cand: Candidate) -> Post:
        st.posts += 1
        st.last_posted_quality = max(st.last_posted_quality, cand.quality)
        return Post(tid, reason, cand)
