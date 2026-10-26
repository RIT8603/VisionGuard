"""IoU tracker matching FindFace's documented track lifecycle.

* a detection joins a track when bbox overlap >= track_overlap_threshold
* a track closes after track_miss_interval seconds without a detection
* a track is force-closed after track_max_duration_frames (0 = never)
* missed frames between two detections are filled by linear interpolation
  (track_interpolate_bboxes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Iterable

import numpy as np

from .detector import Detection


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    id: int
    bbox: np.ndarray
    first_frame: int
    first_time: float
    last_frame: int
    last_time: float
    hits: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    # (frame_idx, bbox, detected) -- detected=False for interpolated entries
    history: list[tuple[int, np.ndarray, bool]] = field(default_factory=list)
    close_reason: str | None = None

    def predicted(self, frame_idx: int) -> np.ndarray:
        return self.bbox + self.velocity * (frame_idx - self.last_frame)

    @property
    def length(self) -> int:
        return self.last_frame - self.first_frame + 1


class Tracker:
    def __init__(self, overlap_threshold: float = 0.25, miss_interval: float = 1.0,
                 max_duration_frames: int = 0, interpolate: bool = True, max_tracks: int = 64):
        self.overlap_threshold = overlap_threshold
        self.miss_interval = miss_interval
        self.max_duration_frames = max_duration_frames
        self.interpolate = interpolate
        self.max_tracks = max_tracks
        self.active: dict[int, Track] = {}
        self._ids = count(1)

    def update(self, detections: Iterable[Detection], frame_idx: int, t: float
               ) -> tuple[list[tuple[Track, Detection]], list[Track]]:
        """Associate detections. Returns (matched pairs incl. new tracks, closed tracks)."""
        dets = list(detections)
        tracks = list(self.active.values())
        pairs = sorted(((iou(tr.predicted(frame_idx), d.bbox), ti, di)
                        for ti, tr in enumerate(tracks) for di, d in enumerate(dets)),
                       reverse=True)
        used_t, used_d, matched = set(), set(), []
        for overlap, ti, di in pairs:
            if overlap < self.overlap_threshold:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._extend(tracks[ti], dets[di].bbox, frame_idx, t)
            matched.append((tracks[ti], dets[di]))

        for di, d in enumerate(dets):
            if di in used_d or len(self.active) >= self.max_tracks:
                continue
            tr = Track(next(self._ids), d.bbox.copy(), frame_idx, t, frame_idx, t)
            tr.history.append((frame_idx, d.bbox.copy(), True))
            self.active[tr.id] = tr
            matched.append((tr, d))

        return matched, self._expire(frame_idx, t)

    def _extend(self, tr: Track, bbox: np.ndarray, frame_idx: int, t: float) -> None:
        gap = frame_idx - tr.last_frame
        if self.interpolate and gap > 1:
            for k in range(1, gap):
                alpha = k / gap
                tr.history.append((tr.last_frame + k, tr.bbox * (1 - alpha) + bbox * alpha, False))
        if gap > 0:
            tr.velocity = 0.5 * tr.velocity + 0.5 * (bbox - tr.bbox) / gap
        tr.bbox = bbox.copy()
        tr.last_frame, tr.last_time = frame_idx, t
        tr.hits += 1
        tr.history.append((frame_idx, bbox.copy(), True))

    def _expire(self, frame_idx: int, t: float) -> list[Track]:
        closed = []
        for tid, tr in list(self.active.items()):
            if t - tr.last_time > self.miss_interval:
                tr.close_reason = "miss_interval"
            elif self.max_duration_frames and frame_idx - tr.first_frame >= self.max_duration_frames:
                tr.close_reason = "max_duration"
            else:
                continue
            closed.append(self.active.pop(tid))
        return closed

    def close_all(self) -> list[Track]:
        closed = list(self.active.values())
        for tr in closed:
            tr.close_reason = "end_of_stream"
        self.active.clear()
        return closed
