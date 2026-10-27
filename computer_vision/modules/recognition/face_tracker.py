"""Face tracker: links face boxes across frames (ported from FindFace-lite).

FindFace Multi links a face from frame to frame by box overlap, closes a track
after ``track_miss_interval`` seconds without a detection, and predicts where a
moving face will be. This is the same lifecycle, from the FindFace-lite
reference (``FindFace/findface_lite/tracker.py``):

* **Constant-velocity prediction.** Each track keeps a smoothed per-frame
  velocity of its box, and a new detection is compared with where the track is
  *expected* to be, not where it last was. A face walking across the frame
  keeps its track instead of splitting into a new one every few frames.
* **Global greedy assignment.** Every track-detection pair is scored by IoU and
  the best pairs are taken first, so two faces passing close to each other do
  not steal each other's tracks the way per-face nearest matching can.
* **Miss interval in seconds,** so the behaviour does not change with the
  frame rate.

Used by face recognition to know which face is "the same person as before"
when there is no person track to attach it to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _center_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between box centres, in units of the larger box side."""
    side = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    dx = (a[0] + a[2] - b[0] - b[2]) / 2
    dy = (a[1] + a[3] - b[1] - b[3]) / 2
    return float((dx * dx + dy * dy) ** 0.5 / side)


@dataclass
class FaceTrack:
    id: int
    bbox: np.ndarray
    last_frame: int
    last_time: float
    hits: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    def predicted(self, frame_id: int) -> np.ndarray:
        # A few frames of coasting through a missed detection; further than
        # that the velocity estimate is a guess and the box would fly off.
        return self.bbox + self.velocity * min(frame_id - self.last_frame, 8)


class FaceTracker:
    """IoU tracker with velocity prediction and a time-based miss interval."""

    def __init__(
        self,
        overlap_threshold: float = 0.25,
        miss_interval: float = 1.0,
        max_tracks: int = 64,
        center_gate: float = 0.0,
    ) -> None:
        #: Max centre distance, in units of the larger box side, for linking
        #: boxes that no longer overlap enough. 0 = IoU only (original behaviour).
        self.center_gate = float(center_gate)
        self.overlap_threshold = float(overlap_threshold)
        self.miss_interval = float(miss_interval)
        self.max_tracks = int(max_tracks)
        self.active: Dict[int, FaceTrack] = {}
        self._ids = count(1)

    def update(self, boxes: Sequence[Box], frame_id: int, t: float) -> Tuple[List[int], List[int]]:
        """Associate this frame's face boxes.

        Returns ``(track_ids, closed_ids)``: one track id per input box (in the
        same order; ``-1`` when ``max_tracks`` is reached), and the ids of
        tracks that expired this frame.
        """
        dets = [np.asarray(b, dtype=np.float32) for b in boxes]
        tracks = list(self.active.values())
        pairs = sorted(
            (
                (box_iou(track.predicted(frame_id), det), ti, di)
                for ti, track in enumerate(tracks)
                for di, det in enumerate(dets)
            ),
            reverse=True,
        )

        ids = [-1] * len(dets)
        used_tracks: set[int] = set()
        for overlap, ti, di in pairs:
            if overlap < self.overlap_threshold:
                break
            if ti in used_tracks or ids[di] != -1:
                continue
            used_tracks.add(ti)
            self._extend(tracks[ti], dets[di], frame_id, t)
            ids[di] = tracks[ti].id

        # Small faces: a few pixels of jitter drops the IoU of a 12 px box
        # below the threshold, so the track would split. Whatever is still
        # unmatched is linked by centre distance (nearest first) instead.
        if self.center_gate > 0:
            loose = sorted(
                (
                    (d, ti, di)
                    for ti, track in enumerate(tracks)
                    if ti not in used_tracks
                    for di, det in enumerate(dets)
                    if ids[di] == -1
                    # Jitter makes the velocity estimate overshoot, so a tiny
                    # face may be nearer to where it was than to where it was
                    # predicted: accept either.
                    for d in [min(_center_gap(track.predicted(frame_id), det), _center_gap(track.bbox, det))]
                    if d <= self.center_gate
                ),
            )
            for _, ti, di in loose:
                if ti in used_tracks or ids[di] != -1:
                    continue
                used_tracks.add(ti)
                self._extend(tracks[ti], dets[di], frame_id, t)
                ids[di] = tracks[ti].id

        for di, det in enumerate(dets):
            if ids[di] != -1 or len(self.active) >= self.max_tracks:
                continue
            track = FaceTrack(next(self._ids), det.copy(), frame_id, t)
            self.active[track.id] = track
            ids[di] = track.id

        return ids, self._expire(t)

    def _extend(self, track: FaceTrack, bbox: np.ndarray, frame_id: int, t: float) -> None:
        gap = frame_id - track.last_frame
        if gap > 0:
            track.velocity = 0.5 * track.velocity + 0.5 * (bbox - track.bbox) / gap
        track.bbox = bbox.copy()
        track.last_frame, track.last_time = frame_id, t
        track.hits += 1

    def _expire(self, t: float) -> List[int]:
        closed = [tid for tid, track in self.active.items() if t - track.last_time > self.miss_interval]
        for tid in closed:
            del self.active[tid]
        return closed

    def reset(self) -> None:
        self.active.clear()
