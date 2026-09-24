"""Reusable multi-object tracking algorithm.

A SORT-style IoU tracker without the Kalman filter: association is by
intersection-over-union against a linearly predicted box. That is enough for
the high frame rates VisionGuard targets, and it needs no extra dependency
(no scipy, no filterpy).

This file knows nothing about people, modules or the pipeline. It takes boxes
in and hands identities back, so Phase 8's object and vehicle trackers can use
it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from app.core.context import BBox, Detection, Track

#: Weight of the newest motion sample when smoothing velocity.
_VELOCITY_SMOOTHING = 0.5


@dataclass
class TrackState:
    """One tracked identity and its recent history."""

    track_id: int
    label: str
    bbox: BBox
    confidence: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)

    def predicted_bbox(self) -> BBox:
        """Where this track is expected to be now.

        For a track matched on the previous frame this is just its last box.
        For a coasting track it is the last box shifted along its velocity,
        which is what lets a person be re-acquired after a brief occlusion.
        """
        if self.time_since_update == 0:
            return self.bbox

        dx = self.velocity[0] * self.time_since_update
        dy = self.velocity[1] * self.time_since_update
        return BBox(
            self.bbox.x1 + dx,
            self.bbox.y1 + dy,
            self.bbox.x2 + dx,
            self.bbox.y2 + dy,
        )

    def register_match(self, detection: Detection) -> None:
        """Absorb a matched detection and refresh the motion estimate."""
        steps = self.time_since_update + 1
        old_cx, old_cy = self.bbox.center
        new_cx, new_cy = detection.bbox.center

        instantaneous = ((new_cx - old_cx) / steps, (new_cy - old_cy) / steps)
        self.velocity = (
            _VELOCITY_SMOOTHING * instantaneous[0]
            + (1.0 - _VELOCITY_SMOOTHING) * self.velocity[0],
            _VELOCITY_SMOOTHING * instantaneous[1]
            + (1.0 - _VELOCITY_SMOOTHING) * self.velocity[1],
        )

        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.label = detection.label
        self.hits += 1
        self.time_since_update = 0

    def as_track(self, source_module: str = "") -> Track:
        """Convert to the shared context type."""
        return Track(
            track_id=self.track_id,
            label=self.label,
            bbox=self.bbox,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            source_module=source_module,
        )


@dataclass
class _Match:
    iou: float
    track_index: int
    detection_index: int


class IoUTracker:
    """Assigns persistent ids to detections across frames.

    Args:
        iou_threshold: Minimum overlap for a detection to continue a track.
        max_age: Frames a track survives unmatched before it is discarded.
            Higher values tolerate longer occlusions at the cost of stale ids.
        min_hits: Consecutive-ish matches before a track is reported. Filters
            out one-frame false positives; costs ``min_hits - 1`` frames of
            latency before a new person appears.
        publish_coasting: Report tracks that were not matched this frame.
            Off by default - a coasting box is a guess, not an observation.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 2,
        publish_coasting: bool = False,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if max_age < 0:
            raise ValueError("max_age must be >= 0")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.publish_coasting = publish_coasting

        self._tracks: List[TrackState] = []
        self._next_id = 1
        self._frames_seen = 0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def active_tracks(self) -> List[TrackState]:
        """Every live track, including unconfirmed and coasting ones."""
        return list(self._tracks)

    @property
    def total_ids_issued(self) -> int:
        return self._next_id - 1

    def reset(self) -> None:
        """Forget everything. Ids restart from 1."""
        self._tracks.clear()
        self._next_id = 1
        self._frames_seen = 0

    # ------------------------------------------------------------------ #
    # Per-frame update
    # ------------------------------------------------------------------ #

    def update(self, detections: Sequence[Detection]) -> List[TrackState]:
        """Advance the tracker by one frame.

        Args:
            detections: This frame's detections, from a detection module.

        Returns:
            The tracks to report, ordered by track id.
        """
        self._frames_seen += 1

        for track in self._tracks:
            track.age += 1

        matches, unmatched_detections = self._associate(detections)

        for match in matches:
            self._tracks[match.track_index].register_match(detections[match.detection_index])

        matched_track_indexes = {match.track_index for match in matches}
        for index, track in enumerate(self._tracks):
            if index not in matched_track_indexes:
                track.time_since_update += 1

        for index in unmatched_detections:
            self._spawn(detections[index])

        self._retire_stale()

        return self._reportable()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _associate(
        self, detections: Sequence[Detection]
    ) -> Tuple[List[_Match], List[int]]:
        """Greedy highest-IoU-first matching.

        Greedy rather than Hungarian on purpose: at video frame rates the two
        agree almost always, and greedy needs no scipy dependency. Candidate
        pairs are sorted by IoU descending with deterministic tie-breaking, so
        the result never depends on dict or set ordering.
        """
        if not detections or not self._tracks:
            return [], list(range(len(detections)))

        predicted = [track.predicted_bbox() for track in self._tracks]

        candidates: List[_Match] = []
        for track_index, track_bbox in enumerate(predicted):
            for detection_index, detection in enumerate(detections):
                # Never associate across classes.
                if detection.label != self._tracks[track_index].label:
                    continue
                iou = track_bbox.iou(detection.bbox)
                if iou >= self.iou_threshold:
                    candidates.append(_Match(iou, track_index, detection_index))

        candidates.sort(key=lambda m: (-m.iou, m.track_index, m.detection_index))

        matches: List[_Match] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        for candidate in candidates:
            if candidate.track_index in used_tracks:
                continue
            if candidate.detection_index in used_detections:
                continue
            matches.append(candidate)
            used_tracks.add(candidate.track_index)
            used_detections.add(candidate.detection_index)

        unmatched_detections = [
            index for index in range(len(detections)) if index not in used_detections
        ]
        return matches, unmatched_detections

    def _spawn(self, detection: Detection) -> None:
        self._tracks.append(
            TrackState(
                track_id=self._next_id,
                label=detection.label,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
        )
        self._next_id += 1

    def _retire_stale(self) -> None:
        self._tracks = [
            track for track in self._tracks if track.time_since_update <= self.max_age
        ]

    def _reportable(self) -> List[TrackState]:
        reportable = [
            track
            for track in self._tracks
            if track.hits >= self.min_hits
            and (self.publish_coasting or track.time_since_update == 0)
        ]
        reportable.sort(key=lambda track: track.track_id)
        return reportable
