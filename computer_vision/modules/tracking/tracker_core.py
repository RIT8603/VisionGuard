"""Reusable multi-object tracking algorithm.

A SORT-style IoU tracker without the Kalman filter: association is by
intersection-over-union against a linearly predicted box, and it needs no
extra dependency (no scipy, no filterpy).

At a low processing rate a walking person moves further than their own width
between two processed frames, and the overlap that association relies on is
simply gone. Two things keep an identity alive there: every track is moved
along its measured velocity before matching, and - when ``max_center_distance``
is set - a detection close to where a track should be can continue it even
with no overlap at all.

This file knows nothing about people, modules or the pipeline. It takes boxes
in and hands identities back, so Phase 8's object and vehicle trackers can use
it unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from computer_vision.core.context import BBox, Detection, Track

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
    #: Pixels per unit of the tracker's clock: per second when the tracker is
    #: given timestamps, per frame when it is not.
    velocity: Tuple[float, float] = (0.0, 0.0)
    #: Tracker clock when this track was last matched.
    last_seen: float = 0.0
    #: Box centre when the track was first seen.
    origin: Tuple[float, float] | None = None
    #: Furthest the centre has been from ``origin``, in units of the box's own
    #: size at that moment. Only ever grows. A mannequin, a poster or a coat
    #: rack mistaken for a person stays near 0; anything that walked into view
    #: passes 1 within a step or two.
    travel: float = 0.0

    def __post_init__(self) -> None:
        if self.origin is None:
            self.origin = self.bbox.center

    def predicted_bbox(self, now: float | None = None) -> BBox:
        """Where this track is expected to be at clock time ``now``.

        The last box moved along the track's velocity for as long as it has
        been unseen. Without ``now`` that is one step per frame since the
        last match, the current frame included. A track with no motion
        estimate yet (seen once) predicts its last box.
        """
        if self.velocity == (0.0, 0.0):
            return self.bbox

        steps = (now - self.last_seen) if now is not None else self.time_since_update + 1
        dx = self.velocity[0] * steps
        dy = self.velocity[1] * steps
        return BBox(
            self.bbox.x1 + dx,
            self.bbox.y1 + dy,
            self.bbox.x2 + dx,
            self.bbox.y2 + dy,
        )

    def register_match(self, detection: Detection, now: float | None = None) -> None:
        """Absorb a matched detection and refresh the motion estimate."""
        if now is None:
            steps = float(self.time_since_update + 1)
            now = self.last_seen + steps
        else:
            steps = max(now - self.last_seen, 1e-6)
        self.last_seen = now
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

        size = max(self.bbox.width, self.bbox.height)
        if size > 0 and self.origin is not None:
            moved = math.hypot(new_cx - self.origin[0], new_cy - self.origin[1])
            self.travel = max(self.travel, moved / size)

    def as_track(self, source_module: str = "", timed: bool = False) -> Track:
        """Convert to the shared context type.

        ``timed`` says the velocity is in pixels per second (the tracker was
        given timestamps); only then is it published, for a viewer that wants
        to draw the box where the thing is *now* rather than where it was.
        """
        extra: dict = {"travel": round(self.travel, 2)}
        if timed:
            extra["velocity"] = [round(self.velocity[0], 1), round(self.velocity[1], 1)]
        return Track(
            track_id=self.track_id,
            label=self.label,
            bbox=self.bbox,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            source_module=source_module,
            extra=extra,
        )


@dataclass
class _Match:
    #: Overlap for an IoU match; closeness (1 - normalised distance) for a
    #: distance match. Higher is better either way.
    iou: float
    track_index: int
    detection_index: int


def _overlap(predicted: BBox, last: BBox, detection: BBox) -> float:
    """Overlap of a detection with where a track is expected *or* last was.

    The velocity estimate comes from box centres, and a person's box jitters
    as limbs swing and the detector re-frames them; on CCTV that pushed the
    prediction off people who had barely moved (measured: 6-15% of the frames
    a visible person lost their box). Accepting the better of the two keeps
    such a track, and still follows a walker the prediction leads correctly.
    """
    return max(predicted.iou(detection), last.iou(detection))


#: A distance match also requires the two boxes to be about the same size,
#: so a far-away person is never continued by a nearby one.
_MAX_AREA_RATIO = 2.5
#: ...and only for a track seen on one of the last two frames. A track that
#: has been gone longer could be anywhere; handing it to whoever is nearest
#: merges two people into one identity.
_DISTANCE_MAX_GAP = 1


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
        match_across_labels: Let a detection continue a track of a different
            label. Off by default (a car must never inherit a dog's id). Turn
            it on only where every label is the same kind of thing and the
            detector wavers between them - a van read as ``car`` one frame and
            ``truck`` the next is one vehicle, not two.
        candidate_iou_threshold: Overlap a *low-confidence candidate* needs to
            continue a track (see :meth:`update`). Defaults to
            ``iou_threshold``.
        max_center_distance: Second chance for a detection that overlaps no
            track: it continues the nearest unmatched track of the same label
            whose predicted centre is within this many box-sizes, provided the
            two boxes are of similar size. 0 (default) turns it off. Meant for
            low processing rates, where a walker clears their own width
            between frames. Measured on a street clip with 0.8: identity
            switches fell 106 -> 41 at 2 fps and 136 -> 88 at 1 fps, and it
            made no useful difference at 5 fps and above - which is why the
            tracking modules switch it on only while the rate is low.
        compact_ids: Issue an id only when a track is confirmed (reaches
            ``min_hits``), instead of to every first sighting. Ids then count
            things that were actually reported rather than every flicker the
            detector produced. Off by default.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 2,
        publish_coasting: bool = False,
        match_across_labels: bool = False,
        candidate_iou_threshold: float | None = None,
        max_center_distance: float = 0.0,
        compact_ids: bool = False,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if max_age < 0:
            raise ValueError("max_age must be >= 0")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if candidate_iou_threshold is None:
            candidate_iou_threshold = iou_threshold
        if not 0.0 < candidate_iou_threshold <= 1.0:
            raise ValueError("candidate_iou_threshold must be in (0, 1]")
        if max_center_distance < 0:
            raise ValueError("max_center_distance must be >= 0")

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.publish_coasting = publish_coasting
        self.match_across_labels = match_across_labels
        self.candidate_iou_threshold = candidate_iou_threshold
        self.max_center_distance = float(max_center_distance)
        self.compact_ids = bool(compact_ids)

        self._tracks: List[TrackState] = []
        self._next_id = 1
        self._frames_seen = 0
        self._reset_clock()

    def _reset_clock(self) -> None:
        #: The tracker's own clock. With timestamps it runs in seconds; with
        #: none it advances by one per frame, so "per clock unit" is then
        #: "per frame" and everything behaves as a frame-counting tracker.
        self._clock = 0.0
        self._last_timestamp: float | None = None
        self._gap = 1.0
        self._timed = False

    @property
    def timed(self) -> bool:
        """True once real timestamps drive the clock (velocities are per second)."""
        return self._timed

    def _advance_clock(self, timestamp: float | None) -> None:
        if (
            timestamp is not None
            and self._last_timestamp is not None
            and timestamp > self._last_timestamp
        ):
            gap = timestamp - self._last_timestamp
            if not self._timed:
                # First real gap: put what was learned so far (in frame
                # units) onto the same footing, so nothing jumps.
                for track in self._tracks:
                    track.velocity = (track.velocity[0] / gap, track.velocity[1] / gap)
                    track.last_seen = self._clock - (self._clock - track.last_seen) * gap
                self._timed = True
            self._gap = gap
        if timestamp is not None:
            self._last_timestamp = timestamp
        self._clock += self._gap

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
        self._reset_clock()

    # ------------------------------------------------------------------ #
    # Per-frame update
    # ------------------------------------------------------------------ #

    def update(
        self,
        detections: Sequence[Detection],
        candidates: Sequence[Detection] = (),
        timestamp: float | None = None,
    ) -> List[TrackState]:
        """Advance the tracker by one frame.

        Args:
            timestamp: When this frame was captured, in seconds on any
                monotonic clock. Optional. With it, motion is measured per
                second, so frames that arrive unevenly (a busy CPU, several
                cameras taking turns) are predicted correctly; without it
                every frame is assumed to be one step after the last.
            detections: This frame's detections, from a detection module.
            candidates: Boxes the detector saw but scored below its reporting
                threshold. A person half hidden behind an umbrella usually
                still produces one of these. They are used only to *continue*
                a confirmed track that found no detection this frame (the
                ByteTrack idea) - a candidate never starts a track, so a
                low-score false positive cannot become an identity.

        Returns:
            The tracks to report, ordered by track id.
        """
        self._frames_seen += 1
        self._advance_clock(timestamp)

        for track in self._tracks:
            track.age += 1

        matches, unmatched_detections = self._associate(detections)

        if self.max_center_distance > 0 and unmatched_detections:
            nearby = self._associate_by_distance(
                detections, unmatched_detections, {m.track_index for m in matches}
            )
            matches.extend(nearby)
            taken = {m.detection_index for m in nearby}
            unmatched_detections = [i for i in unmatched_detections if i not in taken]

        for match in matches:
            self._match(self._tracks[match.track_index], detections[match.detection_index])

        matched_track_indexes = {match.track_index for match in matches}

        if candidates:
            for match in self._associate_candidates(candidates, matched_track_indexes):
                self._match(
                    self._tracks[match.track_index], candidates[match.detection_index]
                )
                matched_track_indexes.add(match.track_index)

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

        predicted = [track.predicted_bbox(self._clock) for track in self._tracks]

        candidates: List[_Match] = []
        for track_index, track_bbox in enumerate(predicted):
            last_bbox = self._tracks[track_index].bbox
            for detection_index, detection in enumerate(detections):
                # Never associate across classes, unless told the labels
                # are interchangeable (match_across_labels).
                if (
                    not self.match_across_labels
                    and detection.label != self._tracks[track_index].label
                ):
                    continue
                iou = _overlap(track_bbox, last_bbox, detection.bbox)
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

    def _associate_by_distance(
        self,
        detections: Sequence[Detection],
        unmatched_detections: Sequence[int],
        matched_tracks: set[int],
    ) -> List[_Match]:
        """Continue tracks that no detection overlaps, by closeness instead.

        Only what IoU matching left over takes part, so this can never take a
        detection away from a track it overlaps. Nearest pair first.
        """
        pairs: List[_Match] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in matched_tracks:
                continue
            predicted = track.predicted_bbox(self._clock)
            size = max(predicted.width, predicted.height)
            if size <= 0 or predicted.area <= 0:
                continue
            px, py = predicted.center
            if track.time_since_update > _DISTANCE_MAX_GAP:
                continue
            for detection_index in unmatched_detections:
                detection = detections[detection_index]
                if not self.match_across_labels and detection.label != track.label:
                    continue
                area = detection.bbox.area
                if area <= 0:
                    continue
                ratio = area / predicted.area
                if ratio > _MAX_AREA_RATIO or ratio < 1.0 / _MAX_AREA_RATIO:
                    continue
                dx, dy = detection.bbox.center
                distance = math.hypot(dx - px, dy - py) / size
                if distance <= self.max_center_distance:
                    pairs.append(_Match(1.0 - distance, track_index, detection_index))

        pairs.sort(key=lambda m: (-m.iou, m.track_index, m.detection_index))

        matches: List[_Match] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for pair in pairs:
            if pair.track_index in used_tracks or pair.detection_index in used_detections:
                continue
            matches.append(pair)
            used_tracks.add(pair.track_index)
            used_detections.add(pair.detection_index)
        return matches

    def _match(self, track: TrackState, detection: Detection) -> None:
        track.register_match(detection, self._clock)
        if track.track_id == 0 and track.hits >= self.min_hits:
            track.track_id = self._issue_id()

    def _issue_id(self) -> int:
        issued = self._next_id
        self._next_id += 1
        return issued

    def _associate_candidates(
        self, candidates: Sequence[Detection], matched_tracks: set[int]
    ) -> List[_Match]:
        """Second stage: low-confidence boxes for still-unmatched tracks.

        Only confirmed tracks take part - an unconfirmed one propped up by
        low-score boxes is how a false positive would earn an id.
        """
        pairs: List[_Match] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in matched_tracks or track.hits < self.min_hits:
                continue
            predicted = track.predicted_bbox(self._clock)
            for candidate_index, candidate in enumerate(candidates):
                if not self.match_across_labels and candidate.label != track.label:
                    continue
                iou = _overlap(predicted, track.bbox, candidate.bbox)
                if iou >= self.candidate_iou_threshold:
                    pairs.append(_Match(iou, track_index, candidate_index))

        pairs.sort(key=lambda m: (-m.iou, m.track_index, m.detection_index))

        matches: List[_Match] = []
        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        for pair in pairs:
            if pair.track_index in used_tracks or pair.detection_index in used_candidates:
                continue
            matches.append(pair)
            used_tracks.add(pair.track_index)
            used_candidates.add(pair.detection_index)
        return matches

    def _spawn(self, detection: Detection) -> None:
        # With compact ids an unconfirmed track carries id 0 (never reported)
        # until _match() confirms it.
        deferred = self.compact_ids and self.min_hits > 1
        self._tracks.append(
            TrackState(
                track_id=0 if deferred else self._issue_id(),
                label=detection.label,
                bbox=detection.bbox,
                confidence=detection.confidence,
                last_seen=self._clock,
            )
        )

    def _retire_stale(self) -> None:
        self._tracks = [
            track for track in self._tracks if track.time_since_update <= self.max_age
        ]

    def coasting(self, within: float) -> List[TrackState]:
        """Confirmed tracks unmatched this frame but seen within ``within`` clock units.

        Not part of :meth:`update`'s result: a coasting box is a guess, so it
        is for drawing continuity only, never for counting or analysis.
        """
        return sorted(
            (
                track
                for track in self._tracks
                if track.hits >= self.min_hits
                and track.track_id
                and track.time_since_update > 0
                and self._clock - track.last_seen <= within
            ),
            key=lambda track: track.track_id,
        )

    def _reportable(self) -> List[TrackState]:
        reportable = [
            track
            for track in self._tracks
            if track.hits >= self.min_hits
            and (self.publish_coasting or track.time_since_update == 0)
        ]
        reportable.sort(key=lambda track: track.track_id)
        return reportable
