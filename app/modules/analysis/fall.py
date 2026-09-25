"""Fall Detection module.

One responsibility: report when someone appears to have fallen.

It runs no model. It reads poses that ``pose_detection`` already produced and
applies geometry to them.

**How a fall is judged.** Three independent signals, combined so no single
one can raise an alarm alone:

1. **Box orientation** - a standing person's box is tall; a lying person's is
   wide. Cheap and robust, but a person crouching or a camera looking straight
   down fools it.
2. **Torso angle** - the shoulder-to-hip line is near vertical when upright
   and near horizontal when prone. Needs those four joints visible.
3. **Body height collapse** - head-to-ankle vertical extent shrinks sharply
   relative to the person's own width.

Each contributes a score; a fall needs the combined score over a threshold
for several consecutive frames. That last part matters more than the geometry:
a single frame of someone bending to tie a shoelace looks exactly like the
start of a fall, and a system that cries wolf gets switched off.

**This is a heuristic, not a medical device.** It will miss falls and it will
raise false alarms. Tune ``min_confidence`` and ``min_frames`` against your own
footage before trusting it, and never make it the only safeguard for someone's
safety.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from app.core.context import BBox, Detection, Event, FrameContext, Pose, Severity
from app.modules.base import BaseAIModule, ModuleResult, ModuleType
from app.modules.tracking.tracker_core import IoUTracker

EVENT_FALL = "fall_detected"
EVENT_RECOVERED = "fall_recovered"

DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_MIN_FRAMES = 5
DEFAULT_COOLDOWN_SEC = 20.0


class FallDetectionModule(BaseAIModule):
    """Emits an event when a tracked person holds a fallen posture.

    Config block::

        fall_detection:
          enabled: true
          min_confidence: 0.6     # combined posture score, 0-1
          min_frames: 5           # consecutive frames before reporting
          cooldown_seconds: 20    # quiet period per person after reporting
          severity: critical
          report_recovery: true   # also emit when they get back up
    """

    module_name = "fall_detection"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = ("pose_detection",)

    source_module = "pose_detection"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._min_confidence = float(self.params.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
        self._min_frames = max(1, int(self.params.get("min_frames", DEFAULT_MIN_FRAMES)))
        self._cooldown = float(self.params.get("cooldown_seconds", DEFAULT_COOLDOWN_SEC))
        self._report_recovery = bool(self.params.get("report_recovery", True))

        # Poses have no identity of their own, so an internal tracker gives
        # one. Reuses the same association code as person_tracking rather
        # than inventing a second, untested matcher.
        self._tracker = IoUTracker(
            iou_threshold=float(self.params.get("iou_threshold", 0.3)),
            max_age=int(self.params.get("max_age", 15)),
            min_hits=1,
        )
        self._streaks: Dict[int, int] = {}
        self._reported: Dict[int, float] = {}
        self._fallen: set[int] = set()
        self._total_falls = 0

        raw_severity = str(self.params.get("severity", Severity.CRITICAL.value)).lower()
        try:
            self._severity = Severity(raw_severity)
        except ValueError:
            valid = [s.value for s in Severity]
            raise ValueError(f"severity must be one of {valid}, got {raw_severity!r}")

    def initialize(self) -> None:
        """Nothing to load. This is geometry, not inference."""
        self.log.info(
            "Ready: consuming '%s', score>=%.2f for %d frame(s), cooldown %.0fs",
            self.source_module,
            self._min_confidence,
            self._min_frames,
            self._cooldown,
        )

    def cleanup(self) -> None:
        self._tracker.reset()
        self._streaks.clear()
        self._fallen.clear()
        super().cleanup()

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def process(self, context: FrameContext) -> ModuleResult:
        poses = context.get_poses(self.source_module)
        now = context.monotonic_ts

        # Give each pose a stable id by tracking its box.
        tracked = self._tracker.update(
            [
                Detection(label="person", confidence=pose.confidence, bbox=pose.bbox)
                for pose in poses
            ]
        )
        by_id = self._align(tracked, poses)

        events: List[Event] = []
        scores: Dict[int, float] = {}

        events.extend(self._retire_vanished(set(by_id), context))

        for track_id, pose in by_id.items():
            score, reasons = fall_score(pose)
            scores[track_id] = round(score, 3)

            if score >= self._min_confidence:
                self._streaks[track_id] = self._streaks.get(track_id, 0) + 1
                if self._streaks[track_id] >= self._min_frames and self._ready(track_id, now):
                    self._reported[track_id] = now
                    self._fallen.add(track_id)
                    self._total_falls += 1
                    events.append(
                        self._event(EVENT_FALL, context, track_id, score, reasons)
                    )
            else:
                self._streaks[track_id] = 0
                if track_id in self._fallen:
                    self._fallen.discard(track_id)
                    if self._report_recovery:
                        events.append(
                            self._event(
                                EVENT_RECOVERED, context, track_id, score, reasons,
                                severity=Severity.INFO,
                            )
                        )

        for event in events:
            context.add_event(event)
            self.log.info("%s", event.describe())

        return self.make_result(
            context,
            data={
                "people": len(by_id),
                "fallen": len(self._fallen),
                "scores": scores,
                "total_falls": self._total_falls,
            },
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _retire_vanished(self, present: set, context: FrameContext) -> List[Event]:
        """Clear state for tracks that disappeared.

        Someone getting up changes their box shape enough that association can
        legitimately lose them. Without this their id stays "fallen" forever,
        the count drifts upward, and no recovery is ever reported.
        """
        events: List[Event] = []

        for track_id in sorted(set(self._streaks) - present):
            self._streaks.pop(track_id, None)
            if track_id in self._fallen:
                self._fallen.discard(track_id)
                if self._report_recovery:
                    event = self._event(
                        EVENT_RECOVERED, context, track_id, 0.0, {},
                        severity=Severity.INFO,
                    )
                    # Distinguishable from actually standing up.
                    event.data["reason"] = "track_lost"
                    events.append(event)
        return events

    @staticmethod
    def _align(tracked, poses: List[Pose]) -> Dict[int, Pose]:
        """Pair each track back to the pose it came from, by best overlap."""
        pairs: Dict[int, Pose] = {}
        for state in tracked:
            best, best_iou = None, 0.0
            for pose in poses:
                overlap = state.bbox.iou(pose.bbox)
                if overlap > best_iou:
                    best, best_iou = pose, overlap
            if best is not None:
                pairs[state.track_id] = best
        return pairs

    def _ready(self, track_id: int, now: float) -> bool:
        last = self._reported.get(track_id)
        return last is None or (now - last) >= self._cooldown

    def _event(
        self,
        event_type: str,
        context: FrameContext,
        track_id: int,
        score: float,
        reasons: Dict[str, float],
        severity: Optional[Severity] = None,
    ) -> Event:
        return Event(
            event_type=event_type,
            source_module=self.module_name,
            frame_id=context.frame_id,
            timestamp=context.timestamp_iso,
            severity=severity or self._severity,
            subject={"track_id": track_id, "class": "person"},
            data={
                "score": round(score, 3),
                # Which signals fired, so a false alarm can be diagnosed
                # instead of just disbelieved.
                "signals": {k: round(v, 3) for k, v in reasons.items()},
                "consecutive_frames": self._streaks.get(track_id, 0),
            },
        )

    @property
    def total_falls(self) -> int:
        return self._total_falls


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def fall_score(pose: Pose) -> Tuple[float, Dict[str, float]]:
    """Score how much a pose looks like a fall, from 0 to 1.

    Returns the score and the individual signals, so a result can be explained.
    Signals that need joints the model could not see are simply left out and
    the remaining ones re-weighted - a partially visible person still gets a
    usable answer rather than a zero.
    """
    signals: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    # 1. Box orientation. Always available.
    if pose.bbox.height > 0:
        ratio = pose.bbox.width / pose.bbox.height
        # ratio 0.5 (upright) -> 0, ratio 1.5 (flat) -> 1
        signals["box_ratio"] = _ramp(ratio, 0.6, 1.4)
        weights["box_ratio"] = 0.4

    # 2. Torso angle from vertical.
    shoulders = pose.midpoint("left_shoulder", "right_shoulder")
    hips = pose.midpoint("left_hip", "right_hip")
    if shoulders and hips:
        dx = hips[0] - shoulders[0]
        dy = hips[1] - shoulders[1]
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            # 0 deg = upright, 90 deg = horizontal
            angle = math.degrees(math.atan2(abs(dx), abs(dy)))
            signals["torso_angle"] = _ramp(angle, 30.0, 70.0)
            weights["torso_angle"] = 0.4

    # 3. Vertical extent collapse, measured against the person's own width so
    #    it does not depend on distance from the camera.
    head = pose.visible("nose", "left_eye", "right_eye", "left_ear", "right_ear")
    feet = pose.visible("left_ankle", "right_ankle")
    if head and feet and pose.bbox.width > 0:
        span = abs(min(p.y for p in feet) - min(p.y for p in head))
        signals["height_collapse"] = 1.0 - _ramp(span / pose.bbox.width, 0.8, 1.8)
        weights["height_collapse"] = 0.2

    if not signals:
        return 0.0, {}

    total_weight = sum(weights.values())
    score = sum(signals[name] * weights[name] for name in signals) / total_weight
    return max(0.0, min(1.0, score)), signals


def _ramp(value: float, low: float, high: float) -> float:
    """0 below ``low``, 1 above ``high``, linear between."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))
