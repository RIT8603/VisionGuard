"""Weapon Detection module.

One responsibility: detect weapon-like objects in the frame and raise an
event when one is seen in most of the recent frames.

Out of the box it uses the same COCO model as the other detectors, which
knows three relevant classes: **knife**, **baseball bat** and **scissors**.
That model has never seen a firearm - COCO has no gun class - so detecting
guns needs a custom-trained model: set ``model`` and ``labels`` to match it,
exactly as for PPE or fire.

Small, partly hidden objects are the hardest thing a general detector does.
Expect misses (a knife held low, at distance) and false alarms (a phone, an
umbrella handle). This is a prompt for a person to look, not a verdict - which
is why each event names who is holding the object (when a person track is
near it) and can save a picture for that person to look at.

**Armed-person tracking.** A weapon is often visible for a moment and then
hidden (pocketed, held low, behind the body). Once a weapon has been near the
same person track in most recent frames, that person is marked *armed* and
stays marked - on their track, in red on the video, with their name when face
recognition knows them - for ``armed_hold_seconds`` after the weapon was last
seen, even while it is out of sight. Each person raises ``armed_person`` once,
not again every cooldown, plus once more if they are only identified later.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2

from computer_vision.core.context import BBox, Detection, Event, FrameContext, Severity, Track
from computer_vision.inference.base import BackendError, DetectorBackend
from computer_vision.inference.factory import create_detector
from computer_vision.modules.analysis.common import make_event, positive_number
from computer_vision.modules.base import ModuleInitializationError, ModuleResult
from computer_vision.modules.detection.alerting import PersistenceAlert
from computer_vision.modules.detection.base import LabelFilteredDetectionModule

DEFAULT_WEAPON_LABELS = ("knife", "baseball bat", "scissors")
EVENT_WEAPON = "weapon_detected"
EVENT_ARMED = "armed_person"
PERSON_TRACKER = "person_tracking"
#: ``context.metadata`` key: ``{track_id: {weapon, confidence, armed_seconds, identity}}``.
ARMED_PERSONS_KEY = "armed_persons"
#: A weapon this far from a person's box (in units of that person's height) is
#: not attributed to them.
_MAX_PERSON_DISTANCE = 0.5
#: Boxes from different models overlapping this much are one object.
_MERGE_IOU = 0.5


class _ExtraModel:
    """An extra detector; optional person crops keep small held weapons in view.

    ``person_crops`` defaults off. Its height floor and crop limit apply only
    to that extra model, leaving the other detectors' behaviour unchanged.
    """

    def __init__(self, owner: str, spec: Dict[str, Any], defaults: Dict[str, Any], offset: int = 0) -> None:
        if not spec.get("model"):
            raise ModuleInitializationError(f"{owner}.models: every entry needs a `model` path")
        self.labels = tuple(str(label) for label in (spec.get("labels") or ()))
        if not self.labels:
            raise ModuleInitializationError(f"{owner}.models[{spec['model']}]: set `labels` (the model's class names)")
        self.label_map = {str(k): str(v) for k, v in (spec.get("label_map") or {}).items()}
        self.min_confidence = float(spec.get("confidence_threshold", defaults.get("confidence_threshold", 0.5)))
        self.person_crops = bool(spec.get("person_crops", False))
        self.min_person_height = float(spec.get("min_person_height", 160))
        self.max_person_crops = max(1, int(spec.get("max_person_crops", 4)))
        # Run every Nth frame and reuse the result in between (measured: two
        # extra models on every frame halved the pipeline's FPS). Models are
        # staggered so they do not all run on the same frame.
        self.interval = max(1, int(spec.get("interval_frames", defaults.get("models_interval_frames", 2))))
        self._offset = offset % self.interval
        self._frames = 0
        self._last: List[Detection] = []
        params = {
            key: spec.get(key, defaults.get(key))
            for key in ("backend", "device", "half", "imgsz", "model_sha256")
            if spec.get(key, defaults.get(key)) is not None
        }
        params.update(model=spec["model"], model_confidence=min(self.min_confidence, 0.25), share_model=False)
        if "model_sha256" not in spec:
            params.pop("model_sha256", None)  # the main model's digest is not this file's
        self.name = str(spec["model"])
        try:
            self.backend: DetectorBackend = create_detector(str(params.pop("backend", "ultralytics")), params)
            self.backend.load()
        except BackendError as exc:
            raise ModuleInitializationError(f"{owner}: could not load {self.name}: {exc}") from exc
        available = self.backend.class_names
        missing = [label for label in self.labels if available is not None and label not in available]
        if missing:
            self.backend.unload()
            raise ModuleInitializationError(
                f"{owner}: {self.name} has no class named {missing}. Its classes: {sorted(available or ())}"
            )

    def detect(self, context: FrameContext, module_name: str) -> List[Detection]:
        due = self._frames % self.interval == self._offset
        self._frames += 1
        if not due:
            return list(self._last)
        out: List[Detection] = []
        regions = [(context.frame, 0, 0)]
        if self.person_crops:
            regions = []
            people = sorted(context.get_detections("person_detection"), key=lambda d: d.bbox.area, reverse=True)
            for person in people[:self.max_person_crops]:
                box = person.bbox
                if person.label != "person" or box.height < self.min_person_height:
                    continue
                # Include the hands and anything held beside the body. Coordinates
                # remain in the processing frame, including the crop offsets.
                x1 = max(0, int(box.x1 - 0.35 * box.width))
                y1 = max(0, int(box.y1 - 0.05 * box.height))
                x2 = min(context.width, int(box.x2 + 0.35 * box.width))
                y2 = min(context.height, int(box.y2 + 0.05 * box.height))
                if x2 > x1 and y2 > y1:
                    regions.append((context.frame[y1:y2, x1:x2], x1, y1))
        for frame, ox, oy in regions:
            for raw in self.backend.infer(frame):
                if raw.label not in self.labels or raw.confidence < self.min_confidence:
                    continue
                x1, y1, x2, y2 = raw.xyxy
                bbox = BBox(x1 + ox, y1 + oy, x2 + ox, y2 + oy).clip(context.width, context.height)
                if bbox.width <= 0 or bbox.height <= 0:
                    continue
                out.append(Detection(
                    label=self.label_map.get(raw.label, raw.label), confidence=raw.confidence, bbox=bbox,
                    class_id=raw.class_id, source_module=module_name, extra={"model": self.name},
                ))
        if self.person_crops:
            out = merge_detections([out])
        self._last = out
        return out

    def forget(self) -> None:
        self._last = []


def merge_detections(groups: Sequence[Sequence[Detection]]) -> List[Detection]:
    """One box per object across models: the most confident of overlapping boxes wins."""
    merged: List[Detection] = []
    for det in sorted((d for group in groups for d in group), key=lambda d: d.confidence, reverse=True):
        if all(det.bbox.iou(kept.bbox) < _MERGE_IOU for kept in merged):
            merged.append(det)
    return merged


def _distance_to_box(point: Tuple[float, float], box: BBox) -> float:
    """Distance from a point to a box (0 inside it)."""
    dx = max(box.x1 - point[0], 0.0, point[0] - box.x2)
    dy = max(box.y1 - point[1], 0.0, point[1] - box.y2)
    return math.hypot(dx, dy)


def nearest_person(weapon: BBox, persons: Sequence[Track]) -> Tuple[Optional[Track], float]:
    """The person most plausibly holding ``weapon``, and the distance in person-heights.

    A weapon inside several body boxes (people overlapping) goes to the
    smallest - the one it is actually on. Otherwise the closest box wins,
    provided it is within half a person-height.
    """
    centre = weapon.center
    best: Optional[Track] = None
    best_key: Tuple[float, float] = (math.inf, math.inf)
    for track in persons:
        height = max(track.bbox.height, 1.0)
        distance = _distance_to_box(centre, track.bbox) / height
        if distance > _MAX_PERSON_DISTANCE:
            continue
        key = (distance, track.bbox.area)
        if key < best_key:
            best, best_key = track, key
    return best, (best_key[0] if best is not None else math.inf)


class _ArmedPerson:
    """What is known about one person track that has been near a weapon."""

    __slots__ = ("seen", "weapon", "confidence", "armed_since", "weapon_seen_at",
                 "track_seen_at", "reported_identity")

    def __init__(self, window: int, now: float) -> None:
        #: Was a weapon near this person, for each recent frame their track was seen.
        self.seen: Deque[bool] = deque(maxlen=window)
        self.weapon = ""
        self.confidence = 0.0
        self.armed_since: Optional[float] = None
        self.weapon_seen_at = now
        self.track_seen_at = now
        #: The name already sent in an ``armed_person`` event ("" = none yet).
        self.reported_identity = ""

    @property
    def armed(self) -> bool:
        return self.armed_since is not None


class WeaponDetectionModule(LabelFilteredDetectionModule):
    """Detects weapon-like objects and emits ``weapon_detected``.

    Config block::

        weapon_detection:
          enabled: true
          model: <same as the other detectors, or a custom weapon model>
          labels: [knife, baseball bat, scissors]
          confidence_threshold: 0.4
          min_frames: 3          # weapon_detected once seen in 3...
          window_frames: 5       # ...of the last 5 frames
          cooldown_seconds: 30   # quiet period after an event
          severity: critical
          link_person: true      # attribute it to the nearest person track
          save_snapshots: true   # crop + annotated frame into snapshot_dir
          snapshot_dir: data/snapshots
          models:                # optional extra detectors, e.g. a firearm model
            - model: computer_vision/Model/weapons/firearm.pt
              labels: ["0"]                # that model's own class names
              label_map: {"0": firearm}    # name published in boxes and events
              confidence_threshold: 0.8
              interval_frames: 2           # run every 2nd frame, reuse the result between
          only_with_people: true # extra models skip frames where person detection found nobody
          track_armed: true        # mark the person holding it as armed (needs person_tracking)
          armed_hold_seconds: 60   # stay armed this long after the weapon was last seen near them
          armed_forget_seconds: 5  # forget a person whose track has been gone this long

    With ``link_person`` the event's subject is that person's ``track_id``,
    plus ``identity`` when face recognition has named them. ``track_armed``
    uses the same ``min_frames`` / ``window_frames`` rule, counted per person.
    """

    module_name = "weapon_detection"
    version = "1.1.0"
    requires = ()
    #: Use this frame's person tracks and recognised names when they are on.
    run_after = (PERSON_TRACKER, "face_recognition")

    default_labels = DEFAULT_WEAPON_LABELS

    @classmethod
    def required_modules(cls, params: Dict[str, Any] | None = None) -> Tuple[str, ...]:
        return ("person_detection",) if any(
            model.get("person_crops") for model in ((params or {}).get("models") or ())
        ) else ()

    def initialize(self) -> None:
        self._alert = PersistenceAlert(
            self.module_name, EVENT_WEAPON, self.params,
            default_min_frames=3, default_window=5,
            default_cooldown=30.0, default_severity=Severity.CRITICAL,
        )
        self._link_person = bool(self.params.get("link_person", True))
        self._save = bool(self.params.get("save_snapshots", False))
        self._snapshot_dir = Path(str(self.params.get("snapshot_dir", "data/snapshots")))
        self._only_with_people = bool(self.params.get("only_with_people", True))
        self._track_armed = bool(self.params.get("track_armed", True))
        self._armed_hold = positive_number(self.params, "armed_hold_seconds", 60.0)
        self._armed_forget = positive_number(self.params, "armed_forget_seconds", 5.0)
        self._people: Dict[int, _ArmedPerson] = {}
        self._armed_total = 0
        super().initialize()
        self._extra: List[_ExtraModel] = []
        try:
            for index, spec in enumerate(self.params.get("models") or ()):
                self._extra.append(_ExtraModel(self.module_name, dict(spec), self.params, offset=index))
        except ModuleInitializationError:
            self.cleanup()
            raise
        if self._extra:
            self.log.info("Extra weapon models: %s", ", ".join(
                f"{m.name} -> {sorted({m.label_map.get(l, l) for l in m.labels})}" for m in self._extra))

    def process(self, context: FrameContext) -> ModuleResult:
        result = super().process(context)
        detections = context.get_detections(self.module_name)
        if self._extra and self._people_in_view(context):
            detections = merge_detections(
                [detections, *(model.detect(context, self.module_name) for model in self._extra)]
            )
            context.detections[self.module_name] = detections
            result.detections = [det.to_dict() for det in detections]
        event = self._alert.update(context, detections)
        if event is not None:
            best = max(detections, key=lambda det: det.confidence)
            event.data["count"] = event.subject.pop("count", len(detections))
            holder = self._attribute(event, best, context) if self._link_person else None
            self._attach_snapshot(event, best, holder, context)
            context.add_event(event)
            who = f", held by #{holder.track_id}" if holder is not None else ""
            if event.subject.get("identity"):
                who += f" ({event.subject['identity']})"
            self.log.warning("%s (%s, %.2f%s)", event.event_type, best.label, best.confidence, who)
            result.data["events"] = [event.to_dict()]

        if self._track_armed:
            armed_events = self._track_armed_people(context, detections)
            if armed_events:
                result.data.setdefault("events", []).extend(e.to_dict() for e in armed_events)
            result.data["armed"] = sum(1 for p in self._people.values() if p.armed)
            result.data["armed_total"] = self._armed_total

        result.data["event_total"] = self._alert.total
        return result

    # ------------------------------------------------------------------ #

    def _track_armed_people(self, context: FrameContext, weapons: Sequence[Detection]) -> List[Event]:
        """Update who is armed from this frame's weapons; return the events due."""
        now = context.monotonic_ts
        persons = context.get_tracks(PERSON_TRACKER)

        # The most confident weapon near each person this frame.
        holding: Dict[int, Detection] = {}
        for weapon in weapons:
            holder, _ = nearest_person(weapon.bbox, persons)
            if holder is None:
                continue
            current = holding.get(holder.track_id)
            if current is None or weapon.confidence > current.confidence:
                holding[holder.track_id] = weapon

        events: List[Event] = []
        armed_now: Dict[int, Dict[str, Any]] = {}
        for track in persons:
            person = self._people.get(track.track_id)
            weapon = holding.get(track.track_id)
            if person is None:
                if weapon is None:
                    track.extra.pop("armed", None)
                    track.extra.pop("weapon", None)
                    continue  # never near a weapon: nothing to remember
                person = self._people[track.track_id] = _ArmedPerson(self._alert.window, now)

            person.track_seen_at = now
            person.seen.append(weapon is not None)
            if weapon is not None:
                person.weapon_seen_at = now
                if not person.armed or weapon.confidence >= person.confidence:
                    person.weapon, person.confidence = weapon.label, float(weapon.confidence)

            if person.armed and now - person.weapon_seen_at > self._armed_hold:
                self.log.info("Person #%d no longer marked armed (no weapon seen for %.0fs)",
                              track.track_id, now - person.weapon_seen_at)
                person.armed_since, person.reported_identity = None, ""
                person.seen.clear()

            identity = str((track.extra or {}).get("identity") or "")
            if not person.armed and sum(person.seen) >= self._alert.min_frames:
                person.armed_since = now
                self._armed_total += 1
                events.append(self._armed_event(context, track, person, identity, weapon, first=True))
            elif person.armed and identity and identity != person.reported_identity:
                # Armed first, recognised later: say who it was.
                events.append(self._armed_event(context, track, person, identity, weapon, first=False))

            if person.armed:
                track.extra["armed"] = True
                track.extra["weapon"] = person.weapon
                armed_now[track.track_id] = {
                    "weapon": person.weapon,
                    "confidence": round(person.confidence, 3),
                    "armed_seconds": round(now - float(person.armed_since), 1),
                    "weapon_visible": weapon is not None,
                    "identity": identity or None,
                }
            else:
                track.extra.pop("armed", None)
                track.extra.pop("weapon", None)

        # Forget people whose track has been gone for a while (left the scene).
        for track_id in [tid for tid, p in self._people.items() if now - p.track_seen_at > self._armed_forget]:
            del self._people[track_id]

        if armed_now:
            context.metadata.setdefault(ARMED_PERSONS_KEY, {}).update(armed_now)
        for event in events:
            context.add_event(event)
        return events

    def _armed_event(
        self,
        context: FrameContext,
        track: Track,
        person: _ArmedPerson,
        identity: str,
        weapon: Optional[Detection],
        *,
        first: bool,
    ) -> Event:
        subject: Dict[str, Any] = {"track_id": track.track_id, "class": person.weapon}
        if identity:
            subject["identity"] = identity
            person.reported_identity = identity
        event = make_event(
            self.module_name, EVENT_ARMED, context, self._alert.severity,
            subject=subject,
            data={
                "weapon": person.weapon,
                "confidence": round(person.confidence, 3),
                "identified_later": None if first else True,
                "armed_seconds": round(context.monotonic_ts - float(person.armed_since), 1),
            },
        )
        if first and weapon is not None:
            self._attach_snapshot(event, weapon, track, context)
        who = f" ({identity})" if identity else ""
        if first:
            self.log.warning("%s: person #%d%s with %s", EVENT_ARMED, track.track_id, who, person.weapon)
        else:
            self.log.warning("%s: armed person #%d identified as %s", EVENT_ARMED, track.track_id, identity)
        return event

    def _people_in_view(self, context: FrameContext) -> bool:
        """Whether the extra models should look at this frame.

        With ``only_with_people`` (default) they are skipped when person
        detection ran and found nobody: an alert is about someone holding a
        weapon, and an empty corridor costs nothing then. Without a person
        detector in the pipeline they always run.
        """
        if not self._only_with_people or "person_detection" not in context.detections:
            return True
        if context.get_detections("person_detection"):
            return True
        for model in self._extra:
            model.forget()
        return False

    def _attribute(self, event: Event, weapon: Detection, context: FrameContext) -> Optional[Track]:
        holder, distance = nearest_person(weapon.bbox, context.get_tracks(PERSON_TRACKER))
        if holder is None:
            event.data["person"] = "none near"
            return None
        event.subject["track_id"] = holder.track_id
        identity = (holder.extra or {}).get("identity")
        if identity:
            event.subject["identity"] = identity
        event.data["person_distance"] = round(distance, 2)  # 0 = inside their box
        return holder

    def _attach_snapshot(
        self, event: Event, weapon: Detection, holder: Optional[Track], context: FrameContext
    ) -> None:
        """Save the weapon (with its holder) and the annotated frame. Never raises."""
        if not self._save:
            return
        try:
            frame = context.frame
            h, w = frame.shape[:2]
            region = weapon.bbox
            if holder is not None:
                region = BBox(
                    min(region.x1, holder.bbox.x1), min(region.y1, holder.bbox.y1),
                    max(region.x2, holder.bbox.x2), max(region.y2, holder.bbox.y2),
                )
            mx, my = region.width * 0.2 + 10, region.height * 0.2 + 10
            x1, y1 = int(max(0, region.x1 - mx)), int(max(0, region.y1 - my))
            x2, y2 = int(min(w, region.x2 + mx)), int(min(h, region.y2 + my))
            annotated = frame.copy()
            wx1, wy1, wx2, wy2 = weapon.bbox.as_int()
            cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{weapon.label} {weapon.confidence:.2f}", (wx1, max(12, wy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            if holder is not None:
                px1, py1, px2, py2 = holder.bbox.as_int()
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 215, 255), 1)
            crop = annotated[y1:y2, x1:x2]
            if crop.size == 0:
                return
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            camera = re.sub(r"[^A-Za-z0-9_-]", "_", str(context.source_id))[:40] or "camera"
            stem = f"weapon_{camera}_{time.strftime('%Y%m%d-%H%M%S')}_{context.frame_id}"
            crop_name, frame_name = f"{stem}.jpg", f"{stem}_frame.jpg"
            cv2.imwrite(str(self._snapshot_dir / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(self._snapshot_dir / frame_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            event.data["snapshot"], event.data["frame_snapshot"] = crop_name, frame_name
        except Exception as exc:  # noqa: BLE001 - a snapshot must never stop the alert
            self.log.warning("Could not save the weapon snapshot: %s", exc)

    def cleanup(self) -> None:
        if hasattr(self, "_alert"):
            self._alert.reset()
        self._people = {}
        for model in getattr(self, "_extra", ()):
            model.backend.unload()
        self._extra = []
        super().cleanup()
