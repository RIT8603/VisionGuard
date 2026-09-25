"""Shared per-frame analysis context.

``FrameContext`` is the only channel through which AI modules communicate.
A module reads what earlier modules wrote and writes its own results back:

    PersonDetectionModule  -> context.add_detections("person_detection", [...])
    PersonTrackingModule   -> context.get_detections("person_detection")
                           -> context.add_tracks("person_tracking", [...])

Results are keyed by the producing module name. That is what keeps
``person_detection`` and (later) ``ppe_detection`` from colliding, and lets a
consumer ask for exactly the producer it depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates of ``FrameContext.frame``.

    Coordinates are ``x1, y1`` (top-left) and ``x2, y2`` (bottom-right).
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"Invalid bbox: ({self.x1}, {self.y1}, {self.x2}, {self.y2}) - "
                "x2/y2 must be >= x1/y1"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def as_int(self) -> Tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def as_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def clip(self, width: int, height: int) -> "BBox":
        """Clamp the box to a frame of the given size."""
        return BBox(
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            x2=max(0.0, min(self.x2, width)),
            y2=max(0.0, min(self.y2, height)),
        )

    def iou(self, other: "BBox") -> float:
        """Intersection-over-union with another box."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def from_xyxy(cls, values: Sequence[float]) -> "BBox":
        x1, y1, x2, y2 = values
        return cls(float(x1), float(y1), float(x2), float(y2))

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        return cls(float(x), float(y), float(x + w), float(y + h))


@dataclass
class Detection:
    """A single detection produced by a detection module."""

    label: str
    confidence: float
    bbox: BBox
    class_id: int | None = None
    source_module: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Standardised detection schema (see PROJECT_SPEC / Agent.md)."""
        payload: Dict[str, Any] = {
            "class": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(v, 2) for v in self.bbox.as_list()],
        }
        if self.class_id is not None:
            payload["class_id"] = self.class_id
        if self.extra:
            payload["extra"] = self.extra
        return payload


@dataclass
class Track:
    """A detection associated with a persistent identity across frames."""

    track_id: int
    label: str
    bbox: BBox
    confidence: float = 0.0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    source_module: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Standardised track schema (see PROJECT_SPEC / Agent.md)."""
        payload: Dict[str, Any] = {
            "track_id": self.track_id,
            "class": self.label,
            "bbox": [round(v, 2) for v in self.bbox.as_list()],
            "confidence": round(float(self.confidence), 4),
            "age": self.age,
            "hits": self.hits,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


#: COCO's 17 body keypoints, in the order every pose model emits them.
KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


@dataclass(frozen=True)
class Keypoint:
    """One body joint in frame pixel coordinates."""

    name: str
    x: float
    y: float
    confidence: float = 0.0

    @property
    def is_visible(self) -> bool:
        """Models emit occluded joints at near-zero confidence, not absent."""
        return self.confidence >= 0.3

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass
class Pose:
    """A person's skeleton for one frame.

    Produced by a pose module; consumed by posture analysis such as fall
    detection. Keeping it in its own lane (rather than stuffing joints into a
    ``Detection``) means a consumer asks for poses and gets poses.
    """

    bbox: BBox
    keypoints: List[Keypoint] = field(default_factory=list)
    confidence: float = 0.0
    source_module: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def keypoint(self, name: str) -> Optional[Keypoint]:
        for point in self.keypoints:
            if point.name == name:
                return point
        return None

    def visible(self, *names: str) -> List[Keypoint]:
        """Only the named joints the model was actually confident about."""
        found = [self.keypoint(name) for name in names]
        return [point for point in found if point is not None and point.is_visible]

    def midpoint(self, left: str, right: str) -> Optional[Tuple[float, float]]:
        """Centre of a symmetric pair, using whichever side is visible."""
        points = self.visible(left, right)
        if not points:
            return None
        return (
            sum(p.x for p in points) / len(points),
            sum(p.y for p in points) / len(points),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "bbox": [round(v, 2) for v in self.bbox.as_list()],
            "confidence": round(float(self.confidence), 4),
            "keypoints": {
                point.name: [round(point.x, 1), round(point.y, 1), round(point.confidence, 3)]
                for point in self.keypoints
                if point.is_visible
            },
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


class Severity(str, Enum):
    """How much attention an event deserves.

    The platform never decides what to *do* about an event - notifying,
    escalating and recording are downstream concerns (Agent.md §16). This is
    only a label the producing module attaches.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Event:
    """Something a higher-level module concluded from lower-level results.

    Detections and tracks describe *what is in the frame*. An event describes
    *what happened* - a person crossed into a restricted zone, a vehicle
    lingered too long. Events are the output of analysis modules, and the
    dashboard and alerting layers will consume them.
    """

    event_type: str
    source_module: str
    frame_id: int
    timestamp: str
    severity: Severity = Severity.INFO
    #: What the event is about, e.g. ``{"track_id": 3, "class": "person"}``.
    subject: Dict[str, Any] = field(default_factory=dict)
    #: Event-specific detail, e.g. zone name and dwell duration.
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "event": self.event_type,
            "module": self.source_module,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
        }
        if self.subject:
            payload["subject"] = self.subject
        if self.data:
            payload["data"] = self.data
        return payload

    def describe(self) -> str:
        """Short human-readable line, for logs and the HUD."""
        who = self.subject.get("class", "")
        track_id = self.subject.get("track_id")
        subject = f"{who} #{track_id}" if track_id is not None else who
        zone = self.data.get("zone")
        where = f" in '{zone}'" if zone else ""
        return f"{self.event_type}: {subject}{where}".strip()


@dataclass
class Overlay:
    """A shape a module wants drawn on the output frame.

    Lets a module contribute to the rendering without the output layer knowing
    what a zone (or anything else) is. Points are in frame pixel coordinates.
    """

    kind: str
    points: List[Tuple[int, int]]
    label: str = ""
    color: Tuple[int, int, int] = (0, 210, 255)
    active: bool = False
    filled: bool = False


@dataclass
class FrameContext:
    """Everything known about one frame as it flows through the pipeline.

    The pipeline creates one instance per frame and passes it to each enabled
    module in dependency order. Modules mutate it; the core never inspects
    module-specific fields.
    """

    frame: np.ndarray
    frame_id: int
    source_id: str
    monotonic_ts: float
    wall_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    detections: Dict[str, List[Detection]] = field(default_factory=dict)
    tracks: Dict[str, List[Track]] = field(default_factory=dict)
    poses: Dict[str, List[Pose]] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    overlays: List[Overlay] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    module_results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    # ---------------- frame geometry ---------------- #

    @property
    def height(self) -> int:
        return int(self.frame.shape[0])

    @property
    def width(self) -> int:
        return int(self.frame.shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.width, self.height

    @property
    def timestamp_iso(self) -> str:
        return self.wall_time.isoformat()

    # ---------------- detections ---------------- #

    def add_detections(self, module_name: str, detections: Iterable[Detection]) -> None:
        """Publish detections under the producing module's name."""
        items = list(detections)
        for det in items:
            if not det.source_module:
                det.source_module = module_name
        self.detections.setdefault(module_name, []).extend(items)

    def get_detections(self, module_name: str | None = None) -> List[Detection]:
        """Detections from one module, or all detections when ``module_name`` is None."""
        if module_name is not None:
            return list(self.detections.get(module_name, []))
        return [det for items in self.detections.values() for det in items]

    def has_detections(self, module_name: str) -> bool:
        return bool(self.detections.get(module_name))

    # ---------------- tracks ---------------- #

    def add_tracks(self, module_name: str, tracks: Iterable[Track]) -> None:
        """Publish tracks under the producing module's name."""
        items = list(tracks)
        for trk in items:
            if not trk.source_module:
                trk.source_module = module_name
        self.tracks.setdefault(module_name, []).extend(items)

    def get_tracks(self, module_name: str | None = None) -> List[Track]:
        if module_name is not None:
            return list(self.tracks.get(module_name, []))
        return [trk for items in self.tracks.values() for trk in items]

    # ---------------- poses ---------------- #

    def add_poses(self, module_name: str, poses: Iterable[Pose]) -> None:
        """Publish poses under the producing module's name."""
        items = list(poses)
        for pose in items:
            if not pose.source_module:
                pose.source_module = module_name
        self.poses.setdefault(module_name, []).extend(items)

    def get_poses(self, module_name: str | None = None) -> List[Pose]:
        if module_name is not None:
            return list(self.poses.get(module_name, []))
        return [pose for items in self.poses.values() for pose in items]

    # ---------------- events ---------------- #

    def add_event(self, event: Event) -> None:
        """Publish an event. Events are append-only within a frame."""
        self.events.append(event)

    def add_events(self, events: Iterable[Event]) -> None:
        self.events.extend(events)

    def get_events(self, module_name: str | None = None) -> List[Event]:
        if module_name is None:
            return list(self.events)
        return [e for e in self.events if e.source_module == module_name]

    # ---------------- overlays ---------------- #

    def add_overlay(self, overlay: Overlay) -> None:
        """Ask the output layer to draw a shape."""
        self.overlays.append(overlay)

    # ---------------- module bookkeeping ---------------- #

    def set_result(self, module_name: str, result: Any) -> None:
        self.module_results[module_name] = result

    def get_result(self, module_name: str) -> Any | None:
        return self.module_results.get(module_name)

    def record_error(self, module_name: str, message: str) -> None:
        self.errors[module_name] = message

    # ---------------- serialisation ---------------- #

    def summary(self) -> Dict[str, Any]:
        """JSON-safe snapshot. Never includes pixel data."""
        return {
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "timestamp": self.timestamp_iso,
            "resolution": {"width": self.width, "height": self.height},
            "detections": {
                name: [det.to_dict() for det in items]
                for name, items in self.detections.items()
            },
            "tracks": {
                name: [trk.to_dict() for trk in items]
                for name, items in self.tracks.items()
            },
            "poses": {
                name: [pose.to_dict() for pose in items]
                for name, items in self.poses.items()
            },
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
            "errors": self.errors,
        }
