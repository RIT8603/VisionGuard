"""Inventory Monitoring module.

One responsibility: count configured item classes in view and report when a
count leaves its allowed range.

"There should be at least three fire extinguishers on this wall" or "no more
than ten pallets in this bay" are both a class, a count and a bound. This
module counts what ``object_detection`` reports - optionally only inside drawn
zones - and raises ``inventory_low`` / ``inventory_high`` when the count moves
out of range, and ``inventory_restored`` when it comes back.

Counts from a detector flicker: an item is missed for a frame, or seen twice.
The count judged here is the **median over a short window**, so one bad frame
changes nothing.

It counts what the camera can see. Items hidden behind other items are not
counted, so this suits shelves and floor bays viewed face-on, not deep stacks.
The items must be classes the detector knows; for anything outside the 80
COCO classes that means a custom-trained model on ``object_detection``.
"""

from __future__ import annotations

from collections import deque
from statistics import median
from typing import Any, Deque, Dict, Tuple

from computer_vision.core.context import FrameContext, Overlay, Severity
from computer_vision.modules.analysis.common import (
    make_event,
    parse_optional_zones,
    parse_severity,
    positive_number,
    zone_containing,
)
from computer_vision.modules.base import BaseAIModule, ModuleInitializationError, ModuleResult, ModuleType

EVENT_LOW = "inventory_low"
EVENT_HIGH = "inventory_high"
EVENT_RESTORED = "inventory_restored"
_COLOR_ZONE = (200, 160, 60)


class InventoryMonitoringModule(BaseAIModule):
    """Counts items and emits events when a count leaves its range.

    Config block::

        inventory_monitoring:
          enabled: true
          source_module: object_detection
          window_frames: 15        # counts are the median over this many frames
          severity: warning
          items:
            bottle: {min: 3}        # at least 3 bottles
            chair: {min: 2, max: 8}
          zones: []                # optional; empty = the whole frame

    Requires ``object_detection``.
    """

    module_name = "inventory_monitoring"
    module_type = ModuleType.ANALYSIS
    version = "1.0.0"
    requires = ("object_detection",)

    @classmethod
    def required_modules(cls, params: Dict[str, Any] | None = None) -> Tuple[str, ...]:
        return (str((params or {}).get("source_module") or "object_detection"),)

    def initialize(self) -> None:
        self._source = self.required_modules(self.params)[0]
        self._bounds = self._parse_items(self.params.get("items"))
        self._window = int(positive_number(self.params, "window_frames", 15))
        self._severity = parse_severity(self.params, Severity.WARNING)
        self._zones = parse_optional_zones(self.params.get("zones"))
        self._history: Dict[str, Deque[int]] = {
            label: deque(maxlen=self._window) for label in self._bounds
        }
        #: "ok" | "low" | "high" per item, as last reported.
        self._status: Dict[str, str] = {label: "ok" for label in self._bounds}
        self._total = 0
        self.log.info("Ready: counting %s from '%s'", self._bounds, self._source)

    @staticmethod
    def _parse_items(raw: Any) -> Dict[str, Tuple[int | None, int | None]]:
        if not isinstance(raw, dict) or not raw:
            raise ModuleInitializationError(
                "inventory_monitoring needs 'items': a class name and its allowed count, e.g.\n"
                "    items:\n"
                "      bottle: {min: 3}\n"
                "      chair: {min: 2, max: 8}"
            )
        bounds: Dict[str, Tuple[int | None, int | None]] = {}
        for label, rule in raw.items():
            if not isinstance(rule, dict) or not ({"min", "max"} & set(rule)):
                raise ModuleInitializationError(
                    f"inventory_monitoring.items.{label} needs 'min' and/or 'max'"
                )
            low = int(rule["min"]) if rule.get("min") is not None else None
            high = int(rule["max"]) if rule.get("max") is not None else None
            if (low is not None and low < 0) or (high is not None and high < 0):
                raise ModuleInitializationError(f"inventory_monitoring.items.{label}: counts must be >= 0")
            if low is not None and high is not None and low > high:
                raise ModuleInitializationError(f"inventory_monitoring.items.{label}: min is above max")
            bounds[str(label)] = (low, high)
        return bounds

    def process(self, context: FrameContext) -> ModuleResult:
        seen: Dict[str, int] = {label: 0 for label in self._bounds}
        for detection in context.get_detections(self._source):
            if detection.label not in seen:
                continue
            if zone_containing(self._zones, detection.bbox.center, context) is None:
                continue
            seen[detection.label] += 1

        events = []
        stable: Dict[str, int] = {}
        for label, (low, high) in self._bounds.items():
            history = self._history[label]
            history.append(seen[label])
            if len(history) < self._window:
                continue  # not enough evidence yet to call a count

            count = int(median(history))
            stable[label] = count
            status = "low" if low is not None and count < low else (
                "high" if high is not None and count > high else "ok"
            )
            if status == self._status[label]:
                continue
            self._status[label] = status

            event_type = {"low": EVENT_LOW, "high": EVENT_HIGH, "ok": EVENT_RESTORED}[status]
            events.append(
                make_event(
                    self.module_name, event_type, context,
                    Severity.INFO if status == "ok" else self._severity,
                    subject={"class": label, "count": count},
                    data={"count": count, "min": low, "max": high},
                )
            )

        for event in events:
            context.add_event(event)
            self.log.info("%s: %s = %s", event.event_type, event.subject["class"], event.data["count"])
        self._total += len(events)

        for zone in self._zones:
            context.add_overlay(
                Overlay(
                    kind="polygon",
                    points=[(int(x), int(y)) for x, y in zone.to_pixels(context.width, context.height)],
                    label=zone.name,
                    color=_COLOR_ZONE,
                )
            )

        return self.make_result(
            context,
            data={
                "seen": seen,
                "counts": stable,
                "status": dict(self._status),
                "events": [event.to_dict() for event in events],
                "event_total": self._total,
            },
        )

    @property
    def item_status(self) -> Dict[str, str]:
        return dict(self._status)
