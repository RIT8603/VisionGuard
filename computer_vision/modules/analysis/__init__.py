"""Higher-level analysis modules.

These consume the results of detection, recognition and tracking modules and
produce **events** - "what happened" rather than "what is in the frame".

They contain the business rules a site cares about. They still obey the
module independence rule (Module_spec.md): an analysis module publishes
structured events and does not notify anyone, write incidents or touch a
database. Those are downstream concerns.

Planned contents (see ROADMAP.md, Phase 9):

    zone.py           -> RestrictedZoneModule   (implemented)
    loitering.py      -> LoiteringModule        (implemented)
    crowd_alert.py    -> CrowdAlertModule       (implemented)
    intrusion.py      -> IntrusionDetectionModule (implemented)
    tailgating.py     -> TailgatingModule       (implemented)
    abandoned_object.py -> AbandonedObjectModule (implemented)
    inventory.py      -> InventoryMonitoringModule (implemented)
    low_light.py      -> LowLightModule         (implemented)
    ppe_violation.py  -> PPEViolationModule     (needs ppe_detection weights)
    attendance.py     -> AttendanceModule       (needs Phase 7 recognition)
    unknown_person.py -> UnknownPersonAlert     (needs Phase 7 recognition)
"""
