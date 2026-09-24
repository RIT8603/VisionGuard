"""Tracking modules.

Tracking consumes detections from the context and assigns persistent IDs. It
never runs a detector itself. Planned contents (see ROADMAP.md):

    person.py  -> PersonTrackingModule  (Phase 3, requires person_detection)
    object.py  -> ObjectTrackingModule  (Phase 8, requires object_detection)
    vehicle.py -> VehicleTrackingModule (Phase 8, requires vehicle_detection)

Empty in Phase 1.
"""
