"""Explicit opt-in continuity for a stationary bag whose COCO label fluctuates."""
import numpy as np

from computer_vision.core.context import BBox, FrameContext, Track
from computer_vision.modules.analysis.abandoned_object import AbandonedObjectModule


def frame(number, seconds, label='backpack', identity=1):
    context = FrameContext(np.zeros((480, 640, 3), np.uint8), number, 'test', seconds)
    context.add_tracks('object_tracking', [Track(identity, label, BBox(200, 200, 250, 260), .9, source_module='object_tracking')])
    return context


def test_configured_bag_label_change_keeps_stationary_clock():
    module = AbandonedObjectModule({'seconds': 2, 'merge_bag_labels': True})
    module.initialize()
    module.process(frame(1, 0))
    context = frame(2, 3, 'suitcase', 2)
    module.process(context)
    assert [e.event_type for e in context.events] == ['abandoned_object']


def test_default_does_not_merge_different_classes():
    module = AbandonedObjectModule({'seconds': 2})
    module.initialize()
    module.process(frame(1, 0))
    context = frame(2, 3, 'suitcase', 2)
    module.process(context)
    assert not context.events


def test_bag_group_never_merges_a_different_kind_of_object():
    module = AbandonedObjectModule({'seconds': 2, 'labels': ['backpack', 'cell phone'], 'merge_bag_labels': True})
    module.initialize()
    module.process(frame(1, 0))
    context = frame(2, 3, 'cell phone', 2)
    module.process(context)
    assert not context.events
