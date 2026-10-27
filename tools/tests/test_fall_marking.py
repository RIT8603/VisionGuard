"""Fall display state follows confirmed poses, independently of event cooldown."""
import cv2
import numpy as np

from backend.services.state import DashboardState
from backend.services.web_sink import WebSink, _Results
from computer_vision.core.config import ProcessingConfig
from computer_vision.core.context import BBox, Detection, Keypoint, Overlay, Pose, Track
from computer_vision.utils.annotate import draw_overlays
from tools.tests.test_pose_fall_attendance import build_fall, lying_pose, pose_ctx, standing_pose


def markings(context):
    return [o for o in context.overlays if o.label == 'FALL DETECTED']


def test_only_confirmed_falls_are_marked_and_cooldown_does_not_hide_them():
    module = build_fall(min_frames=2)
    first = pose_ctx([lying_pose()], 1, ts=0)
    module.process(first)
    assert not markings(first)
    second = pose_ctx([lying_pose()], 2, ts=.1)
    module.process(second)
    assert len(markings(second)) == 1
    third = pose_ctx([lying_pose()], 3, ts=.2)
    module.process(third)
    assert not third.events  # the alarm is on cooldown, the marking stays visible
    assert len(markings(third)) == 1


def test_marking_clears_on_recovery_or_pose_loss():
    for poses in [[standing_pose()], []]:
        module = build_fall(min_frames=1)
        module.process(pose_ctx([lying_pose()], 1, ts=0))
        context = pose_ctx(poses, 2, ts=1)
        module.process(context)
        assert not markings(context)


def test_dashboard_jpeg_keeps_red_marking_visible_above_normal_boxes():
    module = build_fall(min_frames=1)
    context = pose_ctx([lying_pose()], 1, ts=0)
    context.add_detections('pose_detection', [Detection('person', .9, BBox(100, 350, 400, 420))])
    context.add_tracks('person_tracking', [Track(99, 'person', BBox(100, 330, 400, 420), .9)])
    original = context.frame.copy()
    module.process(context)
    expected = draw_overlays(context.frame.copy(), context.overlays)[300:326, 94:310].astype(np.int16)
    caption_red = (expected[:, :, 2] > 180) & (expected[:, :, 2] > expected[:, :, 0] + 100)
    sink = WebSink(DashboardState(), ProcessingConfig(target_width=None), stream_fps=0)
    try:
        # Both file-video rendering and the live-view renderer use the overlay.
        encoded = [sink._encode(context), sink._draw(context.frame.copy(), _Results(
            overlays=context.overlays, detections=context.get_detections(), tracks=context.get_tracks()
        ), context.get_tracks(), fresh=True)]
        for jpeg in encoded:
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            # Caption region outside the normal body/pose box remains red.
            region = frame[300:326, 94:310].astype(np.int16)
            assert ((region[:, :, 2] > 120) & (region[:, :, 2] > region[:, :, 0] + 60)).sum() > 150
            # Even a higher person caption cannot cover the alert caption.
            assert (region[:, :, 2] > region[:, :, 0] + 60)[caption_red].mean() > .9
        assert np.array_equal(context.frame, original)
    finally:
        sink.close()


def test_stale_live_results_hide_alert_box_but_keep_static_zone():
    module = build_fall(min_frames=1)
    context = pose_ctx([lying_pose()], 1, ts=0)
    module.process(context)
    zone = Overlay('polygon', [(10, 10), (60, 10), (60, 60), (10, 60)], color=(0, 220, 0))
    sink = WebSink(DashboardState(), ProcessingConfig(target_width=None), stream_fps=0)
    try:
        jpeg = sink._draw(context.frame.copy(), _Results(overlays=[*context.overlays, zone]), [], fresh=False)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image[300:430, 90:410].max() < 15
        assert image[8:13, 20:50, 1].max() > 100
    finally:
        sink.close()


def test_monitoring_keeps_marking_when_fall_score_drops_but_body_is_still_flat():
    module = build_fall(min_frames=1, monitor_until_upright=True)
    module.process(pose_ctx([lying_pose()], 1, ts=0))
    partial = Pose(BBox(100, 350, 400, 420), [
        Keypoint('left_shoulder', 150, 365, .9), Keypoint('right_shoulder', 180, 365, .9),
        Keypoint('left_hip', 150, 395, .9), Keypoint('right_hip', 180, 395, .9),
    ], .8)
    for number in range(2, 8):
        context = pose_ctx([partial], number, ts=number * .1)
        result = module.process(context)
        assert max(result.data['scores'].values()) < .6
        assert result.data['fallen'] == 1
        assert not context.events
        assert markings(context)
    assert module.total_falls == 1


def test_visible_upright_confirmation_is_required_to_end_monitoring():
    module = build_fall(min_frames=1, monitor_until_upright=True, recovery_confirm_frames=3)
    fallen = Pose(BBox(100, 300, 220, 420), [
        Keypoint('left_shoulder', 120, 350, .9), Keypoint('right_shoulder', 120, 360, .9),
        Keypoint('left_hip', 205, 350, .9), Keypoint('right_hip', 205, 360, .9),
    ], .9)
    upright = Pose(BBox(110, 220, 200, 420), [
        Keypoint('left_shoulder', 125, 270, .9), Keypoint('right_shoulder', 175, 270, .9),
        Keypoint('left_hip', 125, 340, .9), Keypoint('right_hip', 175, 340, .9),
    ], .9)
    module.process(pose_ctx([fallen], 1, ts=0))
    assert module._fallen
    for number in [2, 3]:
        context = pose_ctx([upright], number, ts=number * .1)
        module.process(context)
        assert markings(context) and not context.events
    context = pose_ctx([upright], 4, ts=.4)
    module.process(context)
    assert not markings(context)
    assert [e.event_type for e in context.events] == ['fall_recovered']


def test_brief_pose_loss_remembers_incident_without_drawing_a_stale_box():
    module = build_fall(min_frames=1, monitor_until_upright=True, max_age=2)
    module.process(pose_ctx([lying_pose()], 1, ts=0))
    missing = pose_ctx([], 2, ts=.1)
    assert module.process(missing).data['fallen'] == 1
    assert not missing.events and not markings(missing)
    returned = pose_ctx([lying_pose()], 3, ts=.2)
    module.process(returned)
    assert markings(returned) and not returned.events
    for number in [4, 5, 6]:
        missing = pose_ctx([], number, ts=number * .1)
        module.process(missing)
    assert not module._fallen and not missing.events
