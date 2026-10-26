"""Metric correctness, missing-ground-truth handling and media-clock regression."""
from datetime import datetime, timezone

import numpy as np
import pytest

from computer_vision.core.config import VideoConfig
from computer_vision.sources.base import FramePacket
from computer_vision.sources.video_file import VideoFileSource
from tools.benchmark_cv import MediaTimedSource, enabled_config
from tools.cv_metrics import match_boxes, rates, score_events, score_frames, timing


def box(label='person', track_id=1):
    return {'label': label, 'bbox': [0, 0, 20, 40], 'track_id': track_id}


def test_wrong_class_is_both_false_positive_and_miss():
    score = score_frames([{'frame': 1, 'objects': [box()]}], {1: [box('dog')]})
    assert (score['tp'], score['fp'], score['fn']) == (0, 1, 1)
    assert score['precision'] == score['recall'] == 0


def test_unlabelled_predictions_do_not_become_false_positives():
    score = score_frames([{'frame': 1, 'objects': [box()]}], {1: [box()], 2: [box(), box()]})
    assert score['fp'] == 0 and score['tp'] == 1
    assert score_frames([], {2: [box()]})['status'] == 'not_measured'


def test_missing_frame_prediction_counts_as_false_negative():
    score = score_frames([{'frame': 1, 'objects': [box()]}], {})
    assert score['fn'] == 1 and score['recall'] == 0 and score['precision'] is None


def test_duplicate_predictions_match_once():
    assert len(match_boxes([box()], [box(), box()])) == 1


def test_tracking_switch_and_miss_fragment_are_separate():
    truth = [{'frame': f, 'objects': [box(track_id='ground-truth-a')]} for f in [1, 2, 3, 4]]
    predicted = {1: [box(track_id=7)], 2: [], 3: [box(track_id=7)], 4: [box(track_id=9)]}
    score = score_frames(truth, predicted, tracking=True)
    assert score['id_switches'] == 1 and score['fragmentations'] == 1
    assert score['fn'] == 1 and score['recall'] == 0.75


def test_annotated_exit_and_reentry_is_not_a_fragment():
    truth = [{'frame': 1, 'objects': [box()]}, {'frame': 2, 'objects': []}, {'frame': 3, 'objects': [box()]}]
    assert score_frames(truth, {1: [box()], 3: [box()]}, tracking=True)['fragmentations'] == 0


def test_duplicate_ground_truth_frames_are_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        score_frames([{'frame': 1, 'objects': []}] * 2, {})


def test_duplicate_ground_truth_ids_are_rejected():
    with pytest.raises(ValueError, match='unique'):
        score_frames([{'frame': 1, 'objects': [box(), box()]}], {}, tracking=True)


def test_event_assignment_handles_overlapping_windows_and_duplicate_alerts():
    truth = [{'module': 'gate', 'event': 'entry', 'start_frame': a, 'end_frame': b} for a, b in [(1, 10), (1, 2)]]
    observed = [{'module': 'gate', 'event': 'entry', 'frame': f} for f in [1, 9, 10]]
    score = score_events(truth, observed, ['gate'])['gate']
    assert (score['tp'], score['fp'], score['fn']) == (2, 1, 0)
    assert score_events([], observed, []) == {}


def test_empty_negative_event_run_has_undefined_precision_not_perfect_accuracy():
    score = score_events([], [], ['fall'])['fall']
    assert score['precision'] is None and score['recall'] is None
    assert rates(0, 0, 0)['precision'] is None
    assert timing([])['mean_ms'] is None


def test_video_rule_time_uses_media_time_not_cpu_speed(monkeypatch):
    source = MediaTimedSource(VideoConfig(source='file', path='unused.mp4'))
    source.media_fps, source.wall_origin = 25, None
    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    now = datetime.now(timezone.utc)
    packets = iter([FramePacket(frame, f, 'test', monotonic_ts=500, wall_time=now) for f in [1, 26]])
    monkeypatch.setattr(VideoFileSource, 'read', lambda self: next(packets))
    first, second = source.read(), source.read()
    assert first.monotonic_ts == 0 and second.monotonic_ts == 1
    assert (second.wall_time - first.wall_time).total_seconds() == 1


def test_benchmark_includes_dependencies_without_enabling_everything():
    from computer_vision.core.config import load_config
    config = enabled_config(load_config(use_env=False), ['ppe_violation'], {})
    assert set(config.enabled_module_names()) == {'ppe_violation', 'ppe_detection', 'person_tracking', 'person_detection'}
    assert config.processing.max_fps == 0
