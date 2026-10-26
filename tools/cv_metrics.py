"""Offline metrics. Only explicitly annotated frames/windows are scored.

Boxes use source-image pixels. Matching maximizes the number of same-label
IoU>=threshold pairs, then their IoU. Tracking IDs are compared only within a
scenario. Sparse annotations measure switches/fragments on the annotated
sequence, not on the intervening unlabelled frames.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area - intersection) if area > intersection else 0.0


def match_boxes(truth: list[dict], predicted: list[dict], threshold=0.5):
    if not truth or not predicted:
        return []
    scores = np.zeros((len(truth), len(predicted)))
    for g, item in enumerate(truth):
        for p, candidate in enumerate(predicted):
            overlap = iou(item['bbox'], candidate['bbox'])
            if item['label'] == candidate['label'] and overlap >= threshold:
                scores[g, p] = max(scores.shape) + 1 + overlap
    rows, cols = linear_sum_assignment(-scores)
    return [(int(g), int(p)) for g, p in zip(rows, cols) if scores[g, p] > 0]


def rates(tp: int, fp: int, fn: int) -> dict:
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': tp / (tp + fp) if tp + fp else None,
        'recall': tp / (tp + fn) if tp + fn else None,
    }


def score_frames(annotations: list[dict], predictions: dict[int, list[dict]], *,
                 threshold=0.5, tracking=False) -> dict[str, Any]:
    """Missing prediction on a labelled frame is a miss, never silently dropped."""
    if not annotations:
        return {'status': 'not_measured', 'reason': 'No labelled frames supplied'}
    tp = fp = fn = switches = fragments = 0
    previous_id, ever_matched, was_matched = {}, set(), {}
    classes: dict[str, list[int]] = {}
    seen_frames = set()
    for frame in sorted(annotations, key=lambda f: f['frame']):
        number = frame['frame']
        if number in seen_frames:
            raise ValueError(f'Duplicate annotated frame {number}')
        seen_frames.add(number)
        truth, predicted = frame['objects'], predictions.get(number, [])
        if tracking:
            truth_ids = [str(o['track_id']) for o in truth]
            if len(truth_ids) != len(set(truth_ids)):
                raise ValueError('Ground-truth track IDs must be unique per frame')
        pairs = match_boxes(truth, predicted, threshold)
        matched_g, matched_p = {g for g, _ in pairs}, {p for _, p in pairs}
        tp += len(pairs)
        fp += len(predicted) - len(pairs)
        fn += len(truth) - len(pairs)
        for g, item in enumerate(truth):
            counts = classes.setdefault(item['label'], [0, 0, 0])
            counts[0 if g in matched_g else 2] += 1
        for p, item in enumerate(predicted):
            if p not in matched_p:
                classes.setdefault(item['label'], [0, 0, 0])[1] += 1
        if tracking:
            current = {str(truth[g]['track_id']): str(predicted[p]['track_id']) for g, p in pairs}
            present = {str(item['track_id']) for item in truth}
            for identity in present:
                if identity in current:
                    if identity in previous_id and previous_id[identity] != current[identity]:
                        switches += 1
                    if identity in ever_matched and was_matched.get(identity) is False:
                        fragments += 1
                    previous_id[identity] = current[identity]
                    ever_matched.add(identity)
                was_matched[identity] = identity in current
            # An annotated absence ends visibility; re-entry is not a detector fragmentation.
            for identity in set(was_matched) - present:
                was_matched.pop(identity)
    result = {'status': 'measured', 'annotated_frames': len(annotations),
              'iou_threshold': threshold, **rates(tp, fp, fn),
              'per_class': {label: rates(*values) for label, values in classes.items()}}
    if tracking:
        result.update(id_switches=switches, fragmentations=fragments,
                      tracking_definition='IoU Hungarian on annotated frames; ID switches retain last match across gaps')
    return result


def score_events(truth: list[dict], predicted: list[dict], modules: list[str]) -> dict:
    """Each expected event matches one prediction in its inclusive frame window.

    Event modules listed here assert complete annotation for the entire run.
    Missing a module from that list means its event precision is unknown.
    """
    result = {}
    for module in modules:
        expected = sorted((e for e in truth if e['module'] == module), key=lambda e: e['end_frame'])
        observed = sorted((e for e in predicted if e['module'] == module), key=lambda e: e['frame'])
        # Bipartite assignment avoids a broad window stealing a narrow window's only match.
        scores = np.zeros((len(expected), len(observed)))
        for g, event in enumerate(expected):
            for p, candidate in enumerate(observed):
                if event['event'] == candidate['event'] and event['start_frame'] <= candidate['frame'] <= event['end_frame']:
                    scores[g, p] = 1
        matches = 0
        if scores.size:
            rows, cols = linear_sum_assignment(-scores)
            matches = int(sum(scores[g, p] for g, p in zip(rows, cols)))
        result[module] = {'status': 'measured', **rates(matches, len(observed) - matches, len(expected) - matches),
                          'expected_events': len(expected), 'observed_events': len(observed)}
    return result


def timing(values: list[float]) -> dict:
    if not values:
        return {'samples': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'equivalent_fps': None}
    average = float(np.mean(values))
    return {'samples': len(values), 'mean_ms': average, 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95)),
            'equivalent_fps': 1000 / average if average > 0 else None}
