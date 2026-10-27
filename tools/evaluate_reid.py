"""Small labelled body-appearance diagnostic; separate from cross-camera accuracy.

Uses manually labelled source-video boxes from the benchmark annotation schema.
No faces or names are inferred. Gallery and query frames must be distinct.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from computer_vision.core.config import load_config
from computer_vision.inference.person_embedder import PersonEmbedder
from tools.cv_metrics import rates, timing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--gallery-frame', type=int, required=True)
    parser.add_argument('--query-frame', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args(argv)
    if args.gallery_frame < 1 or args.query_frame < 1 or args.gallery_frame == args.query_frame:
        parser.error('Need distinct positive one-based frame numbers')
    annotation = json.loads(Path(args.annotations).read_text(encoding='utf-8-sig'))
    if annotation.get('coordinate_space') != 'source_pixels':
        parser.error('Need source_pixels annotations')
    truth = {r['frame']: r['objects'] for r in annotation['modules']['person_tracking']['frames']}
    params = load_config(args.config, use_env=False).module_config('person_reid').params
    model = PersonEmbedder(params['model'], sha256=params.get('model_sha256'), device=params.get('device', 'cpu'))
    capture = cv2.VideoCapture(args.video)
    features, skipped, latencies = {}, [], []
    try:
        for number in [args.gallery_frame, args.query_frame]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f'Cannot read frame {number}')
            features[number] = {}
            for item in truth[number]:
                x1, y1, x2, y2 = [int(v) for v in item['bbox']]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
                if y2 - y1 < params.get('min_height', 100) or x2 - x1 < 8:
                    skipped.append({'frame': number, 'identity': item['track_id'], 'reason': 'below configured minimum crop size'})
                    continue
                started = time.perf_counter()
                features[number][str(item['track_id'])] = model.embed(image[y1:y2, x1:x2])
                latencies.append((time.perf_counter() - started) * 1000)
    finally:
        capture.release()
    gallery, queries = features[args.gallery_frame], features[args.query_frame]
    if not gallery or not queries:
        raise ValueError('No eligible labelled crops to evaluate')
    threshold = float(params.get('threshold', .6))
    pairs, decisions = [], []
    tp = fp = fn = tn = correct = 0
    for identity, query in queries.items():
        scores = {name: float(np.dot(query, reference)) for name, reference in gallery.items()}
        winner = max(scores, key=scores.get)
        predicted = winner if scores[winner] >= threshold else None
        correct += predicted == identity
        decisions.append({'expected': identity, 'predicted': predicted, 'similarity': scores[winner]})
        for name, score in scores.items():
            same, accepted = identity == name, score >= threshold
            tp += same and accepted
            fp += not same and accepted
            fn += same and not accepted
            tn += not same and not accepted
            pairs.append({'query': identity, 'gallery': name, 'expected_same': same, 'score': score, 'accepted': accepted})
    result = {'protocol': 'Same-camera two-frame body-crop descriptor diagnostic',
              'cross_camera_accuracy': 'not measured by this diagnostic',
              'limitations': ['Small manually boxed sample with anonymous continuity labels; not a full Re-ID dataset.',
                             'This bypasses production detection, multi-sample averaging, in-view exclusions and gallery updating.'],
              'video': args.video, 'annotations': args.annotations, 'annotation_provenance': annotation.get('provenance'),
              'parameters': params, 'gallery_frame': args.gallery_frame, 'query_frame': args.query_frame,
              'gallery_crops': len(gallery), 'query_crops': len(queries), 'skipped': skipped,
              'pair_metrics': {**rates(int(tp), int(fp), int(fn)), 'tn': int(tn), 'pairs': len(pairs)},
              'top1_correct_queries': int(correct), 'top1_total_queries': len(queries),
              'embedding_latency': timing(latencies), 'decisions': decisions, 'pairs': pairs}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['protocol', 'pair_metrics', 'top1_correct_queries', 'top1_total_queries']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
