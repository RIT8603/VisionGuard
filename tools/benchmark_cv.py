"""Reproducible real-video benchmark using VisionGuard's existing pipeline.

Run ``python -m tools.benchmark_cv --manifest <json> --output runs/<run>``.
See docs/CV_BENCHMARK.md for annotation schema and metric limitations.
No camera settings, enrolments or running-server state are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import time

import numpy as np

from backend.services.alerts import AlertCenter
from computer_vision.core.config import ModuleConfig, VideoConfig, load_config
from computer_vision.core.output import BaseOutputSink
from computer_vision.core.pipeline import RunLimits, VisionPipeline
from computer_vision.identity.shared import replace_shared_store
from computer_vision.identity.store import IdentityStore
from computer_vision.modules.registry import ModuleRegistry
from computer_vision.sources.video_file import VideoFileSource
from tools.cv_metrics import score_events, score_frames, timing


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def registry_store(path: str, threshold: float) -> IdentityStore:
    """Read a snapshot of active templates; never update the real registry."""
    store = IdentityStore(threshold=threshold)
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
        for name, blob, dim in db.execute(
            'SELECT u.name,s.embedding,s.dim FROM face_samples s JOIN users u ON u.id=s.user_id WHERE u.active=1'
        ):
            embedding = np.frombuffer(blob, dtype=np.float32).copy()
            if embedding.size != dim:
                raise ValueError(f'Invalid template dimensions for {name}')
            store.enroll(name, embedding)
    return store


class MediaTimedSource(VideoFileSource):
    """Use original video time for dwell/gap rules while running unthrottled.

    This is an offline evaluation adapter; production source behavior is unchanged.
    """
    def open(self):
        info = super().open()
        if info.fps <= 0:
            self.release()
            raise ValueError('Evaluation needs a video with a known positive FPS')
        self.media_fps = info.fps
        self.wall_origin = None
        return info

    def read(self):
        packet = super().read()
        if packet is None:
            return None
        if self.wall_origin is None:
            self.wall_origin = packet.wall_time
        elapsed = (packet.frame_id - 1) / self.media_fps
        return replace(packet, monotonic_ts=elapsed, wall_time=self.wall_origin + timedelta(seconds=elapsed))


def enabled_config(base, requested: list[str], overrides: dict):
    params = {key: {**cfg.params, **overrides.get(key, {})} for key, cfg in base.modules.items()}
    for key, value in overrides.items():
        params.setdefault(key, dict(value))
    wanted = set(requested)
    pending = list(requested)
    while pending:
        key = pending.pop()
        cls = ModuleRegistry.resolve(key)
        for dependency in cls.required_modules(params.get(key, {})):
            if dependency not in wanted:
                wanted.add(dependency)
                pending.append(dependency)
    modules = {key: ModuleConfig.model_validate({**value, 'enabled': key in wanted}) for key, value in params.items()}
    for key in wanted - set(modules):
        modules[key] = ModuleConfig(enabled=True)
    return base.model_copy(update={'modules': modules, 'processing': base.processing.model_copy(update={'max_fps': 0})})


class BenchmarkSink(BaseOutputSink):
    def __init__(self, scenario: dict, output: Path, warmup: int):
        self.scenario, self.output, self.warmup = scenario, output, warmup
        self.records, self.events, self.errors = [], [], []
        self.latencies = defaultdict(list)
        self.alerts = AlertCenter()
        self.samples, self.seen_frames = Counter(), set()
        self.pipeline_times = []

    def emit(self, context, stats):
        number = self.scenario.get('start_frame', 0) + context.frame_id
        self.seen_frames.add(number)
        sx = context.metadata['original_width'] / context.width
        sy = context.metadata['original_height'] / context.height
        def objects(items, tracking=False):
            rows = []
            for item in items:
                row = {'label': item.label, 'confidence': float(item.confidence),
                       'bbox': [item.bbox.x1 * sx, item.bbox.y1 * sy, item.bbox.x2 * sx, item.bbox.y2 * sy]}
                if tracking:
                    row.update(track_id=item.track_id, time_since_update=item.time_since_update)
                # Never serialize RawFace objects, images or embedding vectors.
                extra = {k: v for k, v in item.extra.items()
                         if k in {'identity', 'recognised', 'verified', 'global_id', 'armed', 'identity_state'}
                         and (v is None or isinstance(v, (str, bool, int, float)))}
                if extra:
                    row['extra'] = extra
                rows.append(row)
            return rows
        detections = {k: objects(v) for k, v in context.detections.items()}
        tracks = {k: objects(v, True) for k, v in context.tracks.items()}
        for key, rows in {**detections, **tracks}.items():
            self.samples[key] += len(rows)
        events = [{**e.to_dict(), 'frame': number} for e in context.events]
        self.events.extend(events)
        self.alerts.ingest(context.source_id, events)
        if context.errors:
            self.errors.append({'frame': number, 'errors': dict(context.errors)})
        if context.frame_id > self.warmup:
            for key, result in context.module_results.items():
                self.latencies[key].append(result.latency_ms)
            self.pipeline_times.append(context.metadata['latency_ms'])
        self.records.append({'frame': number, 'detections': detections, 'tracks': tracks,
                             'events': events, 'errors': dict(context.errors)})
        return True


def run_scenario(base, scenario: dict, output: Path, warmup: int):
    name = scenario['id']
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Scenario id must be a simple filename')
    video_path = Path(scenario['video'])
    annotations = json.loads(Path(scenario['annotations']).read_text(encoding='utf-8-sig')) if scenario.get('annotations') else {}
    if annotations and annotations.get('coordinate_space') != 'source_pixels':
        raise ValueError('Annotations must declare coordinate_space=source_pixels')
    config = enabled_config(base, scenario['modules'], scenario.get('params', {}))
    modules_cfg = dict(config.modules)
    for key in ('weapon_detection', 'unknown_person'):
        if key in modules_cfg:
            modules_cfg[key] = ModuleConfig.model_validate({**modules_cfg[key].params,
                'enabled': modules_cfg[key].enabled, 'save_snapshots': False})
    if 'person_reid' in config.enabled_module_names():
        modules_cfg['person_reid'] = ModuleConfig.model_validate({**modules_cfg['person_reid'].params,
            'enabled': True, 'gallery_path': str(output / 'reid-gallery.npz')})
    config = config.model_copy(update={'modules': modules_cfg})
    video = VideoConfig(source='file', path=str(video_path), loop=False, playback_speed=0,
                        start_frame=scenario.get('start_frame', 0))
    config = config.model_copy(update={'video': video})
    sink = BenchmarkSink(scenario, output, warmup)
    started = time.perf_counter()
    modules = ModuleRegistry.build_enabled(config)
    load_seconds = time.perf_counter() - started
    # Capture resolved parameters before cleanup; makes thresholds and model hashes reproducible.
    parameters = {m.module_name: dict(m.params) for m in modules}
    backends = {m.module_name: m._backend.description for m in modules
                if getattr(m, '_backend', None) is not None}
    pipeline = VisionPipeline(config, MediaTimedSource(video, source_id=scenario.get('camera_id', name)), modules, sink)
    started = time.perf_counter()
    stats = pipeline.run(RunLimits(max_frames=scenario['frames']))
    elapsed = time.perf_counter() - started
    if stats.frames_processed != scenario['frames']:
        raise ValueError(f"{name}: expected {scenario['frames']} frames, read {stats.frames_processed}")
    metrics = {}
    for key, annotation in annotations.get('modules', {}).items():
        if key not in parameters:
            raise ValueError(f'Annotated module {key} was not enabled')
        if any(frame['frame'] not in sink.seen_frames for frame in annotation['frames']):
            raise ValueError(f'{name}: annotated frame outside evaluated range')
        tracking = annotation.get('tracking', False)
        namespace = 'tracks' if tracking else 'detections'
        predictions = {row['frame']: row[namespace].get(key, []) for row in sink.records}
        frames = annotation['frames']
        if annotation.get('labels'):
            labels = set(annotation['labels'])
            predictions = {n: [o for o in rows if o['label'] in labels] for n, rows in predictions.items()}
            frames = [{**f, 'objects': [o for o in f['objects'] if o['label'] in labels]} for f in frames]
        metrics[key] = score_frames(frames, predictions, tracking=tracking)
        metrics[key]['class_scope'] = annotation.get('labels', 'all configured classes')
    for key in parameters:
        metrics.setdefault(key, {'status': 'not_measured', 'reason': 'No labelled frames for this module'})
    result = {'id': name, 'video': str(video_path), 'video_sha256': file_hash(video_path),
              'frames': stats.frames_processed, 'failed_frames': stats.frames_failed,
              'model_load_seconds': load_seconds, 'wall_seconds_including_decode_sink_cleanup': elapsed,
              'end_to_end_fps': stats.frames_processed / elapsed,
              'pipeline_processing': timing(sink.pipeline_times),
              'per_module_latency': {k: timing(v) for k, v in sink.latencies.items()},
              'metrics': metrics, 'event_metrics': score_events(annotations.get('events', []), sink.events, annotations.get('event_modules', [])),
              'events': dict(Counter(e['event'] for e in sink.events)),
              'alerts': sink.alerts.listing(), 'observations': dict(sink.samples), 'errors': sink.errors,
              'parameters': parameters, 'backends': backends, 'clock': 'video media time', 'warmup_frames_excluded_from_latency': warmup,
              'annotation_provenance': annotations.get('provenance'), 'annotations': scenario.get('annotations')}
    with (output / f'{name}-predictions.jsonl').open('w', encoding='utf-8') as handle:
        for row in sink.records:
            handle.write(json.dumps(row) + '\n')
    (output / f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--only', help='Run a single scenario id')
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error('--warmup must be nonnegative')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'))
    base = load_config(args.config, use_env=False)
    if manifest.get('face_registry'):
        threshold = float(base.module_config('face_recognition').params.get('threshold', 0.363))
        store = registry_store(manifest['face_registry'], threshold)
        replace_shared_store(((name, store.get(name).embeddings) for name in store.names), threshold)
    scenarios = [s for s in manifest['scenarios'] if not args.only or s['id'] == args.only]
    if not scenarios or any(not isinstance(s.get('frames'), int) or s['frames'] <= args.warmup for s in scenarios):
        parser.error('Need scenarios with frames > warmup')
    report = {'manifest_sha256': file_hash(Path(args.manifest)), 'python': platform.python_version(),
              'notes': ['Equivalent per-module FPS is inverse call latency, not simultaneous camera throughput.',
                        'Shared-detector inference cost is attributed to its first consumer.',
                        'No annotation means precision/recall/false positives are NOT MEASURED.',
                        'Sparse tracking annotations score only the annotated sequence.'], 'scenarios': []}
    for scenario in scenarios:
        try:
            result = run_scenario(base, scenario, output, args.warmup)
            report['scenarios'].append(result)
            print(json.dumps({'id': result['id'], 'frames': result['frames'], 'failed_frames': result['failed_frames'],
                              'events': result['events'], 'fps': result['end_to_end_fps']}), flush=True)
        except Exception as exc:
            report['scenarios'].append({'id': scenario['id'], 'error': f'{type(exc).__name__}: {exc}'})
            print(f"ERROR {scenario['id']}: {exc}", flush=True)
        (output / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_markdown(report, output / 'benchmark.md')
    return int(any(s.get('error') or s.get('failed_frames') for s in report['scenarios']))


def write_markdown(report: dict, path: Path) -> None:
    def value(number):
        return 'not measured' if number is None else f'{number:.3f}'
    lines = ['# VisionGuard real-video benchmark', '', *['- ' + note for note in report['notes']], '']
    for scenario in report['scenarios']:
        lines.extend(['## ' + scenario['id'], ''])
        if scenario.get('error'):
            lines.extend(['Failed: ' + scenario['error'], ''])
            continue
        lines.extend([f"Frames: {scenario['frames']}; failed: {scenario['failed_frames']}; end-to-end FPS: {scenario['end_to_end_fps']:.2f}.", '',
                      '| Module | Mean ms | p95 ms | Inverse-call FPS | Accuracy scope | Precision | Recall | FP | ID switches | Fragments |',
                      '| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |'])
        for name in scenario['parameters']:
            latency = scenario['per_module_latency'].get(name, {})
            accuracy = scenario['metrics'].get(name, {})
            scope = 'boxes on labelled frames' if accuracy.get('status') == 'measured' else 'not measured'
            if name in scenario['event_metrics']:
                accuracy, scope = scenario['event_metrics'][name], 'events in labelled windows'
            columns = [name, value(latency.get('mean_ms')), value(latency.get('p95_ms')),
                       value(latency.get('equivalent_fps')), scope,
                       value(accuracy.get('precision')), value(accuracy.get('recall')),
                       str(accuracy.get('fp', 'not measured')), str(accuracy.get('id_switches', 'not measured')),
                       str(accuracy.get('fragmentations', 'not measured'))]
            lines.append('| ' + ' | '.join(columns) + ' |')
        lines.extend(['', 'Observed events: `' + json.dumps(scenario['events']) + '`.', '',
                      'Annotation provenance: ' + str(scenario.get('annotation_provenance') or 'No ground truth supplied.'), ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
