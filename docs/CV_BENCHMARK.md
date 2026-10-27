# Real-video evaluation

Run from the repository root with the existing Python environment:

Install the development dependencies in `requirements-dev.txt` first when using
a new environment; SciPy is used by the offline evaluators, not the live pipeline.

```powershell
.\.venv\Scripts\python.exe -B -m tools.benchmark_cv --manifest runs/module-validation-20261026/scenarios.json --output runs/module-validation-20261026/benchmark
```

The tool uses the existing registry, video source, frame processor, pipeline and
alert center. It does not change camera configuration or the running application.
Models are real; no fake detector backend is used. A missing model, short video or
module error is reported as a failure. Individual scenario errors do not suppress
results from other scenarios. `--only <id>` selects one scenario.

## Scenario manifest

Paths are relative to the repository working directory. Frame numbers in
annotations and predictions are **one-based source-video frame numbers**.
`start_frame` is a zero-based source seek offset. A source is never looped.

```json
{
  "face_registry": "data/faces/faces.db",
  "scenarios": [{
    "id": "gate",
    "video": "data/videos/example.mp4",
    "modules": ["gate_counter"],
    "frames": 300,
    "start_frame": 0,
    "params": {"gate_counter": {"line": [[0.05, 0.7], [0.95, 0.7]]}},
    "annotations": "runs/example-truth.json"
  }]
}
```

Dependencies are enabled recursively. Parameters are copied from the current
config and overridden only for this evaluation. All effective parameters are
saved in each result. Snapshot saving is disabled; Re-ID uses a shared gallery
inside the output directory, so evaluation does not pollute the production gallery.
Use a fresh output directory for independent Re-ID experiments. Repeat scenarios
in one invocation to test gallery persistence, or supply distinct `camera_id`
values. Replaying one video as two sources tests gallery plumbing, **not real
cross-camera accuracy**.

The optional registry is read in SQLite read-only mode. Its templates are loaded
only into this evaluation process. It does not enroll or modify users. Without
labelled independent identity examples, loading enrolled templates is not an
identity-accuracy measurement.

## Ground truth

Do not copy model predictions into ground truth. Label raw frames independently,
include all target objects for a scored module/frame, document the reviewer and
source, and reserve independent footage for final accuracy evaluation.

```json
{
  "coordinate_space": "source_pixels",
  "provenance": "Human-reviewed raw frames; reviewer and limitations here",
  "modules": {
    "person_detection": {"frames": [
      {"frame": 1, "objects": [{"label": "person", "bbox": [10, 20, 90, 220]}]}
    ]},
    "person_tracking": {"tracking": true, "frames": [
      {"frame": 1, "objects": [{"label": "person", "bbox": [10, 20, 90, 220], "track_id": "human-a"}]}
    ]}
  },
  "event_modules": ["gate_counter"],
  "events": [{"module": "gate_counter", "event": "gate_entry", "start_frame": 100, "end_frame": 130}]
}
```

`event_modules` asserts complete event annotation across the evaluated interval.
An empty event list with a listed module explicitly denotes a negative scenario.
Omitting a module means no precision/recall claim is made for it. Event matching
uses type and frame window, not an unverified mapping of predicted person IDs to
human identities. Use individual-person scenarios when that distinction matters.

## Interpreting results

- Precision = TP/(TP+FP); recall = TP/(TP+FN). Undefined denominators are `null`.
- Same-class Hungarian matching at IoU >= 0.5 is used. No annotation produces
  `not_measured`, never a made-up zero or perfect score. Unlabelled frames are not
  false positives. False positives on labelled frames are reported as counts.
- Tracking reports ID switches and miss/reacquisition fragmentation on the
  annotated sequence. Sparse annotation can miss errors between sampled frames;
  these are not a full MOTChallenge evaluation, IDF1 or HOTA result.
- Rule timers use video media time during unthrottled offline evaluation, so
  slow hardware does not change the meaning of a 30-second dwell. Production
  capture timestamps are unchanged.
- Latency excludes the initial `--warmup` frames (default 10). Mean/p50/p95 and
  inverse-latency equivalent FPS are per module call; skipped-inference frames
  and cache hits are included. Shared inference cost falls on its first consumer.
  End-to-end FPS includes decoding, sink collection and cleanup, excludes model
  construction, and is measured separately from summed processing times.
- Record whether other camera workers were running. These figures are not an
  isolated GPU capacity test or a five-camera performance guarantee.
- Positive detections and generated dashboard alerts prove that path executed;
  they do not establish generalization to every camera or external notification
  delivery. Real PPE, fall and ownership/tailgating scenarios require appropriate
  footage and independently reviewed event labels.

Outputs are JSON metrics and one JSONL prediction file per scenario. Files can
contain enrolled names and appearance IDs; treat them as local project data.
`benchmark.md` provides the per-module latency/precision/recall/FP/ID-switch table.
An annotation module can set `labels` to restrict its explicitly disclosed class
scope (for example, Safety Vest only); other classes are then unscored.

## Existing enrolled-face diagnostic

```powershell
.\.venv\Scripts\python.exe -B -m tools.evaluate_recognition --registry data/faces/faces.db --photo-manifest "data/faces/New user face data/face-photos/manifest.csv" --config configs/config.yaml --json-output runs/module-validation-20261026/face-registry-diagnostic.json
```

This reads the real active registry and uses the configured detector/embedder.
It scores every supplied photo, including detection failures, uses employee IDs
or names from the export metadata, and reports near-identical template overlap.
The report explicitly labels this as an enrollment-export diagnostic. It does
not measure independent CCTV name accuracy or the production module's temporal
quality/confirmation rules. Do not substitute these scores for an independent,
human-labelled CCTV evaluation. The existing `--faces/--detector/--embedder`
image-split evaluator remains supported.
