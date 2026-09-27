# VisionGuard AI

A configurable, modular real-time computer vision platform.

A camera feeds a Vision Core; the Vision Core runs whichever AI modules the
configuration enables; the modules publish structured results. Swapping a
development webcam for a production RTSP camera is a configuration change, not
a code change.

```
Camera Source -> Video Source Abstraction -> OpenCV Vision Core
              -> Module Registry -> Enabled AI Modules
              -> Shared FrameContext -> Annotated Output / Dashboard
```

---

## Status: Phase 9 (Pose, Fall, Attendance) — in progress

Development workflow: **[docs/WORKFLOW.md](docs/WORKFLOW.md)**

**All three video sources and eleven AI modules are implemented**, and the
platform now produces **events**, not just detections:

```
Webcam / Video file / RTSP camera
        -> OpenCV -> Detection -> Tracking -> Analysis -> Events + annotated video
```

| Delivered | Not yet (and by design) |
|---|---|
| Project structure | Face detection / recognition — Phase 7 |
| Configuration (YAML + env + CLI) | Higher-level features — Phase 9 |
| Structured logging | Dashboard — Phase 10 |
| `BaseVideoSource` abstraction | Cloud / Agentic — Phases 11–12 |
| `WebcamSource` — development | |
| `VideoFileSource` — evaluation | |
| `RTSPSource` — production | |
| `FrameContext` | |
| `BaseAIModule` interface | |
| `ModuleRegistry` + dependency ordering | |
| Pipeline, FPS / latency measurement | |
| Live annotated window with HUD | |
| `DetectorBackend` abstraction + model sharing | |
| `IoUTracker` (reusable algorithm) | |
| Reconnect, timeouts, stale-frame dropping | |
| Credential redaction in every log path | |

### AI modules

| Module | Type | Consumes | Status |
|---|---|---|---|
| `person_detection` | detection | — | working (COCO) |
| `object_detection` | detection | — | working (COCO, 80 classes) |
| `vehicle_detection` | detection | — | working (COCO vehicle subset) |
| `person_tracking` | tracking | `person_detection` | working |
| `object_tracking` | tracking | `object_detection` | working |
| `vehicle_tracking` | tracking | `vehicle_detection` | working |
| `crowd_detection` | analysis | `person_detection` | working |
| `pose_detection` | detection | — | working (COCO 17 keypoints) |
| `restricted_zone` | analysis | any tracking module | working (emits events) |
| `fall_detection` | analysis | `pose_detection` | working (emits events) |
| `face_detection` | recognition | — | needs a model file |
| `face_recognition` | recognition | `face_detection` | needs a model + identity store |
| `unknown_person` | analysis | `face_recognition` | working (emits events) |
| `attendance` | analysis | `face_recognition` | working (emits events) |
| `ppe_detection` | detection | — | needs custom weights |
| `fire_detection` | detection | — | needs custom weights |
| `smoke_detection` | detection | — | needs custom weights |

Still to build: `ppe_violation` (blocked on PPE weights), loitering, intrusion.

Measured with seven modules enabled at once (three detectors sharing one
model, three trackers, one analysis), 640px webcam on CPU:

```
20.0 FPS | 22.7 ms frame latency
person_detection 22.18 ms | every other module <= 0.02 ms
```

The architectural promise from Phase 1 has been paid off in full: moving from a
laptop webcam to a CCTV camera is a `config.yaml` change, and adding a module
is one file plus one registry line.

---

## Install

Python 3.12 recommended (3.10+ works).

```bash
py -3.12 -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install --upgrade pip
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

This installs `ultralytics`, which pulls in torch and torchvision — about
**1.2 GB on disk and ~2 minutes**. It is needed only for person detection; see
[Detection backend](#detection-backend) if you want to avoid it. The model
weights (`yolo11n.pt`, ~5 MB) download automatically on first run.

Optional: `cp .env.example .env` and edit it.

---

## Run

Live webcam preview with the HUD:

```bash
.venv/Scripts/python.exe -m backend.main
```

Press `q` or `ESC` in the window (or close it) to stop.

Headless, fixed number of frames — the repeatable smoke test:

```bash
.venv/Scripts/python.exe -m backend.main --no-window --max-frames 90
```

Watch it in a browser instead of an OpenCV window:

```bash
.venv/Scripts/python.exe -m backend.main --no-window --dashboard
```

Analyse a video file instead — `--source file` is inferred from the flag:

```bash
.venv/Scripts/python.exe -m backend.main --video-path data/samples/office.mp4
```

Show what the platform can currently load:

```bash
.venv/Scripts/python.exe -m backend.main --list-modules
```

Useful flags: `--dashboard`, `--dashboard-port`, `--config`, `--source`,
`--device-id`, `--video-path`, `--loop`,
`--playback-speed`, `--start-frame`, `--width/--height`, `--target-width`,
`--mirror`, `--no-window`, `--max-frames`, `--duration`, `--log-level`. Run
`--help` for the full list.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Unhandled error |
| 2 | Configuration error |
| 3 | Video source error / unavailable |
| 4 | AI module error |

---

## Test

```bash
.venv/Scripts/python.exe -m pytest
```

584 tests. None needs a camera, a display, a model or any real face data — the
webcam source runs
against a fake `cv2.VideoCapture`, the pipeline tests use a synthetic frame
source, and the detection module is tested against a fake backend. Tracking is
tested on scripted detections, which is the only way to assert id persistence
deterministically. The file-source tests write real video with
`cv2.VideoWriter` and read it back, so encode, decode, EOF, looping and seeking
are genuinely exercised.

Four further tests exercise the real model against real images. They are
deselected by default and skip cleanly if the backend is missing:

```bash
.venv/Scripts/python.exe -m pytest -m integration
```

---

## Expected output

Headless run with detection + tracking (1280x720 / 30 fps USB webcam, CPU):

```
INFO | visionguard              | VisionGuard AI 0.3.0 - Phase 3 - Person Tracking
INFO | computer_vision.modules.registry     | Modules - enabled: ['person_detection', 'person_tracking'] | ...
INFO | computer_vision.modules.registry     | Module execution order: person_detection -> person_tracking
INFO | computer_vision.inference.ultralytics.. | Loaded yolo11n.pt on device 'cpu' (imgsz=640, 80 classes)
INFO | computer_vision.inference.ultralytics.. | Model warmup took 1893 ms
INFO | module.person_detection  | Ready: backend=ultralytics:yolo11n.pt@cpu, label='person', confidence>=0.40
INFO | module.person_tracking   | Ready: consuming 'person_detection' (iou>=0.30, max_age=30, min_hits=2)
INFO | computer_vision.sources.opencv_base  | Source ready in 6121 ms: webcam-0 @ 1280x720, 30.0 fps
INFO | computer_vision.core.frame_processor | Processing resolution: 1280x720 -> 960x540
INFO | computer_vision.core.output          | frame=60 fps=19.7 latency=39.7ms detections=0 tracks=0
INFO | visionguard              | Done: 60 frames in 3.5 s | avg 16.8 FPS | avg processing latency 39.7 ms | avg capture wait 10.9 ms
```

Per-module cost:

```
module_latency_ms: {'person_detection': 34.43, 'person_tracking': 0.02}
```

Detection dominates the frame budget on CPU. Tracking is effectively free —
the IoU tracker does no inference, which is exactly why it was chosen over a
deep-appearance tracker for this phase.

Windowed run: each person gets a coloured box labelled `#1 person 0.88`, where
the number is the persistent track id and the colour is tied to that id — so an
id switch is immediately visible. The HUD lists the app name, camera, enabled
modules, FPS, latency, frame number, resolution and the live track list.

### Detection accuracy vs speed

Measured on this machine (CPU, `yolo11n.pt`, a 960x540 frame containing four
people):

| `imgsz` | median latency | est. FPS | people found |
|---|---|---|---|
| 320 | 23.8 ms | 42.1 | 3 (misses the partly occluded one) |
| 416 | 30.5 ms | 32.8 | 4 |
| 512 | 37.8 ms | 26.5 | 4 |
| **640** (default) | 45.2 ms | 22.1 | 4 |

416 found everything 640 did at a third less cost, so it is a good choice for
close-range webcam work. The default stays at 640 because smaller inference
sizes genuinely lose small and distant people, which is exactly the CCTV case
this platform targets. Measure on **your** footage before lowering it.

---

## Project structure

Three layers, and the dependency arrow only ever points one way:

```
frontend/  ──uses──▶  backend/  ──uses──▶  computer_vision/
                                                   │
                                            reads  ▼
                                              models/
```

```
VisionGuard/
├── computer_vision/        THE VISION ENGINE - runs with no server, no API, no UI
│   ├── core/               pipeline, context, config, frame processing, output sinks
│   ├── sources/            webcam, video file, RTSP  (one interface)
│   ├── modules/            detection / recognition / tracking / analysis
│   │   ├── detection/      person, object, vehicle, pose, crowd, PPE, fire, smoke
│   │   ├── recognition/    face detection, face recognition, unknown person
│   │   ├── tracking/       person, object, vehicle  +  the IoU algorithm
│   │   └── analysis/       restricted zone, fall, attendance  (produce events)
│   ├── inference/          model adapters: Ultralytics, OpenCV face, pose, sharing
│   ├── identity/           enrolled face embeddings  (personal data)
│   └── utils/              logging, timing, annotation
│
├── backend/                API + ORCHESTRATION - connects the engine to the UI
│   ├── main.py             CLI entry point
│   ├── api/server.py       read-only HTTP API + MJPEG stream
│   └── services/           dashboard state, the sink that feeds it
│
├── frontend/               USER INTERFACE
│   └── static/index.html
│
├── models/                 ALL AI MODEL WEIGHTS - nothing else
│   ├── detection/          yolo11n.pt
│   ├── pose/               yolo11n-pose.pt
│   ├── face/               YuNet + SFace .onnx, identities.npz
│   └── custom/             your own PPE / fire / smoke weights
│
├── configs/config.yaml     every behavioural switch
├── docs/                   the authoritative specification
├── tests/                  584 unit + 4 integration
└── tools/                  offline evaluation (recognition accuracy)
```

**Why these boundaries.** `computer_vision/` must stay runnable with no web
stack at all — that is what keeps it embeddable and testable. So the sink that
feeds the dashboard lives in `backend/services/web_sink.py`, not in the engine.
A lint you can run at any time:

```bash
grep -rn "from backend" computer_vision/   # must print nothing
```

> **Naming note.** Python package names cannot contain spaces
> (`import "Computer Vision"` is a syntax error), so the folders use PEP 8
> snake_case: `computer_vision`, not `Computer Vision`.

### How a frame flows

```
configs/config.yaml ─┐
.env / environment ──┼─> core/config.py ──> VisionGuardConfig
CLI flags ───────────┘                            │
                                                  ├─> sources/factory.py
                                                  │      └─> WebcamSource
                                                  │          (BaseVideoSource)
                                                  ├─> modules/registry.py
                                                  │      └─> enabled modules,
                                                  │          in dependency order
                                                  └─> core/pipeline.py
                                                           │
                                        FramePacket ──> FrameProcessor
                                                           │
                                                      FrameContext
                                                           │
                                           module.safe_process(context) x N
                                                           │
                                     ┌─────────────────────┴──────────────────┐
                                     ▼                                        ▼
                          core/output.py                        backend/services/
                       (WindowSink / NullSink)                    web_sink.py
                                                                       │
                                                             backend/api/server.py
                                                                       │
                                                              frontend/static/
```

| File | Responsibility |
|---|---|
| [backend/main.py](backend/main.py) | Entry point: flags, wiring, exit codes. No vision logic. |
| [computer_vision/core/config.py](computer_vision/core/config.py) | The only place that reads YAML or the environment. |
| [computer_vision/core/context.py](computer_vision/core/context.py) | `FrameContext`, `BBox`, `Detection`, `Track`, `Event`. The module-to-module channel. |
| [computer_vision/modules/analysis/zone.py](computer_vision/modules/analysis/zone.py) | Restricted Zone. Consumes tracks, emits events. |
| [computer_vision/modules/analysis/geometry.py](computer_vision/modules/analysis/geometry.py) | Polygons, containment and the entry/exit state machine. |
| [computer_vision/core/pipeline.py](computer_vision/core/pipeline.py) | The Vision Core loop. Names no module and no model. |
| [computer_vision/core/frame_processor.py](computer_vision/core/frame_processor.py) | Mirror + resize, once, to a single pipeline resolution. |
| [computer_vision/core/output.py](computer_vision/core/output.py) | Output sinks: window, headless, web, and `MultiSink`. |
| [backend/api/server.py](backend/api/server.py) | Read-only HTTP API and MJPEG stream. |
| [backend/services/state.py](backend/services/state.py) | Thread-safe handoff from pipeline to server. |
| [computer_vision/sources/base.py](computer_vision/sources/base.py) | The source contract. Imports no OpenCV. |
| [computer_vision/sources/opencv_base.py](computer_vision/sources/opencv_base.py) | Shared `cv2.VideoCapture` plumbing. |
| [computer_vision/sources/webcam.py](computer_vision/sources/webcam.py) | Live camera. Development input. |
| [computer_vision/sources/video_file.py](computer_vision/sources/video_file.py) | Video file. EOF, loop, seek, pacing. Evaluation input. |
| [computer_vision/sources/rtsp.py](computer_vision/sources/rtsp.py) | RTSP camera. Reconnect, timeouts, stale-frame dropping, redaction. Production input. |
| [computer_vision/sources/factory.py](computer_vision/sources/factory.py) | The only code that maps a config value to a source class. |
| [computer_vision/modules/base.py](computer_vision/modules/base.py) | `BaseAIModule`, `ModuleResult`, error containment. |
| [computer_vision/modules/registry.py](computer_vision/modules/registry.py) | Catalogue, lazy import, dependency ordering. |
| [computer_vision/modules/detection/base.py](computer_vision/modules/detection/base.py) | Shared detector plumbing: run model, filter labels, publish. |
| [computer_vision/modules/detection/person.py](computer_vision/modules/detection/person.py) | Person Detection. Finds people; nothing else. |
| [computer_vision/modules/detection/crowd.py](computer_vision/modules/detection/crowd.py) | Crowd measurement. Consumes detections, runs no model. |
| [computer_vision/modules/tracking/base.py](computer_vision/modules/tracking/base.py) | Shared tracker plumbing: read a detector, associate, publish ids. |
| [computer_vision/modules/tracking/person.py](computer_vision/modules/tracking/person.py) | Person Tracking. Assigns ids; runs no detector. |
| [computer_vision/modules/tracking/tracker_core.py](computer_vision/modules/tracking/tracker_core.py) | The IoU tracking algorithm. Knows nothing about people or modules. |
| [computer_vision/inference/base.py](computer_vision/inference/base.py) | `DetectorBackend` contract. Modules depend on this, not on a runtime. |
| [computer_vision/inference/shared.py](computer_vision/inference/shared.py) | One load, one inference, many modules. Reference-counted. |
| [computer_vision/identity/store.py](computer_vision/identity/store.py) | Enrolled identities. Embeddings only, never images. |
| [computer_vision/inference/opencv_face.py](computer_vision/inference/opencv_face.py) | YuNet detection and SFace embedding. |
| [computer_vision/inference/ultralytics_backend.py](computer_vision/inference/ultralytics_backend.py) | The only file that imports `ultralytics`. |
| [computer_vision/utils/timing.py](computer_vision/utils/timing.py) | FPS / latency meters, `PipelineStats`. |
| [computer_vision/utils/annotate.py](computer_vision/utils/annotate.py) | Drawing. Separate from both pipeline and modules. |

---

## Architectural guarantees, and where they live

**The pipeline never knows which modules exist.**
[pipeline.py](computer_vision/core/pipeline.py) iterates a list it was handed and calls
`safe_process`. Adding fire detection later does not touch this file.

**Disabled modules never load their models.**
`BUILTIN_MODULES` in [registry.py](computer_vision/modules/registry.py) maps a name to an
import path. `build_enabled` resolves *only* enabled names, so a disabled
`face_recognition` never imports a face-embedding library, let alone loads one.
`initialize()` is called only for enabled modules.

**Detection runs before tracking, without hard-coding either.**
A module declares `requires = ("person_detection",)`. The registry
topologically sorts on that, breaking ties by config order.

**Modules communicate only through `FrameContext`.**
Results are namespaced by producing module, so `person_detection` and
`ppe_detection` cannot collide and a consumer asks for exactly its dependency.

**One broken module cannot take down the platform.**
`safe_process` contains exceptions into the result and the context; after 10
consecutive failures the pipeline drops that module and keeps running.

**Bounding boxes never need rescaling.**
The frame is resized once, and that one resolution is used for both inference
and display.

---

## Pose and fall detection

```yaml
pose_detection:
  enabled: true
  model: yolo11n-pose.pt     # downloaded on first use
  confidence_threshold: 0.4
  keypoint_threshold: 0.3

fall_detection:
  enabled: true              # requires pose_detection
  min_confidence: 0.6        # combined posture score, 0-1
  min_frames: 5              # consecutive frames before reporting
  cooldown_seconds: 20
```

`pose_detection` publishes 17 COCO keypoints per person to `context.poses`,
**and** a plain person box to `context.detections` — so a tracker can consume
it without knowing joints exist.

`fall_detection` runs no model. It combines three independent geometric
signals, so no single one can raise an alarm alone:

| Signal | Reads | Fooled by |
|---|---|---|
| `box_ratio` | tall box vs wide box | crouching, overhead cameras |
| `torso_angle` | shoulder→hip line vs vertical | shoulders or hips occluded |
| `height_collapse` | head→ankle span vs body width | feet out of frame |

Signals whose joints the model could not see are dropped and the rest
re-weighted, so a partly visible person still gets a usable score instead of a
zero.

Measured on real footage: **upright people score 0.000, a horizontal person
scores 1.000.**

```
person 1: visible_joints=17/17  fall_score=0.000
person 2: visible_joints=8/17   fall_score=0.000   (2 of 3 signals used)
rotated : fall_score=1.000      {box_ratio: 1.0, torso_angle: 1.0, height_collapse: 1.0}
```

Cost: pose ~46 ms on CPU at `imgsz: 416`; fall analysis **0.09 ms**.

> **This is a heuristic, not a medical device.** It will miss falls and it will
> raise false alarms. `min_frames` matters more than the geometry — a single
> frame of someone tying a shoelace looks exactly like the start of a fall, and
> a system that cries wolf gets switched off. Tune against your own footage,
> and never make it the only safeguard for someone's safety.

Known limitation: the pose model is trained on upright people, so it detects
fewer people the more horizontal they are. A genuine fall is easier for it than
the rotated-image test above, but partial detection loss is expected.

---

## Attendance

```yaml
attendance:
  enabled: true          # requires face_recognition
  min_sightings: 3       # frames before a check-in
  absence_timeout: 60    # seconds unseen before a check-out
```

Turns "alice's face was recognised" into "alice arrived" and "alice left".
Kept out of face recognition on purpose (Agent.md §14): recognition answers a
question about one frame, attendance is a statement about a span of time.

- **Arrival needs corroboration.** One recognised frame is not an arrival —
  recognition has a false-positive rate, and a mistaken check-in is worse than
  a late one.
- **Departure is an absence, not an event.** Nobody emits "left the frame", so
  check-out fires after `absence_timeout` with nobody seen — long enough to
  survive someone turning around or being briefly occluded.
- **Returning needs fresh corroboration.** Sightings reset on check-out.

Emits `attendance_check_in` / `attendance_check_out`, the latter carrying
`duration_seconds`. Persisting a timesheet is a downstream concern.

---

## Face recognition

**This is the part of VisionGuard that produces personal data.** All three
modules are off by default and stay off unless you enable them deliberately
(Agent.md §24).

Three separate modules, because they answer three separate questions:

| Module | Question | Personal data |
|---|---|---|
| `face_detection` | *Where is a face?* | none beyond a box |
| `face_recognition` | *Whose face is this?* | identities |
| `unknown_person` | *Is this nobody we know?* | identities + events |

### Models

Neither model ships with OpenCV 5 — its Python wheel no longer bundles any
face data. Download both from the OpenCV Zoo and point the config at them:

| File | Size | Source |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | ~340 KB | [opencv_zoo/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `face_recognition_sface_2021dec.onnx` | ~37 MB | [opencv_zoo/face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |

Enabling a module without its model fails at startup with the URL above, not
with a silent zero-detections run.

### The identity store

Enrolment is deliberate. Nothing is ever added by being seen.

```python
from computer_vision.identity import IdentityStore

store = IdentityStore(threshold=0.363)
store.enroll("alice", embedding)   # embedding from SFaceEmbedder.embed()
store.save("models/identities.npz")
```

```yaml
face_recognition:
  enabled: true
  model: models/face_recognition_sface_2021dec.onnx
  identity_store: models/identities.npz
  threshold: 0.363     # cosine similarity; SFace's documented value
```

Design decisions, and why:

- **Embeddings only, never images.** The store holds float vectors. It never
  writes a face crop or photograph. A leaked store exposes numbers you can
  compare against a face you already have; it does not hand anyone a gallery
  of faces. A test asserts the saved archive contains exactly
  `names`, `embeddings`, `enrolled_at`, `version` — nothing else.
- **No pickle on load.** Names are stored as unicode arrays so the file loads
  with `allow_pickle=False`. A store path comes from configuration, and a
  pickled array in a swapped-in file would be arbitrary code execution.
- **Deletion is a first-class operation.** `store.remove(name)` drops an
  identity and every sample of it — the request a data subject is most likely
  to make, so it is one call rather than something you assemble.
- **One portable file.** A single `.npz` that can be inspected, diffed,
  backed up and deleted in one operation.
- **Several samples per identity**, matched best-of rather than averaged —
  averaging distinct poses produces a vector resembling neither.
- **Recognition refuses to start without a store.** Silently labelling
  everyone "unknown" would look like it was working.
- **Dimension mismatches are rejected**, so a store built against a different
  face model cannot be compared against the current one.

### Measuring recognition accuracy

The pipeline measures FPS continuously and accuracy **not at all**, because
accuracy needs ground truth. Supply it:

```bash
.venv/Scripts/python.exe -m tools.evaluate_recognition --faces data/faces \
  --detector models/face_detection_yunet_2023mar.onnx \
  --embedder models/face_recognition_sface_2021dec.onnx \
  --hold-out mallory
```

```
data/faces/
  alice/  img01.jpg img02.jpg img03.jpg ...
  bob/    img01.jpg ...
```

It enrols the first `--enroll` images per person, tests the rest, and reports:

| Metric | Why it matters |
|---|---|
| Accuracy | overall correctness — least useful on its own |
| **False accept rate** | a stranger mistaken for an enrolled person — the *dangerous* error |
| **False reject rate** | a known person called unknown — the *annoying* error |
| Wrong identity | matched, but to the wrong person |
| Threshold sweep | the two rates trade against each other |

`--hold-out` names people to deliberately *not* enrol, which is the only way
to measure false accepts at all.

**Pick `threshold` from the sweep, not from the default.** The right value
depends on your cameras, lighting and how many people are enrolled. Raising it
trades false accepts for false rejects: for access control favour a low FAR,
for convenience favour a low FRR.

---

### Unknown-person alerts

`unknown_person` is **not a visual detector** — there is no "unknown person" in
pixels (Agent.md §13). It is a conclusion drawn from detection + recognition +
the store, and is exactly as reliable as those are. Two guards make it usable:

- `min_sightings` — a single frame of a half-turned head is not evidence of an
  intruder, and recognition is least reliable in exactly those moments.
- `cooldown_seconds` — one person in view must not produce an event per frame.
  Without it a live camera generates thousands an hour.

The event carries `closest_similarity`, so "nobody like them" is
distinguishable from "almost matched someone".

---

## Dashboard

```bash
.venv/Scripts/python.exe -m backend.main --no-window --dashboard
```

Then open <http://localhost:8080>. Live annotated video, which modules are
running and what each costs, the current frame's results, and a live event
feed with severity colouring.

```yaml
dashboard:
  enabled: false
  host: "127.0.0.1"   # NO AUTHENTICATION - see below
  port: 8080
  stream_fps: 15.0    # caps JPEG encoding only; analysis runs at full rate
  jpeg_quality: 75
  max_events: 200
```

**It is read-only.** It observes the pipeline and cannot reconfigure or stop
it — there is no endpoint that mutates anything, and a test asserts that. A
browser tab must not be able to disturb a running camera.

**It has no authentication.** It therefore binds to `127.0.0.1` by default. If
you set `host: 0.0.0.0` it is reachable by anyone on the network, including
the live camera feed; a warning is logged when you do. Put it behind a reverse
proxy with auth before exposing it.

Implementation notes:

- The dashboard is a `BaseOutputSink` (`WebSink`) plus a small FastAPI server
  on a daemon thread. **The pipeline did not change to accommodate it** — Phase
  1 predicted the dashboard would be "another implementation of the same
  interface", and it was.
- `MultiSink` fans frames to several sinks, so the preview window and the
  dashboard can run together.
- Video is MJPEG: a browser renders it with a plain `<img>`, there is no
  client-side buffer to drift behind, and it needs no JavaScript.
- Only the latest frame is kept. Queueing would drift further behind the
  longer it ran — the same reasoning as RTSP stale-frame dropping.
- `stream_fps` throttles **encoding only**. Results and events are published
  every frame regardless; dropping an event to save bandwidth would be a bug.

Measured with detection, tracking and zone analysis running: ~29 FPS with the
dashboard streaming.

Not included: camera management, module selection and config editing from the
browser. Those need the pipeline to be reconfigurable at runtime, which is a
larger change than a read-only view.

---

## Events and restricted zones

Detections and tracks describe *what is in the frame*. An **event** describes
*what happened* — someone crossed into a restricted area and stayed four
seconds. Analysis modules produce events; they publish them to the
`FrameContext` and stop there. Notifying, escalating and recording are
downstream concerns (Agent.md §16).

```yaml
restricted_zone:
  enabled: true
  sources: [person_tracking]      # any tracking modules; sets the dependency
  reference_point: bottom_center  # bottom_center (ground plane) | center
  min_frames: 2                   # frames of agreement before a state flips
  dwell_seconds: 10               # emit zone_dwell after this long; 0 = off
  severity: warning               # info | warning | critical
  classes: []                     # optional label allowlist
  zones:
    - name: loading-bay
      polygon: [[0.55, 0.35], [0.98, 0.35], [0.98, 0.98], [0.55, 0.98]]
```

Emits `zone_entry`, `zone_dwell` and `zone_exit`, each carrying the subject
(track id and class), the zone, and for exits the dwell duration.

Design decisions worth knowing:

- **Polygons are fractions of the frame (0.0–1.0), not pixels.**
  `target_width` changes the frame size and cameras get reconfigured; a zone
  drawn in pixels would silently point somewhere else after either. Pixel
  coordinates are rejected at startup rather than accepted and misread.
- **It consumes tracks, not detections.** Without a persistent id you cannot
  tell "one person stayed a minute" from "sixty separate entries", and every
  event would fire once per frame.
- **`reference_point: bottom_center` is where the feet are** — correct for a
  zone drawn on the ground. A tall person can have their feet inside and their
  midpoint outside; use `center` for overhead cameras.
- **Entry and exit are debounced** (`min_frames`). A box jittering on the
  boundary would otherwise emit an entry/exit pair every frame.
- **A lost track closes its visit.** If the tracker loses someone who was
  inside, an exit fires with `reason: track_lost` — otherwise they would be
  counted as present forever.
- **`sources` sets the dependency.** Watching `vehicle_tracking` makes that
  the required module; enabling the zone without it is rejected at startup.

Cost is ~0.04 ms per frame — it is arithmetic, not inference.

---

## Detection modules

Five detectors share one base
([detection/base.py](computer_vision/modules/detection/base.py)): run a model, keep the
labels this module owns, publish them under its own name. Each keeps its own
config block, enable switch and `FrameContext` namespace — sharing plumbing is
not merging responsibilities.

Three work with the stock COCO model:

```yaml
person_detection:   { enabled: true,  model: yolo11n.pt }
vehicle_detection:  { enabled: false, labels: [car, truck, bus, motorcycle, bicycle] }
object_detection:   { enabled: false, labels: [] }   # empty = all 80 COCO classes
crowd_detection:    { enabled: false, threshold: 5 } # runs no model
```

Three need custom-trained weights — **no stock COCO model contains helmet,
vest, fire or smoke classes**, and pretending otherwise would be worse than
useless in a safety system. Enabling one without a `model:` fails at startup
with instructions rather than silently detecting nothing:

```
Module 'fire_detection' needs a model that can detect ['fire'], and no stock
COCO model contains those classes.
Set modules.fire_detection.model to a path for a custom-trained detector...
```

Set `labels` to match your model's own class names — PPE datasets disagree
(`helmet` vs `hardhat`, `vest` vs `safety-vest`).

### Cost, and model sharing

Detectors configured with the same model **share one load and one inference
per frame**. Agent.md §9 requires this — without it, `person_detection` and
`vehicle_detection` on the same weights run the identical backbone over the
identical pixels twice.

Measured on CPU, three detectors on `yolo11n.pt` at `imgsz: 416`:

| | FPS | frame latency | per-module |
|---|---|---|---|
| `share_model: false` | 9.0 | 87.1 ms | person 28.3, vehicle 28.4, object 29.7 |
| `share_model: true` (default) | **19.3** | **26.7 ms** | person 26.1, vehicle 0.01, object 0.01 |

**2.15× faster, and the results are byte-identical** — verified by comparing
every detection from both modes.

The second and third detectors become nearly free because they reuse the first
one's inference and then filter it down to their own labels. Sharing only
applies when the settings that change what a model *produces* match (weights,
device, `imgsz`, `iou`); labels and confidence thresholds are applied
afterwards and never block it.

Class selection is done by label rather than by restricting the model to class
ids, because measurement showed the restriction saves nothing (+0.2%, within
noise — YOLO applies it after NMS) while preventing sharing entirely.

Modules that consume another module's output (`crowd_detection`,
`person_tracking`) run no model and cost ~0.02 ms regardless.

Opt a module out with `share_model: false`.

### Detection vs. decision

Detectors report what they see; they never decide what it means.
`ppe_detection` reports "a helmet is here", not "this worker is in violation" —
pairing PPE with a person and judging that against a site rule is Phase 9
logic. `fire_detection` reports candidates, because sunsets, brake lights and
high-vis clothing all read as fire; confirming over time belongs upstream.

---

## RTSP cameras (production)

Keep real credentials in `.env`, never in `config.yaml`:

```bash
RTSP_URL=rtsp://user:password@192.168.1.64:554/Streaming/Channels/101
```

```bash
.venv/Scripts/python.exe -m backend.main --rtsp-url "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
```

An RTSP stream fails in ways a webcam and a file never do, so three things are
handled explicitly:

**Freshness over completeness.** A background thread drains the socket and
keeps only the newest frame. Without it, a pipeline processing at 14 FPS
against a 25 FPS camera falls further behind every second and the "live" view
drifts minutes into the past. VisionGuard drops stale frames instead and
reports the count via `frames_dropped`, so a slow pipeline is visible rather
than silent. *If you need every frame, record to disk and use the file source.*

**Reconnect with backoff.** A dropped stream reopens automatically. The delay
doubles per consecutive failure (capped by `reconnect_max_delay_sec`) so a
rebooting camera is not hammered, and the budget resets after any successful
frame.

**Timeouts that actually fire.** An unreachable camera fails in exactly
`open_timeout_sec` — verified at 3.0 s and 6.0 s against a real unroutable
host.

```yaml
video:
  source: rtsp
  rtsp_url: null           # set via RTSP_URL in .env
  rtsp_transport: tcp      # tcp = reliable; udp = lower latency, lossy
  open_timeout_sec: 10.0
  read_timeout_sec: 5.0
  reconnect_attempts: 5    # 0 disables reconnect
  reconnect_delay_sec: 2.0
  reconnect_max_delay_sec: 30.0

processing:
  stall_timeout_sec: 30.0  # stop if no frame arrives for this long; 0 disables
```

### Credentials

Camera URLs contain real, frequently reused passwords. Every path that could
print one is redacted to `user:***` — log lines, error messages, `repr()`, the
source description and `stats_snapshot()`. There are tests asserting the
password never appears in captured log output, and that a URL with a password
but *no* username (`rtsp://:secret@host`) is redacted too.

---

## Video files, and why they matter

A webcam shows a different scene every run, so no two measurements are
comparable. A file gives identical frames every time — which is what makes
*"did that parameter change actually help?"* an answerable question.

```yaml
video:
  source: file
  path: "data/samples/office.mp4"
  loop: false           # restart at the end instead of stopping
  start_frame: 0        # seek here before the first frame
  playback_speed: 0.0   # 0 = as fast as possible; 1.0 = real time
```

Verified: two runs of detection + tracking over the same clip produced
**byte-identical analysis** — same detections, same boxes, same track ids on
every one of 45 frames.

Behaviour worth knowing:

- **A file ends.** EOF raises `SourceExhausted`, which the pipeline treats as a
  clean stop, not an error. Exit code stays 0.
- **EOF is not retried.** A finished file stops immediately rather than
  stalling through `read_retries` with sleeps, unlike a live camera where a
  failed read is worth retrying.
- **`playback_speed: 0` is the default** because benchmarking wants maximum
  throughput. Set `1.0` to watch it at a natural rate.
- **Pacing has no drift.** Frames are timed against a fixed origin, so a slow
  frame does not push every later one late.
- **A corrupt file cannot loop forever.** If a rewind yields no decodable
  frame, the source gives up with a clear message.

Everything above is a property of the *source*. No AI module changed to support
files — `PersonDetectionModule` and `PersonTrackingModule` are byte-for-byte
what they were in Phases 2 and 3.

---

## Tracking

A tracker runs no detector. It reads what its detector published to the
`FrameContext` and writes tracks back:

```
person_detection  -> context.detections["person_detection"]
                  -> person_tracking reads it
                  -> context.tracks["person_tracking"]
```

Three trackers share one base
([tracking/base.py](computer_vision/modules/tracking/base.py)) and one algorithm, differing
only in which detector they consume:

| Tracker | Consumes |
|---|---|
| `person_tracking` | `person_detection` |
| `object_tracking` | `object_detection` (multi-class) |
| `vehicle_tracking` | `vehicle_detection` |

**Id spaces are per-module.** With person and vehicle tracking both enabled
there is a `#1` person *and* a `#1` vehicle — different things, different
namespaces, no collision. Box colours mix the module name into the palette so
the two never look alike.

`object_tracking` handles several classes in one id sequence (`#1 person`,
`#2 car`, `#3 person`), because an id identifies a thing rather than a
thing-of-a-class. The algorithm never associates across labels, so a car cannot
inherit a person's id even at perfect overlap.

Vehicles usually want looser association than people — they move further
between frames — hence the different defaults in `config.yaml`
(`iou_threshold: 0.25`, `max_age: 45`).

The algorithm ([tracker_core.py](computer_vision/modules/tracking/tracker_core.py)) is
SORT-style association by intersection-over-union against a linearly predicted
box — no Kalman filter, no scipy, no appearance model. At video frame rates
that is enough, it costs ~0.02 ms per frame, and it keeps the dependency
footprint at zero.

```yaml
person_tracking:
  enabled: true
  iou_threshold: 0.3      # minimum overlap to continue a track
  max_age: 30             # frames a track survives unmatched (occlusion)
  min_hits: 2             # matches before a track is reported
  publish_coasting: false # report tracks not seen this frame
```

Behaviour worth knowing:

- **Ids are never reused.** A track that dies takes its id with it, so gaps in
  the sequence (1, 2, 4, ...) are correct, not a bug.
- **`min_hits` trades latency for stability.** At 2, a one-frame false positive
  is never reported, at the cost of one frame of delay before a new person
  appears.
- **`max_age` spans occlusions.** An unmatched track coasts along its last
  velocity and can be re-acquired with its original id. Raise it for scenes
  with pillars or crossing crowds; lower it if ids stick to the wrong person.
- **Coasting tracks are hidden by default.** A coasting box is a prediction,
  not an observation. Enable `publish_coasting` and they are drawn thinner and
  marked `?`.
- **Classes are never crossed.** A car cannot inherit a person's id even at
  perfect overlap.

The tracker is generic, so Phase 8's object and vehicle trackers reuse it by
overriding two class attributes.

---

## Detection backend

`PersonDetectionModule` does not know what a neural network is. It depends on
[`DetectorBackend`](computer_vision/inference/base.py) — `load()`, `infer(frame) ->
list[RawDetection]`, `unload()` — and the backend is chosen by config:

```yaml
person_detection:
  backend: ultralytics
  model: yolo11n.pt
```

**Licensing:** `ultralytics` and its pretrained weights are **AGPL-3.0**. That
is fine for development and for open-sourced deployments, but it carries
source-disclosure obligations for a closed-source product. If VisionGuard is
going to be commercial, plan to swap this backend.

Swapping is cheap by design: add `computer_vision/inference/onnx_backend.py` implementing
`DetectorBackend`, register it in
[computer_vision/inference/factory.py](computer_vision/inference/factory.py), and set `backend: onnx`. An
ONNX Runtime or OpenCV-DNN backend also removes the torch dependency entirely
(~1.2 GB). **No AI module, pipeline, source or config-schema change is
involved** — that is what the abstraction is for.

---

## Adding an AI module (the recipe Phase 2 followed)

1. Create the module file with a `BaseAIModule` subclass defining
   `module_name`, `module_type`, `version`, `requires`, `initialize()` and
   `process()`.
2. Add its line to `BUILTIN_MODULES` in
   [computer_vision/modules/registry.py](computer_vision/modules/registry.py).
3. Add its config block with `enabled: true` in `configs/config.yaml`.

No change to the pipeline, the sources, the context, the output sinks or
`main.py`. Phase 2 is the worked example: it touched exactly those three
places plus the drawing helper.

---

## Switching the video source

Development:

```yaml
video:
  source: webcam
  device_id: 0
```

Evaluation:

```yaml
video:
  source: file
  path: "data/samples/office.mp4"
```

Production (Phase 5), with no change to any AI module:

```yaml
video:
  source: rtsp
  rtsp_url: "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"
```

All three are implemented. The AI modules cannot tell which one is running.

---

## Troubleshooting

**Camera takes ~20 seconds to open (Windows).**
Expected with the MSMF backend. It was chosen deliberately: on the reference
hardware MSMF sustains 30.8 fps at 720p while DirectShow was stuck on an
uncompressed format at 10.0 fps. Startup is paid once; frame rate is paid every
frame. For a fast start at a lower frame rate:

```yaml
video:
  backend: dshow
```

**Low FPS.** Measure the source first:

```bash
.venv/Scripts/python.exe -m backend.main --no-window --max-frames 60
```

Compare `avg capture wait` against `avg processing latency`. A high capture
wait means the camera is the bottleneck — try `backend`, a lower `width`/
`height`, or `fourcc: MJPG` (which helps on some cameras and hurts on others,
hence it is opt-in).

**"Could not open webcam device 0".** Another application may hold the camera;
try `--device-id 1`; check Windows camera privacy settings.

**"Unknown AI module 'x'".** Working as intended: the name is misspelled, or
that module belongs to a later phase. Run `--list-modules`.

**Detection is slow.** Check `module_latency_ms` in the final stats line. On
CPU, `person_detection` costs ~45 ms per frame at `imgsz: 640`. Lower `imgsz`
(see the table above), or set `device: cuda` if you have a CUDA GPU — note the
default `torch` wheel installed here is CPU-only, so a GPU needs the CUDA build
from pytorch.org.

**Two detectors, both slow.** They are not sharing a model. Sharing needs
identical `model`, `device`, `imgsz` and `iou`; check for a stray difference,
and that neither sets `share_model: false`. When sharing works, only the first
detector shows a real latency and the rest read ~0.01 ms.

**"The 'ultralytics' package is not installed".** Person detection is enabled
but its backend is missing. Either `pip install ultralytics`, or set
`person_detection.enabled: false` to run the pipeline without detection.

**No window appears.** A headless OpenCV build falls back to headless mode with
a warning. Use `--no-window` to make that explicit.

**RTSP: "Could not open RTSP stream".** Check reachability first
(`ping`, then `ffprobe <url>`). If ffprobe works and VisionGuard does not, try
`rtsp_transport: udp`. The error message masks the password, so it is safe to
paste into a ticket.

**RTSP: high `frames_dropped`.** Not a fault — it means the camera delivers
faster than the pipeline processes, and stale frames are being discarded to
keep the view live. Lower `imgsz`, raise `target_width` downscaling, or use a
GPU. The alternative (queueing) would grow latency without bound.

**RTSP: repeated reconnects.** The stream is unstable. `tcp` transport is more
robust than `udp` on a congested network; raise `reconnect_max_delay_sec` to
back off harder from a camera that is rebooting.

**Pipeline stops with "No frame received for ...".** `stall_timeout_sec`
fired: the stream went silent without closing. Raise it for a camera with long
legitimate gaps, or set it to 0 to disable.

---

## Project documents

`Agent.md` plus `docs/PROJECT_SPEC.md`, `docs/ARCHITECTURE.md`,
`docs/MODULE_SPEC.md` and `docs/ROADMAP.md` are the authoritative specification. Where code and those
documents disagree, the documents win.
