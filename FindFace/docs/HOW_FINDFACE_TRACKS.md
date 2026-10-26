# How FindFace Multi detects and tracks faces — with evidence

Researched 2026-10-19. FindFace is closed-source, so this is reconstructed from
NtechLab's **own public documentation**, not from decompiling anything. Each
claim cites the page it comes from. Inferences are marked as such.

## Sources

| # | Source | What it covers |
|---|--------|----------------|
| S1 | [ntechlab.com/findfacemulti](https://ntechlab.com/findfacemulti) | Marketing page: capabilities, "more than 50 neural network models", feature vectors, webhooks/REST |
| S2 | [FindFace Multi 1.1 — findface-video-manager and findface-video-worker](https://docs.ntechlab.com/projects/ffmulti/en/1.1/video-config.html) | Architecture + every tracker/filter parameter with defaults |
| S3 | [FindFace Multi 2.3 — Video Sources](https://docs.ntechlab.com/projects/ffmulti/en/2.3/video_sources.html) | Per-camera UI settings (offline/real-time modes, ROI/RoT, overlap threshold, track length) |
| S4 | [FindFace SDK — About Video Face Detection](https://miracle-documentation.readthedocs.io/en/latest/fkvideo_about.html) | Older `fkvideo_detector`: motion gate → tracker → best snapshot, the two posting modes |
| S5 | [FindFace SDK — fkvideo_detector configuration](https://miracle-documentation.readthedocs.io/en/latest/fkvideo_configuration.html) | Older CLI flags: `--min-score`, `--min-dir-score`, `--md-threshold`, `--roi`, `--rot`, `--max-persons` |
| S6 | [FindFace Security 4.0.1 — video-config](https://docs.ntechlab.com/projects/ffsecurity/en/4.0.1/video-config.html) | Example config: `overall_only: true`, `realtime_post_interval: 1` |

## The pipeline

```
camera / file ──► findface-video-manager ──(job: stream + settings)──► findface-video-worker (CPU or GPU)
                                                                          │
     1. decode frame (ffmpeg)                                             │
     2. motion detector  ── no motion ──► skip frame                      │
     3. object detector (face / body / car) inside RoT                    │
     4. tracker: link detections across frames by bbox overlap            │
     5. per-detection filters: quality, size, angle, ROI                  │
     6. best-snapshot selection (offline or real-time mode)               │
                                                                          ▼
                          POST snapshot + full frame + bbox ──► findface-multi backend
                                                                 7. extractor: feature vector
                                                                 8. match against watchlists → event / webhook
```

### 1. Split into a manager and stateless workers
*"findface-video-manager"* hands out jobs containing stream data and settings,
and spreads streams over idle workers; *"findface-video-worker"* does detection
and tracking and posts snapshots, full frames and metadata onward (S2). Workers
come in `-cpu` and `-gpu` builds, with `capacity`, `batch_size`, `resize_scale`
and `min_size` knobs (S2).

### 2. A motion gate before the detector
Motion detection comes first *"to reduce system resources consumption"*, and the
face tracker starts only on motion of sufficient intensity (S4). The parameter is
`imotion_threshold`: default `0.002`, `0` disables it, and `0.05` is noted as
too high to catch most motion (S2; older name `--md-threshold`, S5).

### 3. Detection restricted to a region of tracking (RoT)
RoT limits detection and tracking to a rectangle, and nothing outside it is
analysed (S3; `--rot WxH+X+Y`, S5). Detection uses neural networks. S1 claims
50+ models overall. Which detector architecture is used is **not published**.

### 4. Tracking by bbox overlap with a miss timeout
- **Association:** an *"Overlap threshold"* (0–1) on bbox overlap between
  consecutive frames decides whether a detection continues a track (S3).
  → This is an IoU-style tracker (inference from the parameter's definition).
- **Track ends:** `track_miss_interval`, default **1 s** without a new detection (S2).
- **Forced split:** `track_max_duration_frames`, default 0 = off (S2).
- **Gaps filled:** `track_interpolate_bboxes`, default **true**, rebuilds missed
  bboxes between detections (S2).
- **Path output:** `track_send_history` sends the bbox array with the event (S2, S3).
- **Minimum track length** before posting: default 1 frame (S3).
- **Concurrency cap:** `--max-persons`, which strongly affects performance (S5).

### 5. Per-detection filters
| Filter | FindFace name | Default |
|---|---|---|
| image quality | `filter_min_quality` (UI "Quality", 0–1) | per object type, reference **0.42** (S3) |
| min / max size | `filter_min_size` / `filter_max_size` | 1 / 8192 px (S2) |
| head pose | `--min-dir-score` (max deviation from frontal) | −1000 (S5) |
| region of interest | `roi` | unset. An object is posted once it has entered the ROI, then tracked until it leaves (S2, S3) |

### 6. Best-snapshot selection — the key idea
FindFace does not post every detection. While tracking a face, *"the face tracker
searches for its best snapshot"* (S4). There are two modes:

- **Offline (`overall_only`)**: the tracker buffers while the face is visible.
  When the face disappears, it posts only the highest-quality snapshot (S4).
  Docs give the default as true on CPU and false on GPU (S2). The UI describes
  it as one best snapshot per track, to save disk space (S3).
- **Real-time**: post when quality clears a threshold, and the threshold rises
  after each post, so only better snapshots follow (S4). In the modern form:
  `realtime_post_interval` (default 1 s) windows,
  `realtime_post_every_interval` (post each window's best, or only when quality
  increases), and `realtime_post_first_immediately` (S2, S3).
- Older de-duplication knobs (`--post-uniq`, `--uc-max-time-diff`, `--uc-max-dup`)
  limited posts per person per time window (S5).

### 7–8. Recognition happens after tracking
The posted best snapshot gets a feature vector, which is compared against stored
vectors to decide on a match (S1). Results go out through webhooks / REST (S1).
**Tracking is therefore what keeps the costly recognition step cheap:** one
embedding per track instead of one per frame.

## How findface-lite maps onto this

| FindFace stage | findface-lite | Notes |
|---|---|---|
| video-worker | `findface_lite/pipeline.py` | single process; no manager/worker split yet |
| motion gate `imotion_threshold` | `motion.py` | frame differencing, same default 0.002 |
| neural face detector | `detector.py` (OpenCV **YuNet**, 230 KB ONNX, CPU) | gives 5 landmarks |
| quality score | `quality.py` | **not** FindFace's net: transparent mix of confidence, sharpness, size and frontalness |
| overlap tracker + miss interval + interpolation | `tracker.py` | adds constant-velocity prediction |
| filters, ROI / RoT | `pipeline.py::_passes_filters` | same semantics, same `WxH+X+Y` syntax |
| offline / real-time posting | `snapshot.py` | same parameter names and defaults |
| webhook | `sinks.py::WebhookSink` | JSON with base64 JPEGs |
| feature vectors + watchlists | **not built yet** | next step: see README |
