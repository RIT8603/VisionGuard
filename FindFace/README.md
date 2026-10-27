# findface-lite

Face detection and tracking modelled on NtechLab FindFace Multi's video worker.
A motion gate feeds a face detector. Detections are linked into tracks, every
face gets a quality score, and **one best snapshot per track** is posted to a
folder or a webhook. With a watchlist, each posted face is embedded (OpenCV
SFace) and matched against enrolled photos, as FindFace does after tracking.

How FindFace does it, with sources: [docs/HOW_FINDFACE_TRACKS.md](docs/HOW_FINDFACE_TRACKS.md).

## Setup

```bash
py -3 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Both models sit in `models/`, from OpenCV Zoo:
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (detector, MIT) and
[SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
(`face_recognition_sface_2021dec.onnx`, recognition, Apache-2.0).

## Run UI & Demos

### 1. Interactive Web Dashboard (Recommended for Presentations)

Launch a browser dashboard featuring live annotated video streaming, active track stats, and real-time best-face snapshot cards with quality metrics:

```bash
.venv/Scripts/python run_ui.py
```
*Automatically opens in your browser at `http://localhost:5000` (or toggle to your webcam inside the UI).*

### 2. Desktop Window Preview (`--show`)

```bash
# video file with live preview window and continuous loop
.venv/Scripts/python -m findface_lite samples/demo.mp4 --show --loop

# webcam with live preview window, real-time posting
.venv/Scripts/python -m findface_lite 0 --show --mode realtime

# match every posted face against a folder of enrolled photos
.venv/Scripts/python -m findface_lite samples/demo.mp4 --out output/demo --watchlist watchlist/

# RTSP camera, ROI, webhook
.venv/Scripts/python -m findface_lite rtsp://user:pass@cam/stream --set roi=640x360+320+180 --webhook http://localhost:8000/events
```

Each event is a line in `events.jsonl`, with a face crop in `faces/` and the
full frame in `frames/`:

```json
{"event_id": 1, "track_id": 1, "reason": "overall_best", "frame_idx": 54, "timestamp": 2.16,
 "bbox": [613.7, 236.4, 686.8, 341.8], "landmarks": [[645.2, 277.0], ...],
 "quality": {"score": 0.865, "confidence": 0.896, "sharpness": 0.952, "size": 0.653, "frontal": 0.937, "yaw": 14.0, "pitch": -4.0},
 "track": {"first_frame": 0, "last_frame": 74, "hits": 63, "closed": "miss_interval"},
 "match": {"name": "alice", "similarity": 0.9026},
 "face_path": "faces/1_face.jpg", "full_frame_path": "frames/1_full_frame.jpg"}
```

## Recognition (watchlist)

`--watchlist DIR` turns on the recognition stage. On startup every photo in
`DIR` (`.jpg .jpeg .png .bmp .webp`) is enrolled: the largest face is
detected, aligned and embedded. The person's name is the file name without
extension, so `alice.jpg` enrols "alice". Photos with no face are skipped
with a warning. To add several photos of one person, give them the same stem
in different formats, or enrol them from code with `Watchlist.add`.

For each posted best face, the pipeline:

1. aligns the face with `cv2.FaceRecognizerSF.alignCrop`, using the event's bbox and 5 landmarks on the full frame,
2. computes a 128-d SFace embedding (L2-normalised),
3. takes the cosine similarity against every enrolled vector and keeps the best one.

If the best similarity is at least `match_threshold` (default 0.363, the
SFace cosine threshold from OpenCV Zoo), the event gets
`"match": {"name": ..., "similarity": ...}`. Otherwise it gets `"match": null`.
Without `--watchlist`, events have no `match` key.

Recognition is a pipeline hook (`FacePipeline(..., hooks=[recognizer])`).
It runs after best-snapshot selection and before the sinks, so the tracker is
unchanged, and every sink (folder, webhook, callback) receives the match.

## Configuration

Every field in [`findface_lite/config.py`](findface_lite/config.py) can be set
in a JSON file (`--config`) or with `--set key=value`. Field names follow
FindFace's own parameters:

| key | default | meaning |
|---|---|---|
| `imotion_threshold` | 0.002 | motion needed to run the detector (0 = always run) |
| `filter_min_quality` | 0.42 | min quality score 0..1 |
| `filter_min_size` / `filter_max_size` | 40 / 8192 | face size limits in px |
| `filter_max_angle` | 45 | max estimated yaw/pitch, degrees |
| `roi` / `rot` | – | `WxH+X+Y`. Post only after entering the ROI; detect only inside the RoT |
| `track_overlap_threshold` | 0.25 | IoU needed to continue a track |
| `track_miss_interval` | 1.0 | seconds without a detection before a track closes |
| `track_max_duration_frames` | 0 | force-close long tracks (0 = off) |
| `track_min_length` | 1 | frames before a track may post |
| `track_interpolate_bboxes` | true | fill missed frames in the track history |
| `track_send_history` | false | include the bbox path in events |
| `overall_only` | true | offline mode: one post per track, at track end |
| `realtime_post_interval` | 1.0 | real-time window, seconds |
| `realtime_post_every_interval` | false | post each window's best (vs only on improvement) |
| `realtime_post_first_immediately` | false | post the first good face at once |
| `recognizer_model` | `models/face_recognition_sface_2021dec.onnx` | SFace ONNX file |
| `match_threshold` | 0.363 | min cosine similarity to report a watchlist match |
| `post_embedding` | false | include the 128-d embedding in each event |

## Use as a library

```python
from findface_lite import Config, FacePipeline, CallbackSink, Recognizer, run_video

cfg = Config(overall_only=True)
rec = Recognizer.from_config(cfg, "watchlist/")          # optional
pipe = FacePipeline(cfg, [CallbackSink(lambda ev, imgs: print(ev["track_id"], ev.get("match")))], hooks=[rec])
run_video(pipe, "0")   # webcam
```

## Tests

```bash
.venv/Scripts/python -m pytest -q
```

Unit tests cover the tracker and snapshot policies. End-to-end tests build a
synthetic video (a face crosses twice, with blur dropouts) and check: 2 tracks,
1 post per track, the sharp frame chosen, the motion gate skipping frames, and
ROI behaviour. Recognition tests enrol `tests/data/face.jpg` and check that
every event from `samples/demo.mp4` matches it (also through the CLI). They also
check that a threshold set too high gives `match: null`, and that enrolment
skips photos with no face.

## Not built yet (FindFace has these)

- Recognition at scale: an indexed vector database (FindFace uses its own
  sharded index), cards with several photos and metadata, and per-watchlist thresholds.
  Matching here is a brute-force dot product, which is fine up to tens of thousands of faces.
- Manager/worker split for many cameras, and GPU batching.
- A learned quality model in place of the heuristic score.
