# computer_vision/Model/

**Every AI model weight file lives here**, and nothing else.

It sits inside `computer_vision/` beside the two folders it works with:

| Folder | Holds |
|---|---|
| `../modules/` | **what to do** — the CV features |
| `Model/` | **the weights** — binaries only (this folder) |
| `../inference/` | **how to run them** — runtime adapters |

```
computer_vision/Model/
├── detection/   yolo11n.pt              person, object, vehicle, PPE, fire, smoke
│                yolo11s_int8_openvino_model/  the same classes, larger model, OpenVINO 8-bit (default)
│                yolo11s_openvino_model/       the same model, 16-bit (reference)
├── pose/        yolo11n-pose.pt         body keypoints
├── face/        *.onnx, identities.npz  face detection, recognition, enrolled identities
├── anpr/        *.onnx                  number-plate detector and OCR
├── reid/        *.onnx, gallery.npz     person re-identification (you supply the model)
└── custom/      your own .pt files      PPE, fire, smoke, firearms - not in any stock model
```

Weights are **git-ignored** (`.gitignore`: `*.pt`, `*.onnx`, `*.engine`,
`*.npz`, `*_openvino_model/`). They are downloaded or trained, never committed — a repository is
for source, not for hundreds of megabytes of binaries. The folder layout is
committed via `.gitkeep`, so a fresh clone has somewhere to put them.

## Downloading the YOLO weights (opt-in)

`.pt` weights are Python pickles - loading one executes code from it - so
VisionGuard never downloads them implicitly. A missing weights file is a
startup error.

To fetch `yolo11n.pt` / `yolo11n-pose.pt` once, set `allow_download: true` on
the module in `configs/config.yaml` and start VisionGuard. The file is
downloaded (not loaded) into the folder above, its SHA-256 is checked against
`model_sha256`, and only then is it loaded. Set `allow_download` back to
`false` afterwards.

Pinned hashes (also in `configs/config.yaml`):

| File | SHA-256 |
|---|---|
| `detection/yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `pose/yolo11n-pose.pt` | `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` |

If you deliberately switch to different weights, update `model_sha256` (or
remove the line to skip the check). Never load `.pt` files from untrusted
sources.

## The OpenVINO model (default detector)

`detection/yolo11s_int8_openvino_model/` is YOLO11s exported to OpenVINO IR
at a fixed 544x960 input (a 16:9 frame at `target_width: 960`), with weights
quantised to 8 bits. It is a larger, more accurate model than `yolo11n.pt`.
It needs `pip install openvino`.

Why 8-bit: measured on a 2-core CPU with AVX-512 VNNI (the instruction set
8-bit inference uses, present on Intel Core 11th gen and later), per frame at
960 width:

| Model | Time | Person recall / precision vs yolo11x |
|---|---|---|
| `yolo11n.pt` (PyTorch) | 193 ms | 0.51 / 0.95 |
| `yolo11s_openvino_model` (16-bit) | 201 ms | 0.67 / 0.98 |
| `yolo11s_int8_openvino_model` (8-bit) | 88 ms | 0.67 / 0.97 |

The 8-bit model was calibrated on 317 frames of the project's street test
clip and matched the 16-bit one on other frames of that clip. Calibration on
one scene is a reason to re-check accuracy on a very different camera; the
16-bit model is kept beside it for exactly that comparison.

It is a directory (`.xml` + `.bin` + `metadata.yaml`), not a pickle. Its
`model_sha256` in `configs/config.yaml` is a digest over every file in it:

| Directory | Digest |
|---|---|
| `detection/yolo11s_int8_openvino_model/` | `535501d57695171dce9244c053c4c09950b2576fda2b696950bcb76982ef9303` |
| `detection/yolo11s_openvino_model/` | `30ded5cea4672289ffa1ebe00e9a93bbe561ce51e0286f5923a5e09bc84d71a3` |
| `detection/yolo11s_int8_384_openvino_model/` | `7e8f33bee26e42d5a54fa4c1d1b7e0719b80969361c5f0f446cc80f791333313` |

`yolo11s_int8_384_openvino_model` is the same model exported at 384x640. It
is about 2.2x faster (31 ms against 59 ms per frame on the 2-core test CPU)
and noticeably worse at small, distant people: on the street test clip person
recall against yolo11x fell from 0.64 to 0.47 and precision from 0.97 to 0.93.
At equal computing power it did not track better than the full-size model at
half the frame rate, so it is not the default. Use it for a near-field camera;
`python -m tools.diagnose_speed` times both on your machine.

To recreate it, or to export a different size (a 4:3 camera, another
`target_width`), from the repository root:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='openvino', imgsz=(544, 960), half=True)"
# 8-bit: needs `pip install nncf` and a dataset yaml pointing at ~300 frames from your camera
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='openvino', imgsz=(544, 960), int8=True, data='calib.yaml')"
python -c "from computer_vision.inference.base import model_digest; print(model_digest('yolo11s_openvino_model'))"
```

Move the resulting folder here, then set `imgsz` (the same `[height, width]`)
and `model_sha256` (the printed digest) on each detector that uses it.
`yolo11s.pt` itself comes from the official Ultralytics release assets
(SHA-256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`).

`device: intel:cpu` is the tested setting. `device: intel:gpu` runs it on
Intel integrated graphics instead, which may be faster still - measure it.

## Number-plate models (`anpr/`)

Two small ONNX networks from the MIT-licensed
[open-image-models](https://github.com/ankandrew/open-image-models) and
[fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) projects, run
with OpenVINO. ONNX is data, not a pickle; the hashes are still checked.

| File | SHA-256 |
|---|---|
| `anpr/plate_detector_yolov9t_384.onnx` (7.4 MB) | `888397b96d761c89db40bc9c305838e8652660f5e282c2cadebbe8d2951a77a8` |
| `anpr/plate_ocr_cct_s_v2_global.onnx` (5.0 MB) | `384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b` |

The OCR model reads Latin-alphabet plates. Its training list of regions does
not include India; Indian plates use the same characters, but accuracy on
them has not been measured here.

## What you must supply

| File | Goes in | Where from |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` (~340 KB) | `face/` | [opencv_zoo/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `face_recognition_sface_2021dec.onnx` (~37 MB) | `face/` | [opencv_zoo/face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |
| `person_reid_youtu_2021nov.onnx` (107 MB) | `reid/` | [opencv_zoo/person_reid_youtureid](https://github.com/opencv/opencv_zoo/tree/main/models/person_reid_youtureid) — SHA-256 `0579683334d4b9440221606dcb461656dd0dc64143b18f48faedaced9b4f580d` |
| PPE helmet/vest weights | `custom/` | Configured SafetyVision v2 checkpoint; provenance and limits in `custom/README.md` |
| fire / smoke / other custom weights | `custom/` | Supply a model trained for its configured classes; stock COCO does not contain fire/smoke |

Enabling a module without its model fails at startup with the URL, rather than
running and silently detecting nothing.

## `reid/gallery.npz` is personal data

When `person_reid.gallery_path` is set, appearance vectors of the people seen
are written there so several cameras can share them. They describe clothing
and build rather than a face, but they still single out individuals: the file
is written owner-only, entries expire after `identity_ttl_seconds`, and
deleting the file forgets everyone.

## `identities.npz` is personal data

`face/identities.npz` holds face embeddings of enrolled people. It is not an
image gallery, but it is still personal data: back it up deliberately, control
access to it, and delete entries on request with `store.remove(name)`.

## Paths are relative to the repository root

`configs/config.yaml` refers to `computer_vision/Model/detection/yolo11n.pt`,
resolved from the working directory. Run VisionGuard from the project root, or
give absolute paths in your own config.
