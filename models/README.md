# models/

**Every AI model weight file lives here**, and nowhere else. Code that *uses*
a model lives in `computer_vision/inference/`; this directory holds only the
binaries.

```
models/
├── detection/   yolo11n.pt              person, object, vehicle, PPE, fire, smoke
├── pose/        yolo11n-pose.pt         body keypoints
├── face/        *.onnx, identities.npz  face detection, recognition, enrolled identities
└── custom/      your own .pt files      PPE, fire, smoke - not in any stock model
```

Weights are **git-ignored** (`.gitignore`: `*.pt`, `*.onnx`, `*.engine`).
They are downloaded or trained, never committed — a repository is for source,
not for hundreds of megabytes of binaries.

## What downloads automatically

`yolo11n.pt` and `yolo11n-pose.pt` are fetched by Ultralytics on first run,
directly into the folders above because `config.yaml` names the full path.
Nothing is left in the project root.

## What you must supply

| File | Goes in | Where from |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` (~340 KB) | `face/` | [opencv_zoo/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `face_recognition_sface_2021dec.onnx` (~37 MB) | `face/` | [opencv_zoo/face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |
| PPE / fire / smoke weights | `custom/` | trained by you — no stock model has these classes |

Enabling a module without its model fails at startup with the URL, rather than
running and silently detecting nothing.

## `identities.npz` is personal data

`face/identities.npz` holds face embeddings of enrolled people. It is not an
image gallery, but it is still personal data: back it up deliberately, control
access to it, and delete entries on request with `store.remove(name)`.
