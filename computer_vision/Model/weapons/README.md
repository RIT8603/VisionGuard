# Weapon models

Used by `weapon_detection` (see `configs/config.yaml`), next to the shared COCO
`yolo11n.pt`, which covers baseball bats and scissors. The `.pt` files are
git-ignored like every other weight file; this README records where they come
from and how they were checked.

| File | What it detects | Source / license | SHA-256 |
|---|---|---|---|
| `knife_yolo11n_sohas.pt` (5.5 MB) | knife, pistol, smartphone, purse, banknote, card (VisionGuard uses **knife**) | Fine-tuned here from `yolo11n.pt` on the Sohas weapon dataset ([ari-dasci/OD-WeaponDetection](https://github.com/ari-dasci/OD-WeaponDetection), CC BY-SA 4.0) | `7ff962adc5ac96748f6d00a269cd2182dc2f3302e9dd75f7f0ce626776ac38fa` |
| `firearm_gun_dtct_bg1k.pt` (22.5 MB) | one class, `"0"` = gun (published as `firearm`) | [Zcket/gun_dtct](https://huggingface.co/Zcket/gun_dtct), `yolov8_background1k_best.pt`, MIT | `f1b151a061cbc960b7dc3aee4aa5306d7250be05daa4960cc0bf65627e917ef8` |
| `firearm_yolo12_compatible.pt` (18.7 MB) | `long_gun` (published as `firearm`; other classes unused) | [CostiCatargiu/Yolov12_WeaponDetection](https://github.com/CostiCatargiu/Yolov12_WeaponDetection), converted `OriginalModel.pt`; no separate weights license stated upstream | `fd223008ff0ad62217dc6c82e6a66c005b58b23ef65255a941c3130dea382cb5` |

Both are pickle checkpoints. Before loading the downloaded one, its pickle was
scanned statically: it imports only `torch`, `ultralytics`, `set` and
`OrderedDict`.

## Training (knife model)

`yolo11n.pt` → 50 epochs, 640 px, batch 16, seed 7, on the Sohas **train**
split (4,502 images; 500 more held out for validation). The Sohas **test**
split (857 images) was never used for training or threshold choice.

## Measured results (Sohas test split, IoU 0.5)

| Class | Model, threshold | AP@50 | Precision | Recall |
|---|---|---|---|---|
| knife | COCO `yolo11n`, 0.4 | 0.048 | 0.43 | 0.03 |
| knife | fine-tuned, 0.6 | 0.961 | 0.98 | 0.92 |
| pistol | gun_dtct, 0.8 | 0.553 | 0.60 | 0.73 |
| pistol | fine-tuned, 0.6 | 0.919 | 0.82 | 0.89 |

**Caveat - leakage.** Most test images come from the same sources (often the
same videos) as training images, which flatters the fine-tuned model. Checked
two ways:

* **Knife, separate web photos only** (257 images, `knife (N).jpg`): fine-tuned
  recall 0.87 / precision 0.97, COCO 0.05 / 0.41. The knife gain is real.
* **Pistol, sources never seen in training** (158 images, 27 pistols):
  gun_dtct AP@50 0.943, precision 0.89, recall 0.93, 3 of 131 look-alike-only
  images raised an alarm; fine-tuned AP@50 0.856, precision 0.68, 12 of 131.
  So firearms use gun_dtct, not the fine-tuned model.

No test set above has bats or scissors. The rifle CCTV check below is separate
from those pistol/knife measurements.

## Long-gun compatibility and CCTV check (2026-10-26)

The pistol-oriented `gun_dtct` model produced no detections in the 145-frame
`data/videos/evaluation-5.mp4` rifle clip with the previous configuration.
The long-gun model is added alongside it; the pistol and knife models stay intact.

Source: `OriginalModel.pt` at upstream commit
`382d925c0d4ce054c0368ac2778c93b94654954d`, SHA-256
`c3ca821ad15229b5e4992a8ba44a0bd8343458e790e4af00d869d21df7a472e8`.
The checkpoint uses legacy YOLO12 attention with separate `qk` and `v`
projections. They were combined into the installed Ultralytics `qkv` format,
interleaving query/key/value channels per head and preserving batch-normalisation
parameters and the original positional convolution. All eight converted layers
matched the published legacy forward in FP32 within a maximum absolute error of
`4.76837158203125e-07`. No application or installed-library compatibility patch
is needed to load the converted checkpoint.

Full-frame inference generated false positives on ordinary CCTV. Consequently,
only this extra model searches padded person regions, with confidence 0.7,
minimum person height 160 processing pixels, and at most four people per run.
This requires `person_detection`; it intentionally does not cover small/distant
people or unattended firearms. The existing persistence and armed-person
tracking rules still determine events. This clip is a functional regression
case, not an independent benchmark; overlap with upstream training data is unknown.

Actual configured pipeline check: person detection, person tracking and all three
extra weapon models on the RTX 5060, with no fabricated detections:

| Video | Frames | Frames with weapon boxes | Armed-person frames | Events / alerts | Failed frames |
|---|---:|---:|---:|---|---:|
| `evaluation-5.mp4` | 145 | 22 | 111 | `weapon_detected` + `armed_person`, both critical | 0 |
| `student feed-2.mp4` | 300 | 4 | 0 | none | 0 |
| `1000115706.mp4` | 300 | 6 | 0 | none | 0 |
| `License Plate Detection Test.mp4` | 300 | 0 | 0 | none | 0 |

The rifle was linked to person track 1 at frame 35, confidence 0.762. Local HTTP
checks returned 200 for status, events (2), alerts (2) and the generated weapon
snapshot. Transient false boxes remain in the negative clips, so these results
do not establish zero false detections or accuracy on other cameras. The single
attached screenshot's cropped score was 0.561, below the deployed 0.7 threshold:
the accepted behaviour is detecting the rifle in the sequence and maintaining
the armed track, not claiming a positive match on every still frame.
