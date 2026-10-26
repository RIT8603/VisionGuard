# PPE checkpoint provenance

`ppe_yolov8n_construction.pt` is the trained `models/best.pt` from
[snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection),
pinned at commit `19b73225e5c9b653d4101d4723e591f3e6cae615`.

- Size: 6,251,955 bytes.
- SHA-256: `4d07bbd92ca30d5c12dd67ccf52b2f54f533c9ccfef534284124682ef9f56129`.
- Original class names: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
  Person, Safety Cone, Safety Vest, machinery, vehicle.
- VisionGuard enables only Hardhat, Safety Vest, NO-Hardhat and NO-Safety Vest
  for this detector. The existing COCO detector supplies person tracks.
- The separate `ppe_violation` module uses explicit missing-equipment classes
  and head/torso association. Merely missing a positive gear detection does not
  raise a violation with the provided configuration.
- Both modules remain opt-in. This checkpoint does not detect gloves or boots.

Raw evaluation sample: upstream `source_files/hardhat.mp4`, same commit,
stored locally as `data/videos/ppe_hardhat_public.mp4` (4,143,534 bytes,
SHA-256 `e4ea71aa6ffc460c430debe677e9cecea6e2a1712d45ce26d48ace0a1aa76a83`).
It is staged safety footage, not evidence of accuracy on every CCTV camera.
Upstream declares no separate model license in this pinned repository; do not
infer a deployment license from the fact that a file is publicly downloadable.
The runtime also has its existing Ultralytics license requirements.

Actual evaluation results and limitations belong in the generated benchmark report,
not the upstream training metrics.
