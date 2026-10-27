# Re-identification model

`person_reid_youtu_2021nov.onnx` was downloaded from the official
[OpenCV Zoo YoutuReID folder](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/person_reid_youtureid),
pinned at commit `48b10fa505aeb51fbd127fc60a5c660040e823b7`.

Size: 106,878,407 bytes. SHA-256 (verified against the upstream Git LFS pointer):
`0579683334d4b9440221606dcb461656dd0dc64143b18f48faedaced9b4f580d`.
The upstream model folder supplies an Apache-2.0 license.

The existing OpenVINO `PersonEmbedder` runs it on CPU. Enabling Re-ID does not
move the other CUDA detectors to CPU. `person_reid` remains opt-in and requires
person detection and tracking.

The configured shared gallery is `data/reid/gallery.npz`, relative to the
repository working directory. Camera workers must use the same absolute file.
The benchmark deliberately substitutes its own gallery under the output folder.
No synthetic identities are pre-populated. This is an appearance gallery, not
the enrolled face registry; similar clothing can cause wrong matches.
