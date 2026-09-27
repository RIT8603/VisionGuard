"""Model backend wrappers.

This package holds thin adapters around inference runtimes (Ultralytics,
ONNX Runtime, OpenCV DNN, ...) so that an AI module depends on a small local
interface rather than on a third-party API directly. Swapping a detector's
backend then changes one file, not the module.

Note the distinction:
    computer_vision/inference/  -> code (this package)
    models/      -> weight files on disk, at the repository root

Empty in Phase 1; populated from Phase 2 onward.
"""
