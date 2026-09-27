"""Model backend wrappers.

This package holds thin adapters around inference runtimes (Ultralytics,
ONNX Runtime, OpenCV DNN, ...) so that an AI module depends on a small local
interface rather than on a third-party API directly. Swapping a detector's
backend then changes one file, not the module.

Note the distinction, since the two sit side by side:
    computer_vision/inference/  -> code (this package)
    computer_vision/Model/      -> the weight files it loads
"""
