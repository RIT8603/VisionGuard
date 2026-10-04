"""OpenVINO detector backend: a YOLO export run directly, without Ultralytics.

The Ultralytics backend loads an OpenVINO export too, but it brings PyTorch
into the process to do so. Measured on the default model (YOLO11s, 8-bit,
544x960), same detections either way:

==================  =========  ===========  ==============
backend             start-up   memory       per frame
==================  =========  ===========  ==============
``ultralytics``     4.3 s      923 MB       inference + 5 ms
``openvino``        1.1 s      243 MB       inference + 2 ms
==================  =========  ===========  ==============

On a laptop that is also decoding camera streams and running a browser, the
680 MB matters more than the milliseconds: a machine that starts swapping
turns a 50 ms inference into a second.

It reads the directory Ultralytics exports (``*.xml`` + ``*.bin`` +
``metadata.yaml``) and does the three things around the network itself:
letterbox the frame to the model's input, decode the raw output, and suppress
overlapping boxes per class. Only detection exports without built-in NMS are
supported - which is what ``YOLO(...).export(format="openvino")`` produces.

This file imports neither ``ultralytics`` nor ``torch``.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

from computer_vision.inference.base import (
    BackendNotAvailableError,
    DetectorBackend,
    ModelLoadError,
    RawDetection,
    verify_model_file,
)
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

_DEVICE_ALIASES = {
    "cpu": "CPU",
    "intel:cpu": "CPU",
    "gpu": "GPU",
    "intel:gpu": "GPU",
    "auto": "AUTO",
    "intel:auto": "AUTO",
}

#: ``auto`` picks the graphics chip only when it is clearly faster than the
#: CPU: at this fraction of the CPU's time per frame or less.
_AUTO_GPU_ADVANTAGE = 0.8
#: ...and only when it gives the same answer: mean difference between the two
#: devices' class scores on a test picture (scores run 0..1).
_AUTO_MAX_SCORE_DIFFERENCE = 0.02
#: Timed inferences per device when choosing.
_AUTO_TIMED_RUNS = 4

#: Letterbox padding colour (what the model was trained with).
_PAD_VALUE = 114
#: Most boxes kept per frame, as in Ultralytics.
_MAX_DETECTIONS = 300


def resolve_device(device: str) -> str:
    """``intel:gpu`` / ``gpu`` -> ``GPU`` and so on; unknown names pass through upper-cased."""
    name = str(device).strip().lower()
    return _DEVICE_ALIASES.get(name, name.upper())


def _has_device(available: Any, device: str) -> bool:
    """Whether ``device`` (or a numbered one of it, ``GPU.0``) is present."""
    return any(name == device or name.startswith(f"{device}.") for name in available)


def _test_blob(height: int, width: int) -> np.ndarray:
    """A fixed, textured picture for comparing devices (same every run)."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), 3)
    cv2.rectangle(image, (width // 4, height // 4), (width // 2, height // 2), (40, 90, 200), -1)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0


def _time_request(request: Any, blob: np.ndarray) -> Tuple[float, np.ndarray]:
    """Median ms per inference (after one warm-up run) and the class scores."""
    request.infer({0: blob})
    times = []
    for _ in range(_AUTO_TIMED_RUNS):
        started = time.perf_counter()
        request.infer({0: blob})
        times.append((time.perf_counter() - started) * 1000.0)
    scores = np.array(request.get_output_tensor(0).data[0][4:], dtype=np.float32)
    return float(np.median(times)), scores


def letterbox(
    frame: np.ndarray, height: int, width: int
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Fit ``frame`` inside ``height`` x ``width`` without distorting it.

    Returns the padded image, the scale that was applied, and the (left, top)
    padding - everything needed to map a box back onto the frame.
    """
    frame_height, frame_width = frame.shape[:2]
    scale = min(height / frame_height, width / frame_width)
    new_width = int(round(frame_width * scale))
    new_height = int(round(frame_height * scale))

    if (new_width, new_height) != (frame_width, frame_height):
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_x = (width - new_width) / 2.0
    pad_y = (height - new_height) / 2.0
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))

    if (new_width, new_height) == (width, height):
        return frame, scale, (0, 0)

    canvas = np.full((height, width, 3), _PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = frame
    return canvas, scale, (left, top)


class OpenVinoDetector(DetectorBackend):
    """A YOLO detection export, run with the OpenVINO runtime alone.

    Parameters (from the module's config block):
        model: The exported model directory, e.g. ``yolo11s_int8_openvino_model``.
        model_sha256: Optional digest of that directory (see ``model_digest``).
        device: ``intel:cpu`` (default), ``intel:gpu`` for integrated
            graphics, or ``intel:auto``. A device that is not present falls
            back to the CPU with a warning rather than stopping the camera.
            ``intel:auto`` times the model on the CPU and on the graphics
            chip at start-up and keeps the faster one - the graphics chip
            only if it loads, is clearly faster and gives the same scores.
        imgsz: ``[height, width]``. Optional - the export's own size is used;
            when given it must match, so a config pointing at the wrong
            export fails at start-up instead of running slowly or wrongly.
        model_confidence: Score cut inside the backend. Default 0.25; the
            module applies its own, stricter threshold afterwards.
        iou: Per-class NMS overlap threshold. Default 0.45.
        classes: Class ids to keep. Default all.
        threads: CPU threads for inference. Default 0 = let OpenVINO decide.
        cache_dir: Where compiled models are cached. Only matters on a GPU,
            where the first compile takes tens of seconds; with a cache the
            second start is immediate. Default: ``.openvino_cache`` beside
            the model directory. Set to ``""`` to disable.
    """

    name = "openvino"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._model_ref = str(self.params.get("model", ""))
        self._device = resolve_device(self.params.get("device", "intel:cpu"))
        self._conf = float(self.params.get("model_confidence", 0.25))
        self._iou = float(self.params.get("iou", 0.45))
        self._threads = int(self.params.get("threads", 0) or 0)
        self._warmup = bool(self.params.get("warmup", True))

        classes = self.params.get("classes")
        self._classes = np.array([int(c) for c in classes]) if classes else None

        self._request = None
        self._names: Dict[int, str] = {}
        self._height = 0
        self._width = 0
        self._running_on = self._device

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        if self._loaded:
            return

        if importlib.util.find_spec("openvino") is None:
            raise BackendNotAvailableError(
                "The 'openvino' package is not installed, so the detection "
                "model cannot be loaded.\n"
                "Install it with:  pip install openvino\n"
                "Or set 'backend: ultralytics' and a .pt model on the module."
            )
        if not self._model_ref:
            raise ModelLoadError("The openvino backend needs a 'model' directory")

        import openvino as ov

        model_path = Path(
            verify_model_file(self._model_ref, expected_sha256=self.params.get("model_sha256"))
        )
        xml_path = self._find_xml(model_path)
        self._names = self._read_names(xml_path.parent)

        try:
            core = ov.Core()
            model = core.read_model(str(xml_path))
            self._height, self._width = self._input_size(model)
            self._check_output(model)

            device = self._device
            available = core.available_devices
            cache_dir = self._cache_dir(xml_path.parent)
            started = time.perf_counter()

            if device == "AUTO":
                self._request, device = self._fastest(core, model, available, cache_dir)
            else:
                if not _has_device(available, device):
                    logger.warning(
                        "OpenVINO device '%s' is not available here (found: %s) - using CPU",
                        device,
                        ", ".join(available) or "none",
                    )
                    device = "CPU"
                self._request = self._compile(core, model, device, cache_dir)
            self._running_on = device
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as ModelLoadError
            raise ModelLoadError(
                f"Could not load the OpenVINO model {self._model_ref!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._loaded = True
        logger.info(
            "Loaded %s on %s in %.1f s (input %dx%d, %d classes)",
            self._model_ref,
            device,
            time.perf_counter() - started,
            self._width,
            self._height,
            len(self._names),
        )

        if self._warmup:
            self._run_warmup()

    def _compile(self, core: Any, model: Any, device: str, cache_dir: str) -> Any:
        """Compile ``model`` for ``device`` and return an inference request."""
        config: Dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
        if self._threads > 0 and device == "CPU":
            config["INFERENCE_NUM_THREADS"] = self._threads
        if cache_dir and device != "CPU":
            config["CACHE_DIR"] = cache_dir
        return core.compile_model(model, device, config).create_infer_request()

    def _fastest(self, core: Any, model: Any, available: Any, cache_dir: str) -> Tuple[Any, str]:
        """``device: auto`` - the CPU, or the graphics chip when it earns it.

        The graphics chip is used only if it compiles the model, is clearly
        faster than the CPU and returns the same scores for the same picture.
        Anything else - no driver, a driver that fails, a slow or wrong
        result - leaves detection on the CPU, which always works.
        """
        cpu = self._compile(core, model, "CPU", cache_dir)
        if not _has_device(available, "GPU"):
            logger.info("device auto: no graphics device found - using CPU")
            return cpu, "CPU"

        blob = _test_blob(self._height, self._width)
        try:
            logger.info(
                "device auto: preparing the model for the graphics chip "
                "(the first time can take a minute; it is cached afterwards)"
            )
            gpu = self._compile(core, model, "GPU", cache_dir)
            gpu_ms, gpu_scores = _time_request(gpu, blob)
        except Exception as exc:  # noqa: BLE001 - a broken driver must not stop the camera
            logger.warning(
                "device auto: the graphics chip could not run the model (%s: %s) - using CPU",
                type(exc).__name__,
                str(exc).splitlines()[0] if str(exc) else "",
            )
            return cpu, "CPU"
        cpu_ms, cpu_scores = _time_request(cpu, blob)

        difference = float(np.abs(gpu_scores - cpu_scores).mean())
        if not np.isfinite(gpu_scores).all() or difference > _AUTO_MAX_SCORE_DIFFERENCE:
            logger.warning(
                "device auto: the graphics chip gives different results from the CPU "
                "(mean score difference %.3f) - using CPU. Updating the Intel "
                "graphics driver usually fixes this.",
                difference,
            )
            return cpu, "CPU"
        if gpu_ms > _AUTO_GPU_ADVANTAGE * cpu_ms:
            logger.info(
                "device auto: CPU %.0f ms, graphics %.0f ms per frame - using CPU",
                cpu_ms,
                gpu_ms,
            )
            return cpu, "CPU"
        logger.info(
            "device auto: CPU %.0f ms, graphics %.0f ms per frame - using the graphics chip",
            cpu_ms,
            gpu_ms,
        )
        return gpu, "GPU"

    def _find_xml(self, model_path: Path) -> Path:
        if model_path.is_file() and model_path.suffix.lower() == ".xml":
            return model_path
        if model_path.is_dir():
            found = sorted(model_path.glob("*.xml"))
            if found:
                return found[0]
        raise ModelLoadError(
            f"{self._model_ref!r} is not an OpenVINO model: expected a directory "
            "containing a .xml and .bin file (an Ultralytics OpenVINO export).\n"
            "For a .pt model use 'backend: ultralytics'."
        )

    def _read_names(self, directory: Path) -> Dict[int, str]:
        """Class names from the export's metadata.yaml (plain data, safe-loaded)."""
        metadata_path = directory / "metadata.yaml"
        if not metadata_path.is_file():
            raise ModelLoadError(
                f"{directory} has no metadata.yaml, so its class names are unknown. "
                "Export the model with Ultralytics, which writes that file."
            )
        try:
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            names = metadata.get("names") or {}
            task = str(metadata.get("task", "detect"))
            end2end = bool(metadata.get("end2end", False)) or bool(
                (metadata.get("args") or {}).get("nms")
            )
        except (OSError, yaml.YAMLError, AttributeError) as exc:
            raise ModelLoadError(f"Could not read {metadata_path}: {exc}") from exc

        if task != "detect":
            raise ModelLoadError(
                f"{directory} is a '{task}' model; the openvino backend runs "
                "detection models only."
            )
        if end2end:
            raise ModelLoadError(
                f"{directory} was exported with built-in NMS, which the openvino "
                "backend does not decode. Export without nms=True, or use "
                "'backend: ultralytics'."
            )
        if isinstance(names, list):
            names = dict(enumerate(names))
        if not names:
            raise ModelLoadError(f"{metadata_path} lists no class names")
        return {int(key): str(value) for key, value in names.items()}

    def _input_size(self, model: Any) -> Tuple[int, int]:
        shape = model.input(0).get_partial_shape()
        if shape.is_dynamic or len(shape) != 4:
            raise ModelLoadError(
                f"{self._model_ref!r} has a dynamic input shape. Export it with a "
                "fixed size, e.g. imgsz=(544, 960), dynamic=False."
            )
        height, width = int(shape[2].get_length()), int(shape[3].get_length())

        configured = self.params.get("imgsz")
        if configured is not None:
            wanted = (
                (int(configured[0]), int(configured[1]))
                if isinstance(configured, (list, tuple))
                else (int(configured), int(configured))
            )
            if wanted != (height, width):
                raise ModelLoadError(
                    f"imgsz is {list(wanted)} but {self._model_ref!r} was exported "
                    f"for [{height}, {width}]. Set imgsz: [{height}, {width}] on "
                    "every detector that uses this model, or point 'model' at an "
                    "export of the size you want."
                )
        return height, width

    def _check_output(self, model: Any) -> None:
        shape = model.output(0).get_partial_shape()
        expected = 4 + len(self._names)
        if shape.is_dynamic or len(shape) != 3 or int(shape[1].get_length()) != expected:
            raise ModelLoadError(
                f"{self._model_ref!r} does not look like a YOLO detection export: "
                f"output shape is {shape}, expected [1, {expected}, N] for "
                f"{len(self._names)} classes."
            )

    def _cache_dir(self, model_dir: Path) -> str:
        configured = self.params.get("cache_dir")
        if configured is not None:
            return str(configured)
        # Beside the model directory, never inside it: the directory's digest
        # covers every file in it.
        return str(model_dir.parent / ".openvino_cache")

    def _run_warmup(self) -> None:
        """One throwaway inference, so the first real frame is not a spike."""
        blank = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        started = time.perf_counter()
        try:
            self.infer(blank)
        except Exception as exc:  # noqa: BLE001 - warmup must never be fatal
            logger.warning("Model warmup failed (continuing): %s", exc)
            return
        logger.info("Model warmup took %.0f ms", (time.perf_counter() - started) * 1000.0)

    def unload(self) -> None:
        self._request = None
        super().unload()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def infer(self, frame: np.ndarray) -> List[RawDetection]:
        if not self._loaded or self._request is None:
            raise ModelLoadError("OpenVinoDetector.infer() called before load()")

        image, scale, (left, top) = letterbox(frame, self._height, self._width)
        blob = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None])
        blob = blob.astype(np.float32) / 255.0

        self._request.infer({0: blob})
        output = self._request.get_output_tensor(0).data[0]  # (4 + classes, N)

        return self._decode(output, scale, left, top, frame.shape[1], frame.shape[0])

    def _decode(
        self,
        output: np.ndarray,
        scale: float,
        left: int,
        top: int,
        frame_width: int,
        frame_height: int,
    ) -> List[RawDetection]:
        scores = output[4:]
        class_ids = scores.argmax(axis=0)
        confidences = scores[class_ids, np.arange(scores.shape[1])]

        keep = confidences >= self._conf
        if self._classes is not None:
            keep &= np.isin(class_ids, self._classes)
        if not keep.any():
            return []

        confidences = confidences[keep]
        class_ids = class_ids[keep]
        cx, cy, w, h = output[:4, keep]

        # Back from the letterboxed input to frame pixels.
        x1 = np.clip((cx - w / 2.0 - left) / scale, 0, frame_width)
        y1 = np.clip((cy - h / 2.0 - top) / scale, 0, frame_height)
        x2 = np.clip((cx + w / 2.0 - left) / scale, 0, frame_width)
        y2 = np.clip((cy + h / 2.0 - top) / scale, 0, frame_height)

        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        selected = cv2.dnn.NMSBoxesBatched(
            boxes.tolist(),
            confidences.tolist(),
            class_ids.tolist(),
            self._conf,
            self._iou,
        )
        indexes = np.asarray(selected, dtype=int).reshape(-1)
        indexes = indexes[np.argsort(-confidences[indexes], kind="stable")][:_MAX_DETECTIONS]

        return [
            RawDetection(
                label=self._names.get(int(class_ids[i]), str(int(class_ids[i]))),
                confidence=float(confidences[i]),
                xyxy=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                class_id=int(class_ids[i]),
            )
            for i in indexes
        ]

    @property
    def description(self) -> str:
        return f"openvino:{self._model_ref}@{self._running_on}"
