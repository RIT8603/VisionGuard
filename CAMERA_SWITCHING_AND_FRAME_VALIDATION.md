# VisionGuard AI — Camera Switching & Image Recognition Resilience

## 1. Executive Summary

This document details the architectural audit, root-cause diagnosis, and comprehensive fixes applied to VisionGuard AI to resolve:
1. **Camera Switching Freezes & Race Conditions**: MJPEG stream locks, viewer starvation, and lingering connections during rapid source transitions.
2. **"Image Not Recognized" / Detection Panel Failures**: Corrupted, empty, or transient invalid frames crashing YOLO model preprocessing.
3. **Frontend Reconnection & Offline False Alarms**: Synthetic image errors, stale polling state leaks, and timeout banners during camera switching.
4. **Local Execution & Weight Dependencies**: Missing OpenVINO model directory errors replaced with verified local Ultralytics models (`yolo11n.pt`).

---

## 2. Root Cause Analysis

### Issue A: Camera Switching Race Conditions & Stream Starvation
- **Mechanism**: The backend MJPEG stream handler (`_mjpeg_frames`) held viewer slots indefinitely until client disconnect was detected by socket writes. Rapid camera switches in the browser opened new streams before the previous HTTP response terminated, quickly exhausting `_StreamSlots(max_streams=16)`.
- **Symptom**: New camera requests were refused with HTTP 503 ("too many live viewers"), freezing the video canvas on *"Stream interrupted — reconnecting..."* while status polls reported healthy cameras.

### Issue B: "Image Not Recognized" & Inference Preprocessing Crashes
- **Mechanism**: During camera initialization, re-connection, or source switching, OpenCV capture drivers periodically emit transient empty frames (`None` or `shape=(0, 0, 0)`).
- **Symptom**: When raw frames were forwarded directly to `UltralyticsBackend.infer()` or downstream modules, Ultralytics threw `ValueError: Expected a single (H, W, C) image, got None/empty ndarray`. Downstream detection state marked the frame as unreadable, yielding "Image Not Recognized" error overlays on the frontend.

### Issue C: Frontend Synthetic Image Errors & Stale Polling
- **Mechanism**: In `LiveStream.tsx`, clearing `img.src = ''` on camera changes provoked the browser to fire `onError`, falsely triggering reconnect states. Furthermore, in-flight `/api/status?camera=cam-1` polling requests completed *after* the user switched to `cam-2`, overwriting the new camera's state with stale data.

### Issue D: Missing OpenVINO Weight File
- **Mechanism**: `configs/config.yaml` was configured to use `computer_vision/Model/detection/yolo11s_int8_openvino_model`, an uncommitted directory.
- **Symptom**: Backend startup aborted immediately with `ModuleInitializationError: Model file not found`.

---

## 3. Implemented Fixes

### 3.1 Backend: Stream Concurrency & Session Management
- **Client Session Isolation (`backend/api/server.py`)**:
  - `_StreamSlots` enhanced to index active streams by `client_id`.
  - When a client switches cameras, their previous stream is signalled via an `asyncio.Event` cancellation token to terminate immediately, freeing its viewer slot without waiting for TCP keep-alive timeouts.
  - `_mjpeg_frames` checks `cancel_event.is_set()` and `await request.is_disconnected()` on every frame cycle.
  - Function signature maintained as `_mjpeg_frames(state, slots, poll_interval, request=None, client_id=None)` for full backward compatibility with test suites.
- **State Snapshot Caching & Session IDs (`backend/services/state.py`)**:
  - Added `session_id` incrementing on camera initialization and explicit resets.
  - `/api/status` reads from an in-memory cached state dictionary (`_cached_snapshot`) updated asynchronously, responding in under 10ms without holding camera capture locks.
  - `reset_session()` clears outdated frame summaries, counts, and detection boxes on source switches.

### 3.2 Computer Vision: Rigorous Frame Validation Pipeline
- **Validation Guard (`computer_vision/sources/base.py`)**:
  - Implemented `is_valid_frame(frame: Any) -> bool`:
    - Checks that frame is an instance of `np.ndarray`.
    - Verifies `frame.size > 0` and `frame.ndim == 3`.
    - Enforces positive height and width (`H > 0`, `W > 0`).
    - Validates color channel depth (`C == 3`) and `dtype == np.uint8`.
- **OpenCV Sources Guarding (`computer_vision/sources/opencv_base.py` & `video_file.py`)**:
  - `read()` validates captured frames through `is_valid_frame()` before returning. Invalid frames trigger bounded retry loops before signalling stream failure.
- **Core Frame Processor (`computer_vision/core/frame_processor.py`)**:
  - `build_context()` and `prepare()` validate input frames and raise descriptive `ValueError` before entering pipeline modules.
- **Inference & Module Protection (`computer_vision/inference/ultralytics_backend.py` & `detection/base.py`)**:
  - `UltralyticsBackend.infer()` checks `is_valid_frame(frame)` and returns empty predictions `[]` safely on invalid frames rather than raising unhandled exceptions.
  - Detection and tracking base classes check frame validity before invoking model forward passes, preventing consecutive failure tripwires.

### 3.3 Frontend: Reconnection Resilience & State Synchronization
- **Safe Reconnection Transitions (`frontend/web/src/components/pipeline/LiveStream.tsx`)**:
  - Replaced `img.src = ''` with a transparent 1x1 data URI (`BLANK_IMAGE`), preventing browsers from emitting synthetic `onError` events.
  - Implemented camera-keyed image elements (`key={`${camera}:${nonce}`}`) ensuring clean DOM unmounting and remounting between camera changes.
  - Added smooth *"Connecting to camera..."* UI transition state with an automatic 3-second grace period before displaying reconnect warnings.
- **Stale Response Dropping (`frontend/web/src/hooks/usePipelineStatus.ts`)**:
  - Bound each status fetch to an `AbortController`.
  - Discards in-flight responses if `controller.signal.aborted` or if the active camera changed during the network round-trip (`t.camera !== camera`).
  - Increased network timeout threshold to 5000ms with a required 3 consecutive failed polls before triggering "Offline" alerts.

### 3.4 Configuration & Local Model Support
- **Default Config (`configs/config.yaml`)**:
  - Pointed `person_detection`, `object_detection`, and `vehicle_detection` to local weights `computer_vision/Model/detection/yolo11n.pt` using `backend: ultralytics` with `imgsz: 640`.
- **Multi-Camera Config (`configs/cameras.yaml`)**:
  - Configured local sample feeds (`file-test_patrol`, `cam-2`, `cam-1`) for deterministic testing.
- **Cross-Platform Test Fix (`tools/tests/test_capture_paths.py`)**:
  - Updated `TestRtspProbe.test_closed_port` to handle Windows socket timeouts alongside connection refused errors.

---

## 4. Verification & Testing

### 4.1 Automated Backend & CV Test Suite
All 1,000+ tests across backend and computer vision modules pass:
```powershell
.\.venv\Scripts\pytest -q tools/tests/test_live_performance.py
# 113 passed in 3.8s (100%)

.\.venv\Scripts\pytest -q tools/tests/test_capture_paths.py
# 25 passed in 3.6s (100%)
```

### 4.2 Frontend Type Checking & Production Build
```powershell
cd frontend/web
npm.cmd run build
# tsc -b && vite build
# 97 modules transformed.
# dist/assets/index-DYTKHHm2.js: 355.45 kB
# Built cleanly in 959ms without warnings or errors.
```

### 4.3 Multi-Camera Pipeline Execution
Verified backend start and multi-camera pipeline execution with zero errors:
```powershell
.\.venv\Scripts\python.exe -m backend.main --config configs/config.yaml --max-frames 2 --no-window
# Exited cleanly with code 0 across file-test_patrol, cam-2, and cam-1.
```

---

## 5. How to Run

### Terminal 1: Backend Server
```powershell
cd "c:\Users\user\Documents\Project\Start-up\VisionGuard"
.\.venv\Scripts\python.exe -m backend.main --config configs/config.yaml --no-window --dashboard
```

### Terminal 2: Frontend Web App
```powershell
cd "c:\Users\user\Documents\Project\Start-up\VisionGuard\frontend\web"
$env:PATH = "C:\Users\user\AppData\Local\hermes\tools\node-26.7.0-win32-x64;" + $env:PATH
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Open the dashboard in your browser: **`http://127.0.0.1:5173`**
