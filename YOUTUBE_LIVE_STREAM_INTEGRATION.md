# VisionGuard AI — YouTube Live Stream Integration

## 1. Overview & Objectives

VisionGuard AI now natively supports **YouTube Live Streams** and video broadcasts as first-class camera sources alongside RTSP / IP cameras, USB webcams / capture cards, and local video files.

This feature allows users, security operators, and computer vision developers to:
1. Connect VisionGuard directly to any public YouTube Live stream (e.g. traffic cameras, public surveillance feeds, event broadcasts) or recorded video URL.
2. Select **YouTube Stream** in the web dashboard's Cameras configuration page.
3. Automatically resolve stream formats, extract high-quality video frames, and feed them into the VisionGuard AI detection, tracking, and alerting pipelines in real time.
4. Seamlessly persist YouTube camera definitions across application restarts in `configs/cameras.yaml`.
5. Maintain ultra-low latency with zero frame accumulation through background frame draining.

---

## 2. Architecture & Design

```
YouTube URL (https://www.youtube.com/watch?v=...)
       │
       ▼
   yt-dlp Extractor
  (Resolves direct HLS .m3u8 / MP4 manifests; auto-refreshes tokens)
       │
       ▼
  cv2.VideoCapture (FFmpeg backend)
       │
       ▼
 _LatestFrameReader (Background Thread)
  (Drains buffered frames; serves only newest frame to eliminate live latency)
       │
       ▼
  YouTubeSource (OpenCVVideoSource abstraction)
       │
       ▼
  VisionGuard AI Core (FrameProcessor & Model Registry)
  (YOLOv11 Detection → Tracking → Zone Analysis → Alerts)
       │
       ▼
  Live Dashboard & Multi-Camera Stream Hub
```

### Key Technical Pillars

1. **Dynamic Stream Resolution (`yt-dlp`)**:
   - Live streams change chunks and manifest URLs frequently. `YouTubeSource` resolves the direct HLS (`.m3u8`) or progressive MP4 video streams using `yt-dlp`.
   - Prioritizes resolutions up to 1080p/720p for optimal inference efficiency without consuming excessive network bandwidth.
   - Detects the local Node.js runtime if installed to assist extraction and eliminate JS interpreter warnings.

2. **Ultra-Low Latency via Background Frame Draining (`_LatestFrameReader`)**:
   - Because YouTube Live delivers video at 30–60 FPS while heavy multi-model AI inference may process at 10–25 FPS, standard sequential reading buffers frames inside OpenCV/FFmpeg. This causes the "live" view to drift minutes into the past.
   - `_LatestFrameReader` continuously drains the video stream on a dedicated daemon thread and keeps only the newest frame, discarding stale frames and reporting the drop count.

3. **Auto-Token Refresh & Reconnection Resilience**:
   - YouTube HLS URLs include temporary security tokens that expire after several hours.
   - When a network disconnect or stream stall occurs, `_reconnect()` automatically re-invokes `_resolve_stream_url()` to obtain a fresh tokenized manifest from YouTube before reopening the capture with exponential backoff (`reconnect_attempts`, `reconnect_delay_sec`, `reconnect_max_delay_sec`).
   - If reconnect attempts fail temporarily, the source safely transitions to `_FailedReader`, preventing unhandled exceptions and allowing subsequent retry attempts.

4. **Resource Management & Zero Leakage**:
   - On camera stop or camera switching, `YouTubeSource.release()` cleanly signals and joins the background reader thread and closes the native OpenCV video descriptor, avoiding orphaned network threads or descriptor leaks.

---

## 3. Frontend Integration

### 3.1 Source Dropdown & Context-Sensitive Form
- In `frontend/web/src/components/cameras/CameraSlotForm.tsx`:
  - Added `youtube: 'YouTube Stream'` to `SOURCE_LABEL`.
  - When selected, hides RTSP camera address and video file upload controls, displaying the dedicated **YouTube stream URL** input field with helper documentation.
  - Allows full customization of Camera Name and individual or shared AI module selections (person detection, tracking, zones, etc.).

### 3.2 Types & Draft State Management
- In `frontend/web/src/api/types.ts`:
  - Extended `SourceType = 'webcam' | 'file' | 'rtsp' | 'youtube'`.
  - Added `youtube_url?: string | null` to `CameraInfo` and `CameraInput`.
- In `frontend/web/src/lib/cameraDrafts.ts`:
  - Added `youtubeUrl: string` to `CameraDraft`.
  - Form validation enforces valid YouTube URLs (matching `youtube.com/watch?v=...`, `youtube.com/live/...`, `youtu.be/...`).
  - Serializes `youtube_url` into the `POST /api/cameras` payload.

---

## 4. Backend & Configuration Schema

### 4.1 Schema Definitions (`computer_vision/core/config.py`)
- `VideoSourceType`: Added `YOUTUBE = "youtube"`.
- `VideoConfig` & `CameraConfig`: Added `youtube_url: str | None = None`.
- Added `is_youtube_url()` validation ensuring URL format validity before pipeline initialization.

### 4.2 Storage & Persistence (`backend/services/cameras.py`)
- `public_view(camera)`: Exposes `youtube_url` in the API output.
- `save_camera_file(path, cameras)`: Serializes `source: youtube` and `youtube_url` to `configs/cameras.yaml`.
- `load_camera_file(path)` & `parse_cameras(items)`: Restores YouTube camera configurations on application restart.

### 4.3 CLI Overrides (`backend/main.py`)
- Added `--source youtube` and `--youtube-url <url>` to the CLI arguments for headless testing or CLI-driven deployments.

---

## 5. Verification & Testing

### 5.1 Automated Test Suite
Created 26 dedicated unit and integration tests in `tools/tests/test_youtube_source.py`:
- Valid and invalid YouTube URL regex testing.
- Pydantic schema validation and failure on missing/invalid domains.
- `parse_cameras` payload parsing and `public_view` output.
- `save_camera_file` and `load_camera_file` round-trip disk persistence.
- `YouTubeSource` resolution, format selection, frame capture, `peek()` live preview, and release.
- Exponential backoff reconnection and `FrameReadError` exhaustion handling.

```powershell
.\.venv\Scripts\pytest -v tools/tests/test_youtube_source.py
# 26 passed in 0.47s (100%)
```

### 5.2 Full Platform Regression
All existing test suites pass with zero regressions:
```powershell
.\.venv\Scripts\pytest -q tools/tests/test_sources.py tools/tests/test_config.py tools/tests/test_capture_paths.py
# 67 passed (100%)

.\.venv\Scripts\pytest -q tools/tests/test_multi_camera.py
# 90 passed (100%)

.\.venv\Scripts\pytest -q tools/tests/test_live_performance.py
# 113 passed (100%)
```

### 5.3 Frontend Production Build
```powershell
cd frontend/web
npm.cmd run build
# tsc -b && vite build
# Built cleanly in 1.92s with 0 errors or warnings.
```

### 5.4 Live End-to-End Stream Test
Verified live frame capture and real AI inference against the official NASA ISS live stream:
```powershell
.\.venv\Scripts\python.exe -m backend.main --source youtube --youtube-url "https://www.youtube.com/watch?v=M3HKLzjvKPc" --max-frames 2 --no-window
```
**Execution Output:**
```text
INFO | Opening YouTubeSource (target=https://www.youtube.com/watch?v=M3HKLzjvKPc, backend=ffmpeg)
INFO | Source ready in 811 ms: youtube-0 @ 1280x720, 59.9 fps
INFO | YouTube stream live: https://www.youtube.com/watch?v=M3HKLzjvKPc (Live Video from the International Space Station (Official NASA Stream))
INFO | Pipeline started on {'source_id': 'youtube-0', 'source_type': 'youtube', 'width': 1280, 'height': 720, 'fps': 59.94}
INFO | Processing resolution: 1280x720 -> 960x540
INFO | Done: 2 frames in 0.1 s | avg 6.9 FPS | avg processing latency 121.8 ms | avg capture wait 8.9 ms
```

---

## 6. How to Use

### Via Web Dashboard
1. Start the backend and frontend servers:
   ```powershell
   # Terminal 1 (Backend)
   .\.venv\Scripts\python.exe -m backend.main --config configs/config.yaml --no-window --dashboard

   # Terminal 2 (Frontend)
   cd frontend\web
   npm.cmd run dev -- --host 127.0.0.1 --port 5173
   ```
2. Navigate to **`http://127.0.0.1:5173/portal/cameras`**.
3. Under any camera slot, select **YouTube Stream** in the **Source** dropdown.
4. Paste a valid YouTube Live URL (e.g. `https://www.youtube.com/watch?v=M3HKLzjvKPc`).
5. Click **Apply configuration**. The dashboard will automatically connect, start inference, and display the stream on the Live Monitoring page.

### Via `configs/cameras.yaml`
```yaml
cameras:
  - id: cam-yt-live
    name: Public Traffic Feed
    source: youtube
    youtube_url: https://www.youtube.com/watch?v=M3HKLzjvKPc
    modules:
      - person_detection
      - vehicle_detection
      - vehicle_tracking
```

### Via Command-Line
```powershell
.\.venv\Scripts\python.exe -m backend.main --source youtube --youtube-url "https://www.youtube.com/watch?v=M3HKLzjvKPc" --dashboard
```

---

## 7. Operational Notes & Limitations

- **Public vs. Protected Streams**: Compatible with public and unlisted YouTube Live streams and video URLs. Private or age-restricted videos that require signing in with a Google account are not supported without providing custom cookie files.
- **Bandwidth Consumption**: Continuous 720p/1080p live streams consume outbound and inbound internet bandwidth.
- **Geo-Blocking**: Streams geographically restricted by broadcasters will be rejected by YouTube's servers with descriptive error notices.
