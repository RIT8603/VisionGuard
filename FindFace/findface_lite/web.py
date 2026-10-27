"""Web UI for FindFace Lite.

Provides a modern real-time browser dashboard for live video face detection,
tracking, quality estimation, best-snapshot capture, and watchlist recognition.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2

from .config import Config
from .pipeline import FacePipeline
from .recognition import Recognizer
from .sinks import DirectorySink


def get_available_sources() -> list[dict]:
    sources = [{"id": "0", "name": "📹 Webcam (Device Index 0)", "type": "webcam"}]
    samples_dir = Path("samples")
    if samples_dir.exists():
        for f in sorted(samples_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                sources.append({
                    "id": str(f).replace("\\", "/"),
                    "name": f"🎬 {f.name}",
                    "type": "file"
                })
    return sources


class WebPipelineServer:
    def __init__(self, source: str = "samples/demo.mp4", out_dir: str = "output/web",
                 watchlist_dir: Optional[str] = None, mode: str = "offline", loop: bool = True):
        self.source = source
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.watchlist_dir = watchlist_dir
        self.mode = mode
        self.loop = loop

        self.cfg = Config()
        self.cfg.overall_only = (mode == "offline")

        self.lock = threading.Lock()
        self.current_frame_jpeg: Optional[bytes] = None
        self.events: list[dict] = []
        self.stats: dict = {"fps": 0.0, "frames": 0, "tracks": 0, "detections": 0, "detector_runs": 0, "source": self.source}
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_requested = False

        self._start_pipeline()

    def _start_pipeline(self):
        self.stop_requested = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def restart(self, source: Optional[str] = None, mode: Optional[str] = None, loop: Optional[bool] = None):
        with self.lock:
            self.stop_requested = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        with self.lock:
            if source is not None:
                self.source = source
            if mode is not None:
                self.mode = mode
                self.cfg.overall_only = (mode == "offline")
            if loop is not None:
                self.loop = loop
            self.events = []
            self.current_frame_jpeg = None
            self._start_pipeline()

    def _run_loop(self):
        sinks = [DirectorySink(str(self.out_dir), self.cfg.jpeg_quality)]
        hooks = []
        if self.watchlist_dir and Path(self.watchlist_dir).exists():
            try:
                rec = Recognizer.from_config(self.cfg, self.watchlist_dir)
                hooks.append(rec)
            except Exception as e:
                print(f"[WebUI] Watchlist load error: {e}")

        pipeline = FacePipeline(self.cfg, sinks, source_name=str(self.source), hooks=hooks)

        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[WebUI] Error: Cannot open video source: {self.source}")
            return

        live = str(self.source).isdigit() or "://" in str(self.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = 1.0 / max(1.0, fps)
        idx = 0
        start_time = time.monotonic()
        last_t = time.monotonic()

        while not self.stop_requested:
            loop_start = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                if self.loop and not live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    break

            t = time.monotonic() - start_time if live else idx / fps
            new_events = pipeline.process(frame, idx, t)

            vis = pipeline.draw(frame)
            # If the image is large, resize visualization slightly for fast streaming responsiveness
            vh, vw = vis.shape[:2]
            max_dim = 1280
            if max(vh, vw) > max_dim:
                scale = max_dim / float(max(vh, vw))
                vis = cv2.resize(vis, (int(vw * scale), int(vh * scale)))

            _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = buf.tobytes()

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            current_fps = round(1.0 / dt, 1) if dt > 0 else 0.0

            with self.lock:
                self.current_frame_jpeg = jpeg_bytes
                if new_events:
                    self.events.extend(new_events)
                s = pipeline.stats
                self.stats = {
                    "fps": current_fps,
                    "frames": s["frames"],
                    "tracks": len(pipeline.tracker.active),
                    "detections": s["detections"],
                    "detector_runs": s["detector_runs"],
                    "events_count": len(self.events),
                    "source": str(self.source),
                    "mode": self.mode,
                }

            idx += 1
            if not live:
                elapsed = time.monotonic() - loop_start
                sleep_left = frame_interval - elapsed
                if sleep_left > 0.001:
                    time.sleep(sleep_left)

        cap.release()
        trailing = pipeline.finish()
        with self.lock:
            if trailing:
                self.events.extend(trailing)


def create_handler(server_state: WebPipelineServer, out_dir: Path):
    class RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return  # Suppress normal access logs to keep console clean

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                self._serve_html()
            elif path == "/video_feed":
                self._serve_video_stream()
            elif path == "/api/status":
                self._serve_json(server_state.stats)
            elif path == "/api/sources":
                sources = get_available_sources()
                self._serve_json({"sources": sources, "current": str(server_state.source)})
            elif path == "/api/events":
                with server_state.lock:
                    evs = list(server_state.events)
                self._serve_json({"events": evs})
            elif path.startswith("/output/"):
                subpath = path[len("/output/"):]
                target = out_dir / subpath
                if target.exists() and target.is_file():
                    self._serve_file(target)
                else:
                    self.send_error(404, "File not found")
            else:
                self.send_error(404, "Not Found")

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/control":
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                try:
                    data = json.loads(body.decode("utf-8"))
                    source = data.get("source")
                    mode = data.get("mode")
                    loop = data.get("loop", True)
                    server_state.restart(source=source, mode=mode, loop=loop)
                    self._serve_json({"status": "ok", "message": "Pipeline updated"})
                except Exception as e:
                    self._serve_json({"status": "error", "message": str(e)}, status=400)
            else:
                self.send_error(404, "Not Found")

        def _serve_file(self, file_path: Path):
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
            try:
                with open(file_path, "rb") as fh:
                    content = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception:
                self.send_error(500, "Error reading file")

        def _serve_json(self, data, status: int = 200):
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _serve_video_stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()

            while not server_state.stop_requested:
                frame_bytes = None
                with server_state.lock:
                    if server_state.current_frame_jpeg:
                        frame_bytes = server_state.current_frame_jpeg

                if frame_bytes:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.033)

        def _serve_html(self):
            html = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

    return RequestHandler


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FindFace Lite | Real-Time AI Video Analytics</title>
  <style>
    :root {
      --bg-dark: #0a0f1d;
      --card-bg: #11192e;
      --card-border: #1e293b;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 16px;
      min-height: 100vh;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 20px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      margin-bottom: 16px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-badge {
      background: linear-gradient(135deg, #2563eb, #06b6d4);
      color: white;
      font-weight: 800;
      font-size: 14px;
      padding: 6px 10px;
      border-radius: 8px;
      letter-spacing: 0.5px;
    }
    .title {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }
    .subtitle {
      font-size: 12px;
      color: var(--text-muted);
    }
    .live-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      color: #34d399;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
      70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .stats-bar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 16px;
    }
    .stat-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 22px;
      font-weight: 700;
      color: var(--accent-cyan);
    }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 16px;
    }
    @media (max-width: 1024px) {
      .main-grid { grid-template-columns: 1fr; }
    }
    .video-panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .video-wrapper {
      position: relative;
      background: #000;
      min-height: 480px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .video-wrapper img {
      width: 100%;
      height: auto;
      max-height: 72vh;
      object-fit: contain;
      display: block;
    }
    .controls-panel {
      padding: 14px 18px;
      background: #0c1324;
      border-top: 1px solid var(--card-border);
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      justify-content: space-between;
    }
    .control-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    select, input, button {
      background: #1e293b;
      color: var(--text-main);
      border: 1px solid #334155;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
      outline: none;
      transition: all 0.2s;
    }
    select {
      max-width: 320px;
      cursor: pointer;
    }
    select:focus, input:focus {
      border-color: var(--accent-blue);
    }
    button {
      background: #2563eb;
      border: none;
      cursor: pointer;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    button:hover {
      background: #1d4ed8;
    }
    .events-panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      height: calc(100vh - 160px);
    }
    .events-header {
      padding: 14px 16px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .events-header h3 {
      font-size: 15px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .events-badge {
      background: #2563eb;
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 10px;
    }
    .events-list {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .event-card {
      background: #0d1527;
      border: 1px solid #1f2c47;
      border-radius: 8px;
      padding: 10px;
      display: flex;
      gap: 12px;
      transition: transform 0.2s, border-color 0.2s;
    }
    .event-card:hover {
      transform: translateY(-2px);
      border-color: var(--accent-blue);
    }
    .event-thumb {
      width: 72px;
      height: 72px;
      border-radius: 6px;
      object-fit: cover;
      border: 1px solid #334155;
      background: #000;
    }
    .event-details {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 4px;
    }
    .event-title {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 700;
    }
    .match-tag {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 4px;
    }
    .match-found {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .match-unknown {
      background: rgba(148, 163, 184, 0.15);
      color: #94a3b8;
    }
    .quality-bar-container {
      background: #1e293b;
      height: 5px;
      border-radius: 3px;
      overflow: hidden;
      margin-top: 2px;
    }
    .quality-bar {
      height: 100%;
      background: linear-gradient(90deg, #3b82f6, #10b981);
    }
    .metrics-row {
      display: flex;
      gap: 8px;
      font-size: 11px;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="logo-badge">FINDFACE</div>
      <div>
        <div class="title">FindFace Lite</div>
        <div class="subtitle">YuNet Detector &bull; SFace Recognition &bull; Motion Gated Tracker</div>
      </div>
    </div>
    <div class="live-indicator">
      <div class="pulse-dot"></div>
      <span>STREAM ACTIVE</span>
    </div>
  </header>

  <div class="stats-bar">
    <div class="stat-card">
      <div class="stat-label">Frame Rate</div>
      <div class="stat-value" id="val-fps">0.0 <span style="font-size:13px;color:var(--text-muted)">FPS</span></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Active Tracks</div>
      <div class="stat-value" id="val-tracks">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Face Detections</div>
      <div class="stat-value" id="val-detections">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Detector Runs</div>
      <div class="stat-value" id="val-runs">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Best Face Snapshots</div>
      <div class="stat-value" id="val-events" style="color:var(--accent-green)">0</div>
    </div>
  </div>

  <div class="main-grid">
    <div class="video-panel">
      <div class="video-wrapper">
        <img id="stream-img" src="/video_feed" alt="Live Video Feed">
      </div>
      <div class="controls-panel">
        <div class="control-group">
          <label style="font-size:12px;color:var(--text-muted)">Source:</label>
          <select id="source-select">
            <option value="">Loading video list...</option>
          </select>
          <button id="refresh-sources-btn" title="Refresh video list" style="padding:8px 10px;background:#334155;">🔄</button>
        </div>
        <div class="control-group">
          <label style="font-size:12px;color:var(--text-muted)">Mode:</label>
          <select id="mode-select">
            <option value="offline">Offline (1 Best Face per Track)</option>
            <option value="realtime">Real-time (Periodic Best Windows)</option>
          </select>
        </div>
        <div class="control-group">
          <button id="apply-btn">Switch Stream</button>
        </div>
      </div>
    </div>

    <div class="events-panel">
      <div class="events-header">
        <h3>Captured Best Faces <span class="events-badge" id="events-count">0</span></h3>
        <span style="font-size:11px;color:var(--text-muted)">Top Quality Snapshots</span>
      </div>
      <div class="events-list" id="events-container">
        <div style="color:var(--text-muted);font-size:12px;text-align:center;padding:40px 10px;">
          Waiting for face tracks to finish and post best snapshots...
        </div>
      </div>
    </div>
  </div>

  <script>
    const knownEventIds = new Set();

    async function loadSources() {
      try {
        const res = await fetch('/api/sources');
        if (res.ok) {
          const data = await res.json();
          const select = document.getElementById('source-select');
          select.innerHTML = '';
          for (const s of data.sources) {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = s.name;
            if (s.id === data.current) {
              opt.selected = true;
            }
            select.appendChild(opt);
          }
        }
      } catch (err) {}
    }

    async function updateStats() {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const s = await res.json();
          document.getElementById('val-fps').innerHTML = `${s.fps} <span style="font-size:13px;color:var(--text-muted)">FPS</span>`;
          document.getElementById('val-tracks').textContent = s.tracks;
          document.getElementById('val-detections').textContent = s.detections;
          document.getElementById('val-runs').textContent = s.detector_runs;
          document.getElementById('val-events').textContent = s.events_count;
        }
      } catch (err) {}
    }

    async function updateEvents() {
      try {
        const res = await fetch('/api/events');
        if (res.ok) {
          const data = await res.json();
          const list = data.events || [];
          document.getElementById('events-count').textContent = list.length;
          const container = document.getElementById('events-container');

          if (list.length > 0 && container.querySelector('div[style*="text-align:center"]')) {
            container.innerHTML = '';
          }

          for (const ev of list) {
            const key = `${ev.event_id}_${ev.track_id}_${ev.frame_idx}`;
            if (!knownEventIds.has(key)) {
              knownEventIds.add(key);

              const q = ev.quality || {};
              const match = ev.match;
              const card = document.createElement('div');
              card.className = 'event-card';

              const matchBadge = match && match.name
                ? `<span class="match-tag match-found">${match.name} (${(match.similarity * 100).toFixed(0)}%)</span>`
                : `<span class="match-tag match-unknown">Track #${ev.track_id}</span>`;

              const qScore = (q.score || 0).toFixed(2);
              const qWidth = Math.round((q.score || 0) * 100);

              card.innerHTML = `
                <img class="event-thumb" src="/output/${ev.face_path}" alt="Face crop">
                <div class="event-details">
                  <div class="event-title">
                    <span>Face #${ev.track_id}</span>
                    ${matchBadge}
                  </div>
                  <div>
                    <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-muted)">
                      <span>Quality</span>
                      <span>${qScore}</span>
                    </div>
                    <div class="quality-bar-container">
                      <div class="quality-bar" style="width: ${qWidth}%"></div>
                    </div>
                  </div>
                  <div class="metrics-row">
                    <span>Sharp: ${(q.sharpness || 0).toFixed(2)}</span>
                    <span>Frontal: ${(q.frontal || 0).toFixed(2)}</span>
                    <span>t=${(ev.timestamp || 0).toFixed(1)}s</span>
                  </div>
                </div>
              `;
              container.prepend(card);
            }
          }
        }
      } catch (err) {}
    }

    document.getElementById('refresh-sources-btn').addEventListener('click', loadSources);

    document.getElementById('apply-btn').addEventListener('click', async () => {
      const source = document.getElementById('source-select').value;
      const mode = document.getElementById('mode-select').value;
      await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source, mode, loop: true })
      });
      knownEventIds.clear();
      document.getElementById('events-container').innerHTML = '';
      const img = document.getElementById('stream-img');
      img.src = '/video_feed?t=' + Date.now();
    });

    loadSources();
    setInterval(updateStats, 800);
    setInterval(updateEvents, 1000);
  </script>
</body>
</html>
"""


def run_web(source: str = "samples/demo.mp4", port: int = 5000, out_dir: str = "output/web",
            watchlist: Optional[str] = None, mode: str = "offline", loop: bool = True):
    server_state = WebPipelineServer(source=source, out_dir=out_dir, watchlist_dir=watchlist,
                                     mode=mode, loop=loop)
    handler_class = create_handler(server_state, Path(out_dir))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler_class)
    print(f"\n=======================================================")
    print(f" FindFace Lite Web UI is running at:")
    print(f" -> http://localhost:{port}")
    print(f" Source: {source} | Mode: {mode} | Loop: {loop}")
    print(f" Press Ctrl+C to stop the server.")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web UI server...")
    finally:
        server_state.stop_requested = True
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="FindFace Lite Web UI")
    parser.add_argument("--source", default="samples/demo.mp4", help="Video file or camera index (default: samples/demo.mp4)")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--out", default="output/web", help="Output directory for snapshots")
    parser.add_argument("--watchlist", help="Directory of enrolled photos")
    parser.add_argument("--mode", choices=["offline", "realtime"], default="offline")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop video")
    args = parser.parse_args()

    run_web(source=args.source, port=args.port, out_dir=args.out,
            watchlist=args.watchlist, mode=args.mode, loop=not args.no_loop)


if __name__ == "__main__":
    main()
