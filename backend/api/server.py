"""HTTP server for the dashboard.

Runs on a background thread so the pipeline keeps the main thread. The server
only ever reads :class:`DashboardState`; it cannot drive or reconfigure the
pipeline, which keeps a browser tab from being able to disturb a running
camera.

Live video is MJPEG rather than WebRTC or websockets: a browser renders it
with a plain ``<img>`` tag, there is no client-side buffering to drift behind,
and it needs no JavaScript at all to work.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from backend.services.state import DashboardState
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

#: The frontend is a sibling of the backend, not a resource inside it, so the
#: UI can be replaced or served by a real web server without touching Python.
STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend" / "static"
_BOUNDARY = "visionguard-frame"


class DashboardUnavailableError(RuntimeError):
    """FastAPI/uvicorn are not installed."""


def build_app(state: DashboardState):
    """Create the FastAPI application bound to ``state``."""
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
    except ImportError as exc:  # pragma: no cover - depends on install
        raise DashboardUnavailableError(
            "The dashboard needs FastAPI and uvicorn:\n"
            "    pip install 'fastapi>=0.110' 'uvicorn[standard]>=0.29'\n"
            "Or run with dashboard.enabled: false."
        ) from exc

    app = FastAPI(title="VisionGuard AI", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status():
        return JSONResponse(state.snapshot())

    @app.get("/api/events")
    def events(since: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        return JSONResponse({"events": state.events_since(since, limit)})

    @app.get("/api/snapshot.jpg", include_in_schema=False)
    def snapshot():
        jpeg = state.latest_jpeg()
        if jpeg is None:
            return Response(status_code=503, content=b"no frame yet")
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/stream.mjpg", include_in_schema=False)
    def stream():
        return StreamingResponse(
            _mjpeg_frames(state),
            media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        )

    return app


def _mjpeg_frames(state: DashboardState):
    """Yield multipart JPEG chunks, one per new frame."""
    last_seq = 0
    while state.running:
        jpeg, last_seq = state.wait_for_frame(last_seq, timeout=1.0)
        if jpeg is None:
            continue  # timed out; loop re-checks whether the run ended
        yield (
            f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            + jpeg
            + b"\r\n"
        )


class DashboardServer:
    """Runs the dashboard on a daemon thread alongside the pipeline."""

    def __init__(self, state: DashboardState, host: str, port: int) -> None:
        self._state = state
        self._host = host
        self._port = port
        self._thread: Optional[threading.Thread] = None
        self._server = None

    @property
    def url(self) -> str:
        shown = "localhost" if self._host in ("0.0.0.0", "") else self._host
        return f"http://{shown}:{self._port}"

    def start(self) -> None:
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - depends on install
            raise DashboardUnavailableError(
                "The dashboard needs uvicorn: pip install 'uvicorn[standard]>=0.29'"
            ) from exc

        config = uvicorn.Config(
            build_app(self._state),
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(
            target=self._server.run, name="visionguard-dashboard", daemon=True
        )
        self._thread.start()
        logger.info("Dashboard: %s", self.url)

        if self._host in ("0.0.0.0", "::"):
            logger.warning(
                "Dashboard is listening on %s - it is reachable from the whole "
                "network and has no authentication. Use 127.0.0.1 unless that "
                "is what you intended.",
                self._host,
            )

    def stop(self) -> None:
        self._state.stop()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
