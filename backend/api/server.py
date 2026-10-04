"""HTTP server for the dashboard.

Runs on a background thread so the pipeline keeps the main thread. The server
reads :class:`DashboardState` for everything except two endpoints:
``POST /api/modules`` lets an operator change which AI modules are enabled, and
``POST /api/cameras`` lets them change which cameras are watched. Both are
narrow by design: modules can only be toggled (by name, from the catalogue the
platform already knows), and a camera entry is validated to be a webcam index,
a local video file or an ``rtsp://`` URL - nothing else is ever opened. Neither
can change a module's parameters or reach beyond the video sources.

With several cameras every read endpoint takes ``?camera=<id>``; without it
they answer for the first camera, exactly as a single-camera install always did.

Live video is MJPEG rather than WebRTC or websockets: a browser renders it
with a plain ``<img>`` tag, there is no client-side buffering to drift behind,
and it needs no JavaScript at all to work.

Access control (read-only does not mean harmless - the stream is a live
camera, and one endpoint now writes):

* **Host check.** Every request's ``Host`` header must name this server.
  Without it, a web page the operator visits can use DNS rebinding to read a
  dashboard bound to ``127.0.0.1``.
* **Token.** Optional on localhost. Mandatory on a network bind: the
  configured ``dashboard.auth_token``, or a random one generated per run.
  Accepted as ``Authorization: Bearer``, a ``?token=`` query parameter
  (exchanged for an HttpOnly cookie so the browser's ``<img>`` and ``fetch``
  calls keep working), or that cookie.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import secrets
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Iterable, Optional, Sequence
from urllib.parse import parse_qs, quote, unquote

from backend.services.camera_manager import CameraManager
from backend.services.hub import CameraHub
from backend.services.reload import ReloadController
from backend.services.state import DashboardState
from computer_vision.utils.logger import get_logger

logger = get_logger(__name__)

#: The frontend is a sibling of the backend, not a resource inside it, so the
#: UI can be replaced or served by a real web server without touching Python.
STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend" / "static"
_BOUNDARY = "visionguard-frame"

#: Name of the cookie that carries the token after a ``?token=`` visit.
TOKEN_COOKIE = "vg_dashboard_token"

_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::", "*"})

#: Applied to every response. The page uses one inline <script> and <style>,
#: so those are allowed from the page itself and nothing else is.
_SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store"),
    (
        b"content-security-policy",
        b"default-src 'none'; script-src 'self' 'unsafe-inline'; "
        b"style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        b"connect-src 'self'; base-uri 'none'; form-action 'none'; "
        b"frame-ancestors 'none'",
    ),
)


class DashboardUnavailableError(RuntimeError):
    """FastAPI/uvicorn are not installed, or the server could not start."""


# --------------------------------------------------------------------------- #
# Access policy
# --------------------------------------------------------------------------- #

def is_loopback_host(host: str) -> bool:
    """True when ``host`` only accepts connections from this machine."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class AccessPolicy:
    """Who may talk to the dashboard. Built once at startup."""

    #: Accepted ``Host`` names (lower-case, no port). Empty = accept any.
    allowed_hosts: FrozenSet[str] = field(default_factory=lambda: _LOOPBACK_NAMES)
    #: Required token, or None for no token.
    token: Optional[str] = field(default=None, repr=False)
    #: True when the token was generated for this run (it must be shown once).
    token_generated: bool = False

    @classmethod
    def for_bind(
        cls,
        host: str,
        token: Optional[str] = None,
        extra_hosts: Iterable[str] = (),
    ) -> "AccessPolicy":
        """The policy for a server bound to ``host``.

        Localhost binds keep working exactly as before (no token needed) but
        now reject foreign ``Host`` headers. Network binds always require a
        token; if none is configured a strong random one is generated, since
        an unauthenticated network-exposed camera feed is never intended.
        """
        extra = {h.strip().lower().strip("[]") for h in extra_hosts if h.strip()}
        generated = False

        if is_loopback_host(host):
            hosts = frozenset(_LOOPBACK_NAMES | extra)
        else:
            if not token:
                token = secrets.token_urlsafe(24)
                generated = True
            if host in _WILDCARD_HOSTS:
                # Listening everywhere: any name may reach it, so the token -
                # not the Host header - is the guard, unless the operator
                # pinned the names in dashboard.allowed_hosts.
                hosts = frozenset(_LOOPBACK_NAMES | extra) if extra else frozenset()
            else:
                hosts = frozenset(_LOOPBACK_NAMES | extra | {host.lower().strip("[]")})

        return cls(allowed_hosts=hosts, token=token or None, token_generated=generated)


def _host_name(raw: str) -> str:
    """``Host`` header value without the port, lower-cased."""
    raw = raw.strip().lower()
    if raw.startswith("["):  # [::1]:8080
        end = raw.find("]")
        return raw[1:end] if end != -1 else raw
    if raw.count(":") == 1:
        return raw.split(":", 1)[0]
    return raw


class _AccessMiddleware:
    """Pure-ASGI guard: Host check, token check, security headers.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so it never buffers or breaks
    the long-lived MJPEG stream.
    """

    def __init__(self, app, policy: AccessPolicy) -> None:
        self.app = app
        self.policy = policy

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}

        if self.policy.allowed_hosts:
            host = _host_name(headers.get(b"host", b"").decode("latin-1"))
            if host not in self.policy.allowed_hosts:
                await _plain(send, 400, b"Invalid Host header")
                return

        set_cookie: Optional[bytes] = None
        if self.policy.token:
            presented, from_query = _presented_token(scope, headers)
            if presented is None or not hmac.compare_digest(
                presented.encode("utf-8"), self.policy.token.encode("utf-8")
            ):
                await _plain(
                    send,
                    401,
                    b"Unauthorized: open the dashboard URL printed at startup, "
                    b"or send 'Authorization: Bearer <token>'.",
                    extra=[(b"www-authenticate", b"Bearer")],
                )
                return
            if from_query:
                set_cookie = (
                    f"{TOKEN_COOKIE}={quote(presented, safe='')}; Path=/; "
                    "HttpOnly; SameSite=Strict"
                ).encode("latin-1")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                out = list(message.get("headers", []))
                present = {k.lower() for k, _ in out}
                out.extend(h for h in _SECURITY_HEADERS if h[0] not in present)
                if set_cookie is not None:
                    out.append((b"set-cookie", set_cookie))
                message = {**message, "headers": out}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _presented_token(scope, headers) -> tuple[Optional[str], bool]:
    """``(token, came_from_query)`` from the request, or ``(None, False)``."""
    auth = headers.get(b"authorization", b"").decode("latin-1")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip(), False

    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    if query.get("token"):
        return query["token"][0], True

    cookie_header = headers.get(b"cookie", b"").decode("latin-1")
    for part in cookie_header.split(";"):
        name, _, value = part.strip().partition("=")
        if name == TOKEN_COOKIE and value:
            return unquote(value), False
    return None, False


async def _plain(send, status: int, body: bytes, extra: Sequence = ()) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
                *extra,
                *_SECURITY_HEADERS,
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

class _StreamSlots:
    """Counts live MJPEG viewers so they cannot exhaust the server."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._active = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self._active >= self.limit:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def active(self) -> int:
        with self._lock:
            return self._active


def build_app(
    state: DashboardState,
    *,
    policy: AccessPolicy | None = None,
    max_streams: int = 4,
    stream_fps: float = 15.0,
    reload: ReloadController | None = None,
    known_modules: FrozenSet[str] = frozenset(),
    hub: CameraHub | None = None,
    cameras: CameraManager | None = None,
):
    """Create the FastAPI application bound to ``state``.

    ``policy`` defaults to the localhost policy: no token, loopback Host
    names only. ``reload`` wires up ``POST /api/modules``; leaving it unset
    (the default for anything that builds an app directly, e.g. tests) makes
    that endpoint answer 503 rather than silently accepting a change nothing
    will apply.

    ``hub`` makes the read endpoints camera-aware (``?camera=<id>``; the first
    camera when omitted). Without one, ``state`` is the only camera. ``cameras``
    wires up ``/api/cameras``; without it the endpoint lists the one camera and
    refuses changes.
    """
    try:
        from fastapi import Body, FastAPI, Query
        from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
    except ImportError as exc:  # pragma: no cover - depends on install
        raise DashboardUnavailableError(
            "The dashboard needs FastAPI and uvicorn:\n"
            "    pip install -r requirements.txt\n"
            "Or run with dashboard.enabled: false."
        ) from exc

    policy = policy or AccessPolicy()
    slots = _StreamSlots(max_streams)
    # Poll a little faster than frames are encoded; never busier than 50 Hz.
    poll_interval = min(0.1, max(0.02, 0.5 / stream_fps)) if stream_fps > 0 else 0.02

    app = FastAPI(title="VisionGuard AI", docs_url=None, redoc_url=None)

    def state_for(camera: Optional[str]) -> Optional[DashboardState]:
        """The state for ``camera``; None when there is no such camera."""
        if hub is None:
            return state if camera is None else None
        return hub.get(camera)

    def no_such_camera() -> "JSONResponse":
        return JSONResponse({"error": "unknown camera"}, status_code=404)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status(camera: Optional[str] = Query(None, max_length=64)):
        chosen = state_for(camera)
        if chosen is None:
            return no_such_camera()
        return JSONResponse(chosen.snapshot())

    @app.get("/api/events")
    def events(
        since: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
        camera: Optional[str] = Query(None, max_length=64),
    ):
        chosen = state_for(camera)
        if chosen is None:
            return no_such_camera()
        return JSONResponse(
            {"events": chosen.events_since(since, limit), "last_id": chosen.last_event_id}
        )

    @app.get("/api/cameras")
    def list_cameras():
        if cameras is not None:
            return JSONResponse(cameras.listing())
        # No manager (a bare app, e.g. tests): describe the single camera.
        return JSONResponse(
            {
                "cameras": hub.describe() if hub is not None else [],
                "max_cameras": 1,
                "editable": False,
                "apply_error": None,
            }
        )

    @app.post("/api/cameras", include_in_schema=False)
    def set_cameras(payload: dict = Body(...)):
        """Replace the camera list. Queued, like ``POST /api/modules``.

        ``payload`` is ``{"cameras": [{"id"?, "name"?, "source", "rtsp_url" |
        "path" | "device_id"}, ...]}`` - the complete list that should run.
        Validation errors come back as 400 without echoing any credentials.
        """
        if cameras is None:
            return JSONResponse(
                {"error": "This run was started without --dashboard camera support."},
                status_code=503,
            )
        code, body = cameras.submit(payload.get("cameras"))
        return JSONResponse(body, status_code=code)

    @app.post("/api/modules", include_in_schema=False)
    def set_modules(payload: dict = Body(...)):
        """Replace the enabled module set. The only write this server does.

        ``payload`` is ``{"enabled": [<module name>, ...]}`` - the complete
        set that should be on, dependencies included; anything left out is
        turned off. Accepting the request only means it was queued: the
        pipeline thread rebuilds the modules and restarts between frames,
        which takes a few seconds and can still fail (e.g. a dependency not
        included, or a model file missing) without this call knowing that yet.
        """
        if reload is None:
            return JSONResponse(
                {"error": "This run was started without --dashboard module reload support."},
                status_code=503,
            )
        desired = payload.get("enabled")
        if not isinstance(desired, list) or not all(isinstance(x, str) for x in desired):
            return JSONResponse(
                {"error": "'enabled' must be a list of module name strings"}, status_code=400
            )
        unknown = sorted(set(desired) - known_modules)
        if unknown:
            return JSONResponse(
                {"error": f"Unknown module(s): {', '.join(unknown)}"}, status_code=400
            )
        reload.request(desired)
        return JSONResponse({"status": "applying", "enabled": desired}, status_code=202)

    @app.get("/api/snapshot.jpg", include_in_schema=False)
    def snapshot(camera: Optional[str] = Query(None, max_length=64)):
        chosen = state_for(camera)
        if chosen is None:
            return Response(status_code=404, content=b"unknown camera")
        jpeg = chosen.latest_jpeg()
        if jpeg is None:
            return Response(status_code=503, content=b"no frame yet")
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/stream.mjpg", include_in_schema=False)
    async def stream(camera: Optional[str] = Query(None, max_length=64)):
        chosen = state_for(camera)
        if chosen is None:
            return Response(status_code=404, content=b"unknown camera")
        if slots.active >= slots.limit:
            return Response(
                status_code=503,
                content=b"too many live viewers",
                headers={"Retry-After": "5"},
            )
        return StreamingResponse(
            _mjpeg_frames(chosen, slots, poll_interval),
            media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        )

    app.state.stream_slots = slots
    return _AccessMiddleware(app, policy)


#: Longest a live stream stays silent before the current frame is sent again.
_STREAM_KEEPALIVE_SEC = 2.0


async def _mjpeg_frames(state: DashboardState, slots: _StreamSlots, poll_interval: float):
    """Yield multipart JPEG chunks, one per new frame.

    Asynchronous on purpose: the previous generator blocked a worker thread
    per viewer, so a few dozen open streams starved every other endpoint.
    Waiting with ``asyncio.sleep`` costs no thread at all.
    """
    # The slot is taken here, not in the endpoint: if the client goes away
    # before streaming starts, an un-started generator never runs `finally`
    # and a slot taken earlier would leak for good.
    if not slots.acquire():
        return
    last_seq = 0
    last_sent = time.monotonic()
    try:
        while True:
            jpeg, seq = state.latest_frame()
            now = time.monotonic()
            # Repeating the current picture while nothing new arrives keeps
            # the connection visibly alive (a camera reconnecting, a slow
            # frame), and is also how a viewer that has gone away is noticed
            # and its slot freed instead of being held until the next frame.
            repeat = jpeg is not None and (now - last_sent) >= _STREAM_KEEPALIVE_SEC
            if jpeg is not None and (seq > last_seq or repeat):
                last_seq = seq
                last_sent = now
                yield (
                    f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
            if not state.running:
                break
            await asyncio.sleep(poll_interval)
    finally:
        slots.release()


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #

def _bind_socket(host: str, port: int) -> socket.socket:
    """Bind the listening socket on the calling thread.

    Binding here - not inside uvicorn's thread - is what lets "port already in
    use" reach the caller as an ``OSError``. Uvicorn reports that failure with
    ``sys.exit`` on its own thread, which silently kills only that thread.
    """
    bind_host = "0.0.0.0" if host in ("", "*") else host.strip("[]")
    # Same address-family rule uvicorn uses: IPv6 only for an IPv6 literal,
    # so "localhost" keeps binding 127.0.0.1 exactly as before.
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    infos = socket.getaddrinfo(
        bind_host, port, family=family, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    family, socktype, proto, _, address = infos[0]
    sock = socket.socket(family, socktype, proto)
    try:
        if sys.platform != "win32":
            # POSIX: allow a quick restart over TIME_WAIT. Not on Windows,
            # where SO_REUSEADDR lets another process steal a bound port.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(address)
        sock.listen(128)
        sock.setblocking(False)
    except OSError:
        sock.close()
        raise
    return sock


class DashboardServer:
    """Runs the dashboard on a daemon thread alongside the pipeline."""

    #: How long start() waits for uvicorn to report it is serving.
    START_TIMEOUT_SEC = 10.0

    def __init__(
        self,
        state: DashboardState,
        host: str,
        port: int,
        *,
        policy: AccessPolicy | None = None,
        max_streams: int = 4,
        stream_fps: float = 15.0,
        reload: ReloadController | None = None,
        known_modules: FrozenSet[str] = frozenset(),
        hub: CameraHub | None = None,
        cameras: CameraManager | None = None,
    ) -> None:
        self._hub = hub
        self._cameras = cameras
        self._state = state
        self._host = host
        self._port = port
        self._policy = policy or AccessPolicy.for_bind(host)
        self._max_streams = max_streams
        self._stream_fps = stream_fps
        self._reload = reload
        self._known_modules = known_modules
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._socket: Optional[socket.socket] = None

    @property
    def policy(self) -> AccessPolicy:
        return self._policy

    @property
    def state(self) -> DashboardState:
        """The shared state object, so ``backend.main`` can update ``info``
        and flip ``running`` back on after rebuilding modules in place."""
        return self._state

    @property
    def url(self) -> str:
        """The dashboard address, without any token (safe to log)."""
        shown = "localhost" if self._host in _WILDCARD_HOSTS else self._host
        if ":" in shown:
            shown = f"[{shown}]"
        return f"http://{shown}:{self._port}"

    def url_with_token(self) -> str:
        """Address including ``?token=``. Show it once; never log it."""
        if not self._policy.token:
            return self.url
        return f"{self.url}/?token={quote(self._policy.token, safe='')}"

    def start(self) -> None:
        """Bind, then serve on a background thread.

        Raises:
            OSError: the address could not be bound (e.g. port in use).
            DashboardUnavailableError: uvicorn is missing or failed to start.
        """
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - depends on install
            raise DashboardUnavailableError(
                "The dashboard needs uvicorn: pip install -r requirements.txt"
            ) from exc

        app = build_app(
            self._state,
            policy=self._policy,
            max_streams=self._max_streams,
            stream_fps=self._stream_fps,
            reload=self._reload,
            known_modules=self._known_modules,
            hub=self._hub,
            cameras=self._cameras,
        )
        self._socket = _bind_socket(self._host, self._port)  # OSError -> caller

        config = uvicorn.Config(app, log_level="warning", access_log=False, lifespan="off")
        self._server = uvicorn.Server(config)

        errors: list[BaseException] = []

        def serve() -> None:
            try:
                self._server.run(sockets=[self._socket])
            except BaseException as exc:  # noqa: BLE001 - incl. SystemExit from uvicorn
                errors.append(exc)

        self._thread = threading.Thread(target=serve, name="visionguard-dashboard", daemon=True)
        self._thread.start()

        deadline = time.monotonic() + self.START_TIMEOUT_SEC
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                self._close_socket()
                reason = f": {errors[0]!r}" if errors else ""
                raise DashboardUnavailableError(
                    f"Dashboard failed to start on {self._host}:{self._port}{reason}"
                )
            time.sleep(0.02)

        logger.info("Dashboard: %s", self.url)
        if not is_loopback_host(self._host):
            logger.warning(
                "Dashboard is listening on %s - reachable from the network. "
                "Access requires the dashboard token%s.",
                self._host,
                " generated for this run" if self._policy.token_generated else "",
            )

    def stop(self) -> None:
        self._state.stop()
        if self._hub is not None:
            self._hub.stop_all()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._close_socket()

    def _close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:  # pragma: no cover
                pass
            self._socket = None
