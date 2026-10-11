"""Say *why* an RTSP stream would not open.

OpenCV's FFmpeg backend reports every RTSP failure the same way: the capture
simply is not opened. That leaves an operator guessing between a blocked
network, a wrong address, an expired token and a codec problem. This module
asks the server directly - one TCP connect and one RTSP ``DESCRIBE`` - and
turns the answer into one plain sentence.

It never logs or returns the URL itself: callers add the masked form.
"""

from __future__ import annotations

import socket
from typing import Optional
from urllib.parse import urlsplit

#: RTSP status codes worth naming; anything else is reported by number.
_STATUS_HINTS = {
    401: "the server wants a user name and password (or they are wrong)",
    403: "the server refused access - the token or password is wrong or has expired",
    404: "the server has no stream at this path - check the path after the host name",
    453: "the server is busy or out of connections (free test servers allow very few)",
    454: "the server says this session is gone - the link has probably expired",
    461: "the server does not support this transport - try rtsp_transport: udp",
}


def _default_port(scheme: str) -> int:
    return 322 if scheme.lower() == "rtsps" else 554


def probe_rtsp(url: str, timeout: float = 5.0) -> Optional[str]:
    """One sentence on why ``url`` cannot be played, or None if it looks fine.

    None also covers "could not tell" (an unusual reply): the caller then keeps
    its generic message. Never raises.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port or _default_port(parts.scheme)
    except ValueError:
        return "the address could not be read - check for typos"
    if not host:
        return "the address has no host name"

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return f"the host name '{host}' could not be found (DNS) - check the spelling and the internet connection"

    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except socket.timeout:
        return (
            f"{host}:{port} did not answer - the network or a firewall is blocking it, "
            "or the camera is off"
        )
    except ConnectionRefusedError:
        return f"{host}:{port} refused the connection - nothing is serving RTSP on that port"
    except OSError as exc:
        return f"{host}:{port} could not be reached ({exc.strerror or exc})"

    if parts.scheme.lower() == "rtsps":
        sock.close()
        return None  # TLS: reachable is as far as a plain probe can tell

    try:
        with sock:
            sock.settimeout(timeout)
            request = (
                f"DESCRIBE {url} RTSP/1.0\r\n"
                "CSeq: 1\r\n"
                "Accept: application/sdp\r\n"
                "User-Agent: VisionGuard-probe\r\n\r\n"
            )
            sock.sendall(request.encode("utf-8", "replace"))
            reply = sock.recv(1024).decode("latin-1", "replace")
    except socket.timeout:
        return f"{host}:{port} accepted the connection but never answered the RTSP request"
    except OSError as exc:
        return f"{host}:{port} dropped the connection ({exc.strerror or exc})"

    status_line = reply.split("\r\n", 1)[0].split()
    if len(status_line) < 2 or not status_line[0].startswith("RTSP/"):
        return f"{host}:{port} is not an RTSP server (it answered something else)"
    try:
        code = int(status_line[1])
    except ValueError:
        return None
    if 200 <= code < 300:
        return None
    hint = _STATUS_HINTS.get(code)
    if hint:
        return f"RTSP {code}: {hint}"
    return f"the server answered RTSP {code} {' '.join(status_line[2:])}".rstrip()
