#!/usr/bin/env python3
"""Launcher for FindFace Lite Web UI with automatic browser opening."""
import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from findface_lite.web import run_web


def find_free_port(start_port: int = 5000) -> int:
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port


def main():
    parser = argparse.ArgumentParser(description="Launch FindFace Lite Web UI")
    parser.add_argument("--source", default="samples/demo.mp4",
                        help="Video file or camera index (default: samples/demo.mp4)")
    parser.add_argument("--camera", action="store_true", help="Shortcut to use webcam 0 as source")
    parser.add_argument("--port", type=int, default=0, help="Server port (default: auto-pick around 5000)")
    parser.add_argument("--mode", choices=["offline", "realtime"], default="offline")
    parser.add_argument("--watchlist", help="Directory of enrolled photos")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    source = "0" if args.camera else args.source
    port = args.port if args.port > 0 else find_free_port(5000)
    url = f"http://localhost:{port}"

    if not args.no_browser:
        def open_browser():
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    run_web(source=source, port=port, watchlist=args.watchlist, mode=args.mode, loop=True)


if __name__ == "__main__":
    main()
