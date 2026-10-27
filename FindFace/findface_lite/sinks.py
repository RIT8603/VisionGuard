"""Where posted face events go: a folder, a webhook, or your own callback."""
from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path
from typing import Callable

import cv2


class DirectorySink:
    """Writes events.jsonl plus face crops and full frames as JPEG files."""

    def __init__(self, out_dir: str | Path, jpeg_quality: int = 95):
        self.root = Path(out_dir)
        (self.root / "faces").mkdir(parents=True, exist_ok=True)
        (self.root / "frames").mkdir(parents=True, exist_ok=True)
        self.params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        self._log = open(self.root / "events.jsonl", "a", encoding="utf-8")

    def __call__(self, event: dict, images: dict) -> None:
        record = dict(event)
        for name, img in images.items():
            folder = "faces" if name == "face" else "frames"
            path = self.root / folder / f"{event['event_id']}_{name}.jpg"
            cv2.imwrite(str(path), img, self.params)
            record[f"{name}_path"] = str(path.relative_to(self.root)).replace("\\", "/")
        self._log.write(json.dumps(record) + "\n")
        self._log.flush()

    def close(self) -> None:
        self._log.close()


class WebhookSink:
    """POSTs each event as JSON; images are base64 JPEG (like FindFace webhooks)."""

    def __init__(self, url: str, jpeg_quality: int = 95, timeout: float = 3.0):
        self.url = url
        self.params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        self.timeout = timeout

    def __call__(self, event: dict, images: dict) -> None:
        body = dict(event)
        for name, img in images.items():
            ok, buf = cv2.imencode(".jpg", img, self.params)
            if ok:
                body[f"{name}_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=self.timeout).close()
        except OSError as exc:  # never let a slow receiver stall the video loop
            print(f"[webhook] post failed: {exc}")

    def close(self) -> None:
        pass


class CallbackSink:
    def __init__(self, fn: Callable[[dict, dict], None]):
        self.fn = fn

    def __call__(self, event: dict, images: dict) -> None:
        self.fn(event, images)

    def close(self) -> None:
        pass
