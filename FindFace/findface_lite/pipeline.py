"""Video -> motion gate -> detector -> tracker -> quality filter -> best snapshot -> hooks -> sinks.

Hooks (e.g. recognition.Recognizer) see each posted event with its full frame
and may add fields to it before the sinks run."""
from __future__ import annotations

import time
from dataclasses import asdict
from itertools import count
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from .config import Config, Rect
from .detector import Detection, FaceDetector
from .motion import MotionGate
from .quality import Quality, assess
from .snapshot import Candidate, Post, SnapshotSelector
from .tracker import Track, Tracker


def _inside(rect: Rect, x: float, y: float) -> bool:
    w, h, rx, ry = rect
    return rx <= x < rx + w and ry <= y < ry + h


def _crop(frame: np.ndarray, bbox: np.ndarray, margin: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    return frame[y1:y2, x1:x2].copy()


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


class FacePipeline:
    def __init__(self, cfg: Config, sinks: Iterable = (), source_name: str = "camera",
                 detector: Optional[FaceDetector] = None,
                 hooks: Iterable[Callable[[dict, np.ndarray], None]] = ()):
        self.cfg = cfg
        self.sinks = list(sinks)
        self.hooks = list(hooks)
        self.source_name = source_name
        self.motion = MotionGate(cfg.imotion_threshold, cfg.motion_scale)
        self.detector = detector or FaceDetector(cfg.detector_model, cfg.detector_score_threshold,
                                                 cfg.detector_nms_threshold, cfg.resize_scale)
        self.tracker = Tracker(cfg.track_overlap_threshold, cfg.track_miss_interval,
                               cfg.track_max_duration_frames, cfg.track_interpolate_bboxes,
                               cfg.max_tracks)
        self.selector = SnapshotSelector(cfg.overall_only, cfg.realtime_post_interval,
                                         cfg.realtime_post_every_interval,
                                         cfg.realtime_post_first_immediately)
        self._entered_roi: set[int] = set()
        self._first_last: dict[int, list] = {}      # track_id -> [(t, frame), (t, frame)]
        self._event_ids = count(1)
        self.stats = {"frames": 0, "detector_runs": 0, "detections": 0,
                      "tracks": 0, "events": 0, "rejected": {}}
        self.last_overlay: list[tuple[Track, Detection, Quality, bool]] = []

    # ------------------------------------------------------------------ core --
    def process(self, frame: np.ndarray, frame_idx: int, t: float) -> list[dict]:
        cfg = self.cfg
        self.stats["frames"] += 1
        dets: list[Detection] = []
        if self.motion(frame):
            region, offset = frame, (0, 0)
            if cfg.rot:
                w, h, x, y = cfg.rot
                region, offset = frame[y:y + h, x:x + w], (x, y)
            dets = self.detector.detect(region, offset)
            self.stats["detector_runs"] += 1
            self.stats["detections"] += len(dets)

        matched, closed = self.tracker.update(dets, frame_idx, t)
        events: list[dict] = []
        self.last_overlay = []
        for track, det in matched:
            if track.id not in self._first_last:
                self.stats["tracks"] += 1
                self._first_last[track.id] = [(frame_idx, t, frame), None]
            self._first_last[track.id][1] = (frame_idx, t, frame)
            q = assess(frame, det)
            ok = self._passes_filters(track, det, q)
            self.last_overlay.append((track, det, q, ok))
            if not ok:
                continue
            cand = Candidate(q.score, frame_idx, t,
                             {"bbox": det.bbox, "landmarks": det.landmarks, "quality": q, "frame": frame})
            events += self._emit(self.selector.offer(track.id, cand), track)

        events += self._emit(self.selector.tick(t), None)
        for track in closed:
            events += self._close(track)
        return events

    def finish(self) -> list[dict]:
        events = []
        for track in self.tracker.close_all():
            events += self._close(track)
        for sink in self.sinks:
            sink.close()
        return events

    # -------------------------------------------------------------- helpers --
    def _reject(self, why: str) -> bool:
        self.stats["rejected"][why] = self.stats["rejected"].get(why, 0) + 1
        return False

    def _passes_filters(self, track: Track, det: Detection, q: Quality) -> bool:
        cfg = self.cfg
        if cfg.roi:
            cx, cy = (det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2
            if _inside(cfg.roi, cx, cy):
                self._entered_roi.add(track.id)
            if track.id not in self._entered_roi:
                return self._reject("outside_roi")
        if min(det.width, det.height) < cfg.filter_min_size:
            return self._reject("too_small")
        if max(det.width, det.height) > cfg.filter_max_size:
            return self._reject("too_large")
        if max(abs(q.yaw), abs(q.pitch)) > cfg.filter_max_angle:
            return self._reject("angle")
        if q.score < cfg.filter_min_quality:
            return self._reject("low_quality")
        if track.hits < cfg.track_min_length:
            return self._reject("track_too_short")
        return True

    def _close(self, track: Track) -> list[dict]:
        posts, _ = self.selector.close(track.id)
        if track.hits < self.cfg.track_min_length:
            posts = []
        events = self._emit(posts, track)
        self._entered_roi.discard(track.id)
        self._first_last.pop(track.id, None)
        return events

    def _emit(self, posts: list[Post], track: Optional[Track]) -> list[dict]:
        cfg, events = self.cfg, []
        for post in posts:
            tr = track if track and track.id == post.track_id else self.tracker.active.get(post.track_id)
            c = post.candidate
            p = c.payload
            q: Quality = p["quality"]
            event = {
                "event_id": next(self._event_ids),
                "source": self.source_name,
                "track_id": post.track_id,
                "reason": post.reason,
                "frame_idx": c.frame_idx,
                "timestamp": round(c.t, 3),
                "wall_time": time.time(),
                "bbox": [round(float(v), 1) for v in p["bbox"]],
                "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in p["landmarks"]],
                "quality": {k: round(v, 4) for k, v in asdict(q).items()},
            }
            if tr is not None:
                event["track"] = {"first_frame": tr.first_frame, "last_frame": tr.last_frame,
                                  "hits": tr.hits, "closed": tr.close_reason}
                if cfg.track_send_history:
                    event["track"]["history"] = [[i, [round(float(v), 1) for v in b], d]
                                                 for i, b, d in tr.history]
            for hook in self.hooks:
                hook(event, p["frame"])
            images = {"face": _crop(p["frame"], p["bbox"], cfg.crop_margin)}
            if cfg.post_best_track_frame:
                images["full_frame"] = p["frame"]
            fl = self._first_last.get(post.track_id)
            if fl and cfg.post_first_track_frame:
                images["first_frame"] = fl[0][2]
            if fl and fl[1] and cfg.post_last_track_frame:
                images["last_frame"] = fl[1][2]
            for sink in self.sinks:
                sink(event, images)
            self.stats["events"] += 1
            events.append(event)
        return events

    # -------------------------------------------------------------- drawing --
    def draw(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        for rect, colour, label in ((self.cfg.rot, (255, 160, 0), "ROT"), (self.cfg.roi, (0, 200, 255), "ROI")):
            if rect:
                w, h, x, y = rect
                cv2.rectangle(out, (x, y), (x + w, y + h), colour, 1)
                cv2.putText(out, label, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        for track, det, q, ok in self.last_overlay:
            col = _color(track.id)
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 if ok else 1)
            pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b, _ in track.history[-40:]]
            for a, b in zip(pts, pts[1:]):
                cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), col, 1)
            for lx, ly in det.landmarks:
                cv2.circle(out, (int(lx), int(ly)), 2, col, -1)
            cv2.putText(out, f"#{track.id} q={q.score:.2f}{'' if ok else ' x'}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(out, f"motion {self.motion.last_intensity:.4f}  tracks {len(self.tracker.active)}",
                    (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out


def run_video(pipeline: FacePipeline, source, show: bool = False, max_frames: int = 0,
              write_preview: Optional[str] = None, loop: bool = False) -> list[dict]:
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    live = str(source).isdigit() or "://" in str(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = 1 if live else max(1, int(1000 / fps))
    writer, events, idx, start = None, [], 0, time.monotonic()
    last_vis = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok and loop and not live:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or (max_frames and idx >= max_frames):
                if show and last_vis is not None:
                    finished = last_vis.copy()
                    cv2.putText(finished, "Finished - Press any key to exit", (16, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("findface-lite", finished)
                    cv2.waitKey(0)
                break
            t = time.monotonic() - start if live else idx / fps
            events += pipeline.process(frame, idx, t)
            if show or write_preview:
                vis = pipeline.draw(frame)
                last_vis = vis
                if write_preview:
                    if writer is None:
                        writer = cv2.VideoWriter(write_preview, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if show:
                    cv2.imshow("findface-lite", vis)
                    if cv2.waitKey(delay_ms) & 0xFF in (27, ord("q")):
                        break
            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        events += pipeline.finish()
    return events
