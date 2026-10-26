"""Recognition: embed each posted best face and match it against a watchlist.

This is the stage FindFace Multi runs after the video worker posts a face:
the face is aligned by its landmarks, turned into a feature vector, and the
vector is compared with every enrolled card. Here the extractor is OpenCV
SFace (128-d) and comparison is cosine similarity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import FaceDetector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# OpenCV Zoo SFace: cosine >= 0.363 means "same person".
SFACE_COSINE_THRESHOLD = 0.363


def read_image(path: str | Path) -> Optional[np.ndarray]:
    """cv2.imread that also works with non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


class FaceEmbedder:
    """OpenCV SFace: aligned 112x112 crop -> L2-normalised 128-d vector."""

    def __init__(self, model_path: str):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Recognition model not found: {model_path}")
        self._net = cv2.FaceRecognizerSF.create(model_path, "")

    def embed(self, frame: np.ndarray, bbox, landmarks) -> np.ndarray:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        # alignCrop wants a YuNet row: x, y, w, h, 5 landmark (x, y) pairs, score
        row = np.array([x1, y1, x2 - x1, y2 - y1,
                        *np.asarray(landmarks, dtype=np.float32).reshape(-1), 1.0], dtype=np.float32)
        aligned = self._net.alignCrop(frame, row.reshape(1, -1))
        feat = self._net.feature(aligned).reshape(-1).astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-12)


@dataclass
class Match:
    name: str
    similarity: float


class Watchlist:
    """Enrolled people. Several photos of one person share a name."""

    def __init__(self, threshold: float = SFACE_COSINE_THRESHOLD):
        self.threshold = threshold
        self.names: list[str] = []
        self._vectors = np.zeros((0, 128), np.float32)

    def __len__(self) -> int:
        return len(self.names)

    def add(self, name: str, vector: np.ndarray) -> None:
        self.names.append(name)
        self._vectors = np.vstack([self._vectors, np.asarray(vector, np.float32).reshape(1, -1)])

    def match(self, vector: np.ndarray) -> Optional[Match]:
        if not self.names:
            return None
        sims = self._vectors @ vector
        best = int(np.argmax(sims))
        if sims[best] < self.threshold:
            return None
        return Match(self.names[best], float(sims[best]))

    @classmethod
    def from_dir(cls, folder: str | Path, detector: FaceDetector, embedder: FaceEmbedder,
                 threshold: float = SFACE_COSINE_THRESHOLD) -> "Watchlist":
        """Enrol every photo in `folder`; the person's name is the file name."""
        wl = cls(threshold)
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Watchlist folder not found: {folder}")
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            img = read_image(path)
            dets = detector.detect(img) if img is not None else []
            if not dets:
                print(f"[watchlist] no face found in {path.name}, skipped")
                continue
            det = max(dets, key=lambda d: d.width * d.height)   # the main face in the photo
            wl.add(path.stem, embedder.embed(img, det.bbox, det.landmarks))
        return wl


class Recognizer:
    """Pipeline hook: adds `match: {name, similarity}` (or None) to each event."""

    def __init__(self, embedder: FaceEmbedder, watchlist: Watchlist, post_embedding: bool = False):
        self.embedder = embedder
        self.watchlist = watchlist
        self.post_embedding = post_embedding

    def __call__(self, event: dict, frame: np.ndarray) -> None:
        vec = self.embedder.embed(frame, event["bbox"], event["landmarks"])
        m = self.watchlist.match(vec)
        event["match"] = {"name": m.name, "similarity": round(m.similarity, 4)} if m else None
        if self.post_embedding:
            event["embedding"] = [round(float(v), 6) for v in vec]

    @classmethod
    def from_config(cls, cfg, watchlist_dir: str | Path,
                    detector: Optional[FaceDetector] = None) -> "Recognizer":
        embedder = FaceEmbedder(cfg.recognizer_model)
        detector = detector or FaceDetector(cfg.detector_model, cfg.detector_score_threshold,
                                            cfg.detector_nms_threshold)
        wl = Watchlist.from_dir(watchlist_dir, detector, embedder, cfg.match_threshold)
        return cls(embedder, wl, cfg.post_embedding)
