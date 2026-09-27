"""Frame annotation helpers.

Drawing is intentionally separate from both the pipeline and the AI modules: a
module returns structured data, and the annotator decides how that data looks.
In Phase 1 only the HUD is drawn — detection and track overlays arrive with the
modules that produce them.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

BGR = Tuple[int, int, int]

COLOR_TEXT: BGR = (235, 235, 235)
COLOR_ACCENT: BGR = (0, 210, 255)
COLOR_OK: BGR = (90, 220, 120)
COLOR_MUTED: BGR = (160, 160, 160)
COLOR_PANEL: BGR = (20, 20, 20)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_panel(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: BGR = COLOR_PANEL,
    alpha: float = 0.55,
) -> None:
    """Blend a translucent rectangle into ``frame`` in place."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return

    region = frame[y1:y2, x1:x2]
    overlay = np.full(region.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, region, 1.0 - alpha, 0.0, dst=region)


def draw_hud(
    frame: np.ndarray,
    lines: Sequence[Tuple[str, BGR, float]],
    origin: Tuple[int, int] = (12, 12),
    line_height: int = 22,
    padding: int = 10,
) -> np.ndarray:
    """Draw a text panel in the top-left corner.

    Args:
        frame: Image to draw on (modified in place).
        lines: ``(text, bgr_color, font_scale)`` triples, top to bottom.
        origin: Panel top-left corner.
        line_height: Vertical spacing between baselines.
        padding: Inner padding of the panel.

    Returns:
        The same ``frame``, for chaining.
    """
    if not lines:
        return frame

    widths: List[int] = []
    for text, _color, scale in lines:
        (tw, _th), _ = cv2.getTextSize(text, _FONT, scale, 1)
        widths.append(tw)

    panel_w = max(widths) + padding * 2
    panel_h = line_height * len(lines) + padding * 2 - 6
    ox, oy = origin

    draw_panel(frame, (ox, oy), (ox + panel_w, oy + panel_h))

    y = oy + padding + 12
    for text, color, scale in lines:
        cv2.putText(frame, text, (ox + padding, y), _FONT, scale, color, 1, cv2.LINE_AA)
        y += line_height

    return frame


#: Stable, readable colours cycled by label so each class looks consistent.
_LABEL_PALETTE: Tuple[BGR, ...] = (
    (90, 220, 120),    # green
    (0, 210, 255),     # amber
    (255, 170, 80),    # blue
    (200, 120, 255),   # pink
    (255, 220, 90),    # cyan
)


#: Wider palette for track ids, so neighbouring people are easy to tell apart.
_TRACK_PALETTE: Tuple[BGR, ...] = (
    (90, 220, 120),    # green
    (0, 210, 255),     # amber
    (255, 170, 80),    # blue
    (200, 120, 255),   # pink
    (255, 220, 90),    # cyan
    (120, 120, 255),   # red
    (255, 120, 200),   # violet
    (140, 255, 255),   # yellow
)


def color_for_label(label: str) -> BGR:
    """Deterministic colour for a class label."""
    return _LABEL_PALETTE[sum(label.encode()) % len(_LABEL_PALETTE)]


def color_for_track_id(track_id: int, source_module: str = "") -> BGR:
    """Deterministic, stable colour for a track id.

    A person keeps the same colour for as long as they keep the same id, which
    makes an id switch immediately visible.

    ``source_module`` is mixed in because id spaces are per-module: with both
    ``person_tracking`` and ``vehicle_tracking`` enabled there is a ``#1``
    person *and* a ``#1`` vehicle, and they should not look alike.
    """
    offset = sum(source_module.encode()) if source_module else 0
    return _TRACK_PALETTE[(track_id + offset) % len(_TRACK_PALETTE)]


def draw_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    caption: str = "",
    color: BGR = COLOR_OK,
    thickness: int = 2,
) -> None:
    """Draw one labelled bounding box in place."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if not caption:
        return

    scale = 0.45
    (tw, th), baseline = cv2.getTextSize(caption, _FONT, scale, 1)

    # Put the caption inside the box when it would fall off the top edge.
    top = y1 - th - baseline - 4
    if top < 0:
        top = y1 + 2
    bottom = top + th + baseline + 4

    cv2.rectangle(frame, (x1, top), (x1 + tw + 8, bottom), color, -1)
    cv2.putText(
        frame,
        caption,
        (x1 + 4, bottom - baseline - 2),
        _FONT,
        scale,
        (15, 15, 15),
        1,
        cv2.LINE_AA,
    )


def draw_detections(frame: np.ndarray, detections: Iterable[Any]) -> np.ndarray:
    """Draw ``Detection`` objects with their label and confidence.

    Typed loosely on purpose: the annotator needs ``.label``, ``.confidence``
    and ``.bbox``, not a dependency on any module.
    """
    for detection in detections:
        color = color_for_label(detection.label)
        caption = f"{detection.label} {detection.confidence:.2f}"
        draw_box(frame, detection.bbox.as_int(), caption, color)
    return frame


def draw_tracks(frame: np.ndarray, tracks: Iterable[Any]) -> np.ndarray:
    """Draw ``Track`` objects with their persistent id.

    A coasting track (not matched this frame) is drawn thinner and marked,
    so a guess is never mistaken for an observation.
    """
    for track in tracks:
        color = color_for_track_id(track.track_id, getattr(track, "source_module", ""))
        coasting = getattr(track, "time_since_update", 0) > 0

        caption = f"#{track.track_id} {track.label} {track.confidence:.2f}"
        if coasting:
            caption += " ?"

        draw_box(
            frame,
            track.bbox.as_int(),
            caption,
            color,
            thickness=1 if coasting else 2,
        )
    return frame


def draw_overlays(frame: np.ndarray, overlays: Iterable[Any]) -> np.ndarray:
    """Draw shapes modules asked for, without knowing what they mean.

    An analysis module publishes an :class:`~computer_vision.core.context.Overlay`; this
    renders it. That is how zones get drawn without the output layer having any
    concept of a zone.
    """
    for overlay in overlays:
        points = getattr(overlay, "points", None)
        if not points or len(points) < 2:
            continue

        polygon = np.array(points, dtype=np.int32)
        color = tuple(overlay.color)

        if getattr(overlay, "filled", False) or overlay.active:
            # A translucent wash, so an occupied zone reads at a glance
            # without hiding what is inside it.
            wash = frame.copy()
            cv2.fillPoly(wash, [polygon], color)
            cv2.addWeighted(wash, 0.18, frame, 0.82, 0.0, dst=frame)

        cv2.polylines(
            frame,
            [polygon],
            isClosed=True,
            color=color,
            thickness=2 if overlay.active else 1,
            lineType=cv2.LINE_AA,
        )

        if overlay.label:
            anchor = min(points, key=lambda p: (p[1], p[0]))
            _draw_tag(frame, overlay.label, anchor, color)

    return frame


def _draw_tag(frame: np.ndarray, text: str, anchor: Tuple[int, int], color: BGR) -> None:
    scale = 0.45
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = anchor
    top = max(0, y - th - baseline - 4)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + baseline + 4), color, -1)
    cv2.putText(
        frame, text, (x + 4, top + th + 2), _FONT, scale, (15, 15, 15), 1, cv2.LINE_AA
    )


def draw_banner(frame: np.ndarray, text: str, color: BGR = COLOR_ACCENT) -> np.ndarray:
    """Draw a centred message across the middle of the frame."""
    h, w = frame.shape[:2]
    scale = 0.8
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 2)
    x = max(0, (w - tw) // 2)
    y = h // 2
    draw_panel(frame, (x - 16, y - th - 16), (x + tw + 16, y + 16), alpha=0.7)
    cv2.putText(frame, text, (x, y), _FONT, scale, color, 2, cv2.LINE_AA)
    return frame
