"""
utils.py
--------
Pure utility functions — no side effects, no imports from other src modules.
IoU math, cross-product, color palette, frame helpers.
"""

from __future__ import annotations

import colorsys
import time
from typing import Generator

import cv2
import numpy as np


# ── Color palette ─────────────────────────────────────────────────────────────

def generate_color_palette(n: int = 256) -> list[tuple[int, int, int]]:
    """
    Generate N visually distinct BGR colors using HSV spacing.
    Track ID → color: palette[track_id % len(palette)]
    """
    palette: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        palette.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return palette


# Pre-built palette used by tracker and visualizer
COLOR_PALETTE = generate_color_palette(256)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR color for a given track ID."""
    return COLOR_PALETTE[int(track_id) % len(COLOR_PALETTE)]


# ── IoU ───────────────────────────────────────────────────────────────────────

def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.

    Parameters
    ----------
    box1, box2 : array-like of shape (4,)

    Returns
    -------
    float in [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two boolean masks of the same shape.

    Parameters
    ----------
    mask1, mask2 : np.ndarray of dtype bool, same H×W shape
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


# ── Cross product / line math ─────────────────────────────────────────────────

def cross_product_2d(
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """2-D cross product of vectors A and B. Sign encodes left/right side."""
    return ax * by - ay * bx


def point_line_side(
    point: tuple[float, float],
    line_p1: tuple[float, float],
    line_p2: tuple[float, float],
) -> float:
    """
    Returns the signed cross-product of (line_p2 - line_p1) × (point - line_p1).
    Positive → point is on the left of the directed line p1→p2.
    Negative → point is on the right.
    Zero     → point is on the line.
    """
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    px = point[0] - line_p1[0]
    py = point[1] - line_p1[1]
    return cross_product_2d(dx, dy, px, py)


def segments_intersect(
    p1: tuple[float, float], p2: tuple[float, float],
    p3: tuple[float, float], p4: tuple[float, float],
) -> bool:
    """
    Return True if line segment p1→p2 intersects segment p3→p4.
    Used to detect when a track's motion trail crosses a counting line.
    """
    d1 = point_line_side(p3, p1, p2)
    d2 = point_line_side(p4, p1, p2)
    d3 = point_line_side(p1, p3, p4)
    d4 = point_line_side(p2, p3, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


# ── Frame helpers ─────────────────────────────────────────────────────────────

def resize_frame(
    frame: np.ndarray,
    width: int,
    height: int,
    keep_aspect: bool = True,
) -> np.ndarray:
    """Resize a frame. If keep_aspect, letterbox to fit within (width, height)."""
    if not keep_aspect:
        return cv2.resize(frame, (width, height))

    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x_off = (width - new_w) // 2
    y_off = (height - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes for MJPEG streaming."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    return bytes(buffer)


def bbox_center(box: np.ndarray | list) -> tuple[float, float]:
    """Return (cx, cy) center of a [x1, y1, x2, y2] bounding box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_bottom_center(box: np.ndarray | list) -> tuple[float, float]:
    """Return the bottom-center point — used as foot position for counting lines."""
    return ((box[0] + box[2]) / 2.0, float(box[3]))


# ── FPS counter ───────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-window FPS counter. Call tick() each frame, read fps property."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._times: list[float] = []

    def tick(self) -> None:
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ── Misc ──────────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def batched(iterable, n: int) -> Generator:
    """Yield successive n-sized chunks from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch
