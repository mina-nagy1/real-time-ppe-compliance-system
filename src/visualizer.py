"""
visualizer.py
-------------
All cv2 drawing lives here. Tracker and engine return data — this module
turns data into pixels. Nothing else should call cv2.rectangle / cv2.putText.

Draws per frame:
  • Segmentation mask overlay (coloured per track ID)
  • Bounding boxes with compliance colour coding
  • Motion trails
  • Compliance badge + violation flags label
  • Track ID chip
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.config_loader import AppConfig, get_config
from src.tracker import Track
from src.utils import track_color

# Compliance colours
COLOR_COMPLIANT  = (50, 205, 50)    # green
COLOR_VIOLATION  = (30,  30, 220)   # red (BGR)
COLOR_UNKNOWN    = (180, 180, 180)  # grey


class Visualizer:
    """
    Stateless renderer — all draw_* methods take frame + data and return
    an annotated copy (or modify in-place if inplace=True).
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._cfg = config or get_config()
        vc = self._cfg.tracker.visualization

        self._trail_length  = vc.trail_length
        self._trail_thick   = vc.trail_thickness
        self._mask_alpha    = vc.mask_alpha
        self._bbox_thick    = vc.bbox_thickness
        self._font_scale    = vc.font_scale

    # ── Main entry point ──────────────────────────────────────────────────

    def render(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Render all tracks onto frame.
        Applies masks → trails → boxes → labels in that order.
        """
        out = frame.copy()
        out = self._draw_masks(out, tracks)
        out = self._draw_trails(out, tracks)
        out = self._draw_boxes_and_labels(out, tracks)
        return out

    # ── Mask overlay ──────────────────────────────────────────────────────

    def _draw_masks(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        overlay = frame.copy()
        for t in tracks:
            if t.mask is None:
                continue
            color = track_color(t.track_id) if t.is_compliant else COLOR_VIOLATION
            overlay[t.mask] = color
        return cv2.addWeighted(overlay, self._mask_alpha, frame, 1.0 - self._mask_alpha, 0)

    # ── Motion trails ─────────────────────────────────────────────────────

    def _draw_trails(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        for t in tracks:
            pts = list(t.history)
            if len(pts) < 2:
                continue
            color = track_color(t.track_id)
            for i in range(1, len(pts)):
                # Fade alpha by position in history (older = more transparent)
                alpha = i / len(pts)
                thickness = max(1, int(self._trail_thick * alpha))
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]),     int(pts[i][1]))
                cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
        return frame

    # ── Bounding boxes and labels ─────────────────────────────────────────

    def _draw_boxes_and_labels(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)

            # Box colour: green = compliant, red = violation
            if t.class_name == "person":
                box_color = COLOR_COMPLIANT if t.is_compliant else COLOR_VIOLATION
            else:
                box_color = track_color(t.track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, self._bbox_thick, cv2.LINE_AA)

            # ── Label chip ────────────────────────────────────────────────
            badge = "OK" if t.is_compliant else "!!"
            label = f"#{t.track_id} {t.class_name} {badge}"
            if t.violation_flags:
                label += f"  [{', '.join(t.violation_flags)}]"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, 1)

            chip_y1 = max(0, y1 - th - 8)
            chip_y2 = y1
            cv2.rectangle(frame, (x1, chip_y1), (x1 + tw + 6, chip_y2), box_color, -1)
            cv2.putText(
                frame, label,
                (x1 + 3, chip_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Confidence score (small, bottom-right of box)
            conf_label = f"{t.confidence:.2f}"
            cv2.putText(
                frame, conf_label,
                (x2 - 35, y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                box_color,
                1,
                cv2.LINE_AA,
            )

        return frame
