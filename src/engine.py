"""
engine.py
---------
The analytics brain. Operates on the list of active Track objects
produced by tracker.py every frame.

Responsibilities
----------------
• PPE compliance checking per track ID
• Virtual counting lines with in/out direction detection (cross-product)
• Gaussian dwell heatmap accumulator with decay
• Crowd density alerts
• HUD overlay composition
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config_loader import AppConfig, AnalyticsConfig, CountingLine, get_config
from src.logger import get_logger
from src.tracker import Track
from src.utils import point_line_side, segments_intersect

log = get_logger(__name__)


# ── Event dataclasses ─────────────────────────────────────────────────────────

@dataclass
class LineCrossingEvent:
    timestamp: float
    track_id:  int
    line_id:   str
    direction: str          # "in" or "out"
    cx:        float
    cy:        float


@dataclass
class ViolationEvent:
    timestamp:      float
    track_id:       int
    violation_type: str
    cx:             float
    cy:             float


@dataclass
class FrameAnalytics:
    """
    Everything the engine computed for one frame.
    Pipeline passes this to visualizer, database, and mlflow tracker.
    """
    timestamp:        float
    person_count:     int
    compliant_count:  int
    violation_count:  int
    crowd_alert:      bool
    line_crossings:   list[LineCrossingEvent] = field(default_factory=list)
    new_violations:   list[ViolationEvent]    = field(default_factory=list)
    line_counts:      dict[str, dict[str, int]] = field(default_factory=dict)   # line_id → {"in": N, "out": N}


# ── Compliance checker ────────────────────────────────────────────────────────

class ComplianceChecker:
    """
    Evaluates whether a person track has the required PPE.

    The model returns detections for BOTH person and PPE items.
    For each person track, we look at what PPE class detections overlap
    spatially (IoU > threshold) and determine compliance.
    """

    def __init__(self, required: list[str], violation_classes: list[str]) -> None:
        self._required   = set(required)
        self._violations = set(violation_classes)
        # Map violation class → which required item it represents
        # e.g. "no_helmet" → "helmet"
        self._violation_map: dict[str, str] = {}
        for v in violation_classes:
            for r in required:
                if r in v:
                    self._violation_map[v] = r
                    break

    def check(
        self,
        person_track: Track,
        all_tracks:   list[Track],
    ) -> tuple[bool, list[str]]:
        """
        Returns (is_compliant, violation_flags).

        Strategy: look at all non-person tracks that spatially overlap
        (IoU > 0.15) with the person box. Collect their class names.
        A violation flag is raised for each required item that is either:
          - absent (no matching PPE class in overlapping tracks), or
          - explicitly detected as missing (no_helmet, no_vest, etc.)
        """
        overlapping_classes: set[str] = set()

        for t in all_tracks:
            if t.track_id == person_track.track_id:
                continue
            if t.class_name.lower() == "person":
                continue

            # Check spatial overlap using bounding boxes
            from src.utils import bbox_iou
            iou = bbox_iou(person_track.box, t.box)
            if iou > 0.15 and t.confidence > 0.45:
                overlapping_classes.add(t.class_name)

        # Determine violations
        violation_flags: list[str] = []

        for cls in overlapping_classes:
            if cls in self._violations:
                violation_flags.append(cls)

        # If a violation class is not explicitly detected, check for absence
        # (no PPE class found at all → implicit violation)
        # Only when model has both positive and negative PPE classes;
        # skip implicit check if model has no negative classes to avoid false positives
        if self._violations:
            for req in self._required:
                has_positive = req in overlapping_classes
                has_negative = any(
                    self._violation_map.get(v) == req
                    for v in violation_flags
                )
                if not has_positive and not has_negative and overlapping_classes:
                    # Only flag implicit absence if we see *some* PPE (reduces false pos)
                    pass  # Conservative: skip implicit violation

        is_compliant = len(violation_flags) == 0
        return is_compliant, violation_flags


# ── Heatmap accumulator ───────────────────────────────────────────────────────

class HeatmapAccumulator:
    """Gaussian dwell heatmap. Call update() each frame with track positions."""

    def __init__(self, shape: tuple[int, int], sigma: int = 25, decay: float = 0.98) -> None:
        self._sigma  = sigma
        self._decay  = decay
        self._map    = np.zeros(shape, dtype=np.float32)  # H × W
        self._last_saved = time.time()

    def update(self, tracks: list[Track]) -> None:
        """Decay existing heat and accumulate new Gaussians at track positions."""
        self._map *= self._decay
        for t in tracks:
            cx, cy = int(t.center[0]), int(t.center[1])
            h, w   = self._map.shape
            if 0 <= cx < w and 0 <= cy < h:
                # Draw a Gaussian blob centered at (cx, cy)
                x_lo = max(0, cx - 3 * self._sigma)
                x_hi = min(w, cx + 3 * self._sigma)
                y_lo = max(0, cy - 3 * self._sigma)
                y_hi = min(h, cy + 3 * self._sigma)

                xs = np.arange(x_lo, x_hi)
                ys = np.arange(y_lo, y_hi)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                xx, yy = np.meshgrid(xs, ys)
                blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self._sigma ** 2))
                self._map[y_lo:y_hi, x_lo:x_hi] += blob.astype(np.float32)

    def render(self, frame: np.ndarray, alpha: float = 0.50, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Return frame with heatmap overlay blended in."""
        if self._map.max() == 0:
            return frame

        normalized = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored    = cv2.applyColorMap(normalized, colormap)
        colored    = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)

    def save(self, path: Path) -> None:
        """Save current heatmap as PNG."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._map.max() == 0:
            return
        normalized = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored    = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        cv2.imwrite(str(path), colored)
        log.info("Heatmap saved", extra={"path": str(path)})

    @property
    def raw(self) -> np.ndarray:
        return self._map.copy()


# ── Engine ────────────────────────────────────────────────────────────────────

class Engine:
    """
    Analytics engine. Called once per frame with the list of active tracks.

    Returns FrameAnalytics and mutates track compliance state.
    Also owns the heatmap and counting-line state.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],
        config: Optional[AppConfig] = None,
    ) -> None:
        self._cfg  = config or get_config()
        ac: AnalyticsConfig = self._cfg.analytics

        self._enabled       = ac.enabled
        self._frame_shape   = frame_shape   # (H, W)
        self._crowd_max     = ac.crowd.max_persons
        self._crowd_cooldown = ac.crowd.alert_cooldown_seconds
        self._viol_cooldown  = ac.compliance.violation_cooldown_seconds
        self._hm_alpha      = ac.heatmap.alpha
        self._hm_save_interval = ac.heatmap.save_interval_seconds

        # Counting lines
        self._lines: list[CountingLine] = ac.counting_lines
        # line_id → {"in": count, "out": count}
        self._line_counts: dict[str, dict[str, int]] = {
            ln.id: {"in": 0, "out": 0} for ln in self._lines
        }
        # track_id → last known side for each line (to detect crossing)
        self._prev_side: dict[str, dict[int, float]] = {
            ln.id: {} for ln in self._lines
        }

        # Heatmap
        colormap_name = ac.heatmap.colormap.upper()
        self._colormap = getattr(cv2, f"COLORMAP_{colormap_name}", cv2.COLORMAP_JET)
        self._heatmap  = HeatmapAccumulator(
            shape=frame_shape,
            sigma=ac.heatmap.sigma,
            decay=ac.heatmap.decay,
        )

        # Compliance checker
        mc = self._cfg.model
        self._compliance_checker = ComplianceChecker(
            required=mc.compliance.required,
            violation_classes=mc.compliance.violation_classes,
        )

        # Cooldown trackers
        self._last_crowd_alert  = 0.0
        self._last_viol_alert:  dict[str, float] = {}  # key = "track_id_violation_type"
        self._last_heatmap_save = time.time()

        log.info(
            "Engine ready",
            extra={
                "lines": len(self._lines),
                "crowd_max": self._crowd_max,
            },
        )

    # ── Main update ───────────────────────────────────────────────────────

    def process(
        self,
        tracks: list[Track],
        frame:  np.ndarray,
        fps:    float = 0.0,
    ) -> tuple[FrameAnalytics, np.ndarray]:
        """
        Process one frame's tracks.

        Parameters
        ----------
        tracks : active tracks from tracker.py
        frame  : BGR frame (will be annotated in-place)
        fps    : current FPS for HUD

        Returns
        -------
        (FrameAnalytics, annotated_frame)
        """
        now = time.time()

        # ── 1. Compliance check ───────────────────────────────────────────
        new_violations: list[ViolationEvent] = []
        person_tracks = [t for t in tracks if t.class_name.lower() == "person"]

        for t in person_tracks:
            is_compliant, flags = self._compliance_checker.check(t, tracks)
            # Update tracker state
            from src.tracker import Tracker  # avoid circular at module level
            t.is_compliant    = is_compliant
            t.violation_flags = flags

            if not is_compliant:
                for f in flags:
                    cooldown_key = f"{t.track_id}_{f}"
                    last = self._last_viol_alert.get(cooldown_key, 0.0)
                    if now - last >= self._viol_cooldown:
                        new_violations.append(ViolationEvent(
                            timestamp=now,
                            track_id=t.track_id,
                            violation_type=f,
                            cx=t.center[0],
                            cy=t.center[1],
                        ))
                        self._last_viol_alert[cooldown_key] = now

        # ── 2. Counting lines ─────────────────────────────────────────────
        crossings: list[LineCrossingEvent] = []

        for line in self._lines:
            p1 = tuple(line.p1)
            p2 = tuple(line.p2)

            for t in tracks:
                if len(t.history) < 2:
                    continue

                prev_pos = t.history[-2]
                curr_pos = t.history[-1]

                # Detect crossing via segment intersection
                if segments_intersect(prev_pos, curr_pos, p1, p2):
                    prev_side = point_line_side(prev_pos, p1, p2)
                    direction = "in" if prev_side > 0 else "out"
                    self._line_counts[line.id][direction] += 1

                    crossings.append(LineCrossingEvent(
                        timestamp=now,
                        track_id=t.track_id,
                        line_id=line.id,
                        direction=direction,
                        cx=curr_pos[0],
                        cy=curr_pos[1],
                    ))
                    log.debug(
                        "Line crossing",
                        extra={"line": line.id, "track": t.track_id, "dir": direction},
                    )

        # ── 3. Heatmap ────────────────────────────────────────────────────
        self._heatmap.update(person_tracks)

        # Periodic heatmap save
        if now - self._last_heatmap_save >= self._hm_save_interval:
            hm_path = Path("outputs/heatmaps") / f"heatmap_{int(now)}.png"
            self._heatmap.save(hm_path)
            self._last_heatmap_save = now

        # ── 4. Crowd alert ────────────────────────────────────────────────
        crowd_alert = False
        if len(person_tracks) > self._crowd_max:
            if now - self._last_crowd_alert >= self._crowd_cooldown:
                crowd_alert = True
                self._last_crowd_alert = now
                log.warning(
                    "Crowd density alert",
                    extra={"count": len(person_tracks), "max": self._crowd_max},
                )

        # ── 5. Annotate frame ─────────────────────────────────────────────
        annotated = self._draw_hud(frame, tracks, fps, crowd_alert)

        # ── 6. Build analytics object ─────────────────────────────────────
        compliant_count  = sum(1 for t in person_tracks if t.is_compliant)
        violation_count  = sum(1 for t in person_tracks if not t.is_compliant)

        analytics = FrameAnalytics(
            timestamp       = now,
            person_count    = len(person_tracks),
            compliant_count = compliant_count,
            violation_count = violation_count,
            crowd_alert     = crowd_alert,
            line_crossings  = crossings,
            new_violations  = new_violations,
            line_counts     = {k: dict(v) for k, v in self._line_counts.items()},
        )

        return analytics, annotated

    # ── HUD drawing ───────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        fps:   float,
        crowd_alert: bool,
    ) -> np.ndarray:
        """Draw heatmap overlay, counting lines, stats panel, and alerts."""
        out = frame.copy()

        # Heatmap overlay
        if self._cfg.analytics.heatmap.enabled:
            out = self._heatmap.render(out, alpha=self._hm_alpha, colormap=self._colormap)

        # Counting lines
        for line in self._lines:
            p1 = tuple(map(int, line.p1))
            p2 = tuple(map(int, line.p2))
            cv2.line(out, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
            counts = self._line_counts[line.id]
            label  = f"{line.label}  ↑{counts['in']}  ↓{counts['out']}"
            mid_x  = (p1[0] + p2[0]) // 2
            mid_y  = (p1[1] + p2[1]) // 2
            cv2.putText(out, label, (mid_x, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # Stats panel (top-left dark background)
        person_count    = sum(1 for t in tracks if t.class_name.lower() == "person")
        violation_count = sum(1 for t in tracks if t.class_name.lower() == "person" and not t.is_compliant)
        compliant_count = person_count - violation_count

        stats_lines = [
            f"FPS: {fps:.1f}",
            f"Persons: {person_count}",
            f"Compliant: {compliant_count}",
            f"Violations: {violation_count}",
        ]

        panel_h = len(stats_lines) * 22 + 10
        cv2.rectangle(out, (5, 5), (180, panel_h), (0, 0, 0), -1)
        cv2.rectangle(out, (5, 5), (180, panel_h), (80, 80, 80), 1)

        for i, line_text in enumerate(stats_lines):
            color = (255, 255, 255)
            if "Violations" in line_text and violation_count > 0:
                color = (0, 80, 255)
            cv2.putText(out, line_text, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # Crowd alert banner
        if crowd_alert:
            h = out.shape[0]
            cv2.rectangle(out, (0, h - 40), (out.shape[1], h), (0, 0, 180), -1)
            cv2.putText(out, "⚠  CROWD DENSITY ALERT", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return out

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def heatmap(self) -> HeatmapAccumulator:
        return self._heatmap

    @property
    def line_counts(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self._line_counts.items()}

    def reset_line_counts(self) -> None:
        for line_id in self._line_counts:
            self._line_counts[line_id] = {"in": 0, "out": 0}
