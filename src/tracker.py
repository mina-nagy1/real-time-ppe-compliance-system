"""
tracker.py
----------
Wraps BoxMOT with three swappable trackers: ByteTrack, DeepOCSORT, BoTSORT.

Responsibilities
----------------
• Assign persistent track IDs across frames
• Maintain full position history per track ID (motion trails)
• Re-associate segmentation masks to tracks by IoU
• Expose per-track compliance state (set by engine.py)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.config_loader import AppConfig, get_config
from src.detector import DetectionResult
from src.logger import get_logger, log_exception
from src.utils import bbox_iou, bbox_bottom_center, track_color

log = get_logger(__name__)


# ── Track state ───────────────────────────────────────────────────────────────

@dataclass
class Track:
    """Live state for a single tracked object."""
    track_id:       int
    class_name:     str
    box:            np.ndarray          # [x1, y1, x2, y2]
    confidence:     float
    mask:           Optional[np.ndarray] = None   # H×W bool or None
    is_compliant:   bool = True
    violation_flags: list[str] = field(default_factory=list)
    last_seen:      float = field(default_factory=time.time)

    # Rolling history of bottom-center positions (cx, cy)
    history: deque = field(default_factory=lambda: deque(maxlen=40))

    @property
    def color(self) -> tuple[int, int, int]:
        return track_color(self.track_id)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return bbox_bottom_center(self.box)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0,
                (self.box[1] + self.box[3]) / 2.0)


# ── Tracker ───────────────────────────────────────────────────────────────────

class Tracker:
    """
    Multi-object tracker backed by BoxMOT.

    Parameters
    ----------
    config : AppConfig

    Usage
    -----
    tracker = Tracker()
    tracks  = tracker.update(detection_result, frame)
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._cfg = config or get_config()
        tc = self._cfg.tracker
        vc = tc.visualization

        self._tracker_name = tc.name
        self._trail_length  = vc.trail_length

        # Active tracks keyed by track_id
        self._tracks: dict[int, Track] = {}

        self._boxmot = self._build_tracker()
        log.info("Tracker ready", extra={"backend": self._tracker_name})

    # ── Build BoxMOT tracker ──────────────────────────────────────────────

    def _build_tracker(self):
        try:
            from boxmot import BYTETracker, DeepOCSORT, BoTSORT  # type: ignore
        except ImportError:
            raise ImportError(
                "boxmot is not installed. Run: pip install boxmot"
            )

        tc = self._cfg.tracker
        name = tc.name.lower()

        if name == "bytetrack":
            bt = tc.bytetrack
            return BYTETracker(
                track_thresh  = bt.track_high_thresh,
                match_thresh  = bt.match_thresh,
                track_buffer  = bt.track_buffer,
                frame_rate    = bt.frame_rate,
            )

        if name == "deepocsort":
            dc = tc.deepocsort
            return DeepOCSORT(
                model_weights = Path(tc.reid_weights),
                device        = tc.device,
                half          = tc.half,
                det_thresh    = dc.det_thresh,
                max_age       = dc.max_age,
                min_hits      = dc.min_hits,
                iou_threshold = dc.iou_threshold,
                delta_t       = dc.delta_t,
                asso_func     = dc.asso_func,
                inertia       = dc.inertia,
                use_byte      = dc.use_byte,
            )

        if name == "botsort":
            bs = tc.botsort
            return BoTSORT(
                model_weights     = Path(tc.reid_weights),
                device            = tc.device,
                half              = tc.half,
                track_high_thresh = bs.track_high_thresh,
                track_low_thresh  = bs.track_low_thresh,
                new_track_thresh  = bs.new_track_thresh,
                track_buffer      = bs.track_buffer,
                match_thresh      = bs.match_thresh,
                proximity_thresh  = bs.proximity_thresh,
                appearance_thresh = bs.appearance_thresh,
                frame_rate        = bs.frame_rate,
            )

            raise ValueError(f"Unknown tracker: {name}")

    # ── Update ────────────────────────────────────────────────────────────

    def update(
        self,
        detections: DetectionResult,
        frame: np.ndarray,
    ) -> list[Track]:
        """
        Feed one frame's detections into the tracker.

        Parameters
        ----------
        detections : DetectionResult from detector.py
        frame      : original BGR frame (needed by ReID trackers)

        Returns
        -------
        list[Track]  — all currently active tracks
        """
        if detections.n == 0:
            # Still need to tick the tracker with empty input
            try:
                self._boxmot.update(
                    np.empty((0, 6), dtype=np.float32), frame
                )
            except Exception:
                pass
            self._age_out_tracks()
            return list(self._tracks.values())

        # BoxMOT expects [x1, y1, x2, y2, conf, cls]
        dets_np = np.column_stack([
            detections.boxes,
            detections.confidences,
            detections.class_ids.astype(np.float32),
        ]).astype(np.float32)

        try:
            raw_tracks = self._boxmot.update(dets_np, frame)
        except Exception:
            log_exception(log, "BoxMOT update failed")
            return list(self._tracks.values())

        # raw_tracks rows: [x1, y1, x2, y2, track_id, conf, cls, ...]
        if raw_tracks is None or len(raw_tracks) == 0:
            self._age_out_tracks()
            return list(self._tracks.values())

        active_ids: set[int] = set()

        for row in raw_tracks:
            box        = row[:4].astype(np.float32)
            track_id   = int(row[4])
            confidence = float(row[5])
            class_id   = int(row[6])
            class_name = detections.class_names[
                self._match_detection_idx(box, detections.boxes)
            ] if detections.n > 0 else str(class_id)

            # Re-associate segmentation mask by IoU
            mask = self._associate_mask(box, detections)

            if track_id in self._tracks:
                t = self._tracks[track_id]
                t.box        = box
                t.confidence = confidence
                t.class_name = class_name
                t.mask       = mask
                t.last_seen  = time.time()
            else:
                t = Track(
                    track_id=track_id,
                    class_name=class_name,
                    box=box,
                    confidence=confidence,
                    mask=mask,
                    history=deque(maxlen=self._trail_length),
                )
                self._tracks[track_id] = t
                log.debug("New track", extra={"id": track_id, "class": class_name})

            # Update history with bottom-center position
            t.history.append(t.bottom_center)
            active_ids.add(track_id)

        self._age_out_tracks(active_ids)
        return list(self._tracks.values())

    # ── Helpers ───────────────────────────────────────────────────────────

    def _match_detection_idx(
        self,
        box: np.ndarray,
        det_boxes: np.ndarray,
    ) -> int:
        """Return index of detection box with highest IoU to a track box."""
        if len(det_boxes) == 0:
            return 0
        ious = np.array([bbox_iou(box, db) for db in det_boxes])
        return int(np.argmax(ious))

    def _associate_mask(
        self,
        track_box: np.ndarray,
        detections: DetectionResult,
    ) -> Optional[np.ndarray]:
        """
        Find the detection mask whose box IoU with track_box is highest.
        Returns None if no masks are available or IoU < 0.30.
        """
        if detections.masks is None or detections.n == 0:
            return None

        best_iou   = 0.30   # minimum threshold
        best_mask  = None

        for i, det_box in enumerate(detections.boxes):
            iou = bbox_iou(track_box, det_box)
            if iou > best_iou:
                best_iou  = iou
                best_mask = detections.masks[i]

        return best_mask

    def _age_out_tracks(self, active_ids: Optional[set[int]] = None) -> None:
        """Remove tracks that were not seen in the last update."""
        if active_ids is None:
            active_ids = set()
        stale = [tid for tid in self._tracks if tid not in active_ids]
        for tid in stale:
            del self._tracks[tid]

    # ── Compliance setter (called by engine.py) ───────────────────────────

    def set_compliance(
        self,
        track_id: int,
        is_compliant: bool,
        violation_flags: list[str],
    ) -> None:
        """Called by engine.py to record per-track compliance status."""
        if track_id in self._tracks:
            self._tracks[track_id].is_compliant    = is_compliant
            self._tracks[track_id].violation_flags = violation_flags

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def active_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all track state (e.g. when switching video source)."""
        self._tracks.clear()
        self._boxmot = self._build_tracker()
        log.info("Tracker reset")
