"""
pipeline.py
-----------
Orchestrator. Runs the full detect → track → analyse → visualise chain
in a background thread. Exposes:

  • latest_frame_jpeg : bytes          — for MJPEG stream
  • latest_analytics  : FrameAnalytics — for WebSocket push
  • stats             : dict           — current live stats snapshot
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from src.config_loader import AppConfig, get_config
from src.database import Database, ViolationEvent, LineCrossingEvent, OccupancySnapshot, AlertEvent
from src.detector import Detector
from src.engine import Engine, FrameAnalytics
from src.logger import get_logger, log_exception
from src.tracker import Tracker
from src.utils import FPSCounter, frame_to_jpeg, resize_frame
from src.visualizer import Visualizer

log = get_logger(__name__)

# How often to snapshot occupancy to DB (seconds) — not every frame
OCCUPANCY_INTERVAL = 5.0


class Pipeline:
    """
    Main processing pipeline.

    Usage
    -----
    pipeline = Pipeline()
    pipeline.start()
    jpeg = pipeline.latest_frame_jpeg   # in MJPEG route
    pipeline.stop()                     # on shutdown
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._cfg = config or get_config()
        sc = self._cfg.stream

        self._source      = sc.source
        self._width       = sc.width
        self._height      = sc.height
        self._jpeg_quality= sc.jpeg_quality
        self._save_video  = sc.save_video
        self._output_path = Path(sc.output_path)

        # Components (lazy-init in _run)
        self._detector:   Optional[Detector]  = None
        self._tracker:    Optional[Tracker]   = None
        self._engine:     Optional[Engine]    = None
        self._visualizer: Optional[Visualizer]= None
        self._db:         Optional[Database]  = None

        # Shared state — written by bg thread, read by API thread
        self.latest_frame_jpeg: bytes = b""
        self.latest_analytics:  Optional[FrameAnalytics] = None
        self.stats: dict = {
            "running":         False,
            "fps":             0.0,
            "frame_count":     0,
            "person_count":    0,
            "violation_count": 0,
            "compliant_count": 0,
        }

        self._fps_counter   = FPSCounter(window=30)
        self._stop_event    = threading.Event()
        self._thread:       Optional[threading.Thread]   = None
        self._video_writer: Optional[cv2.VideoWriter]    = None
        self._last_occupancy_snap                        = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            log.warning("Pipeline already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="pipeline", daemon=True
        )
        self._thread.start()
        log.info("Pipeline started", extra={"source": self._source})

    def stop(self) -> None:
        log.info("Stopping pipeline…")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.stats["running"] = False
        log.info("Pipeline stopped")

    # ── Main loop ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._init_components()

            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                log.error("Cannot open video source", extra={"source": self._source})
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   self._cfg.stream.buffer_size)

            self.stats["running"] = True
            frame_id = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()

                if not ret:
                    if isinstance(self._source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    log.warning("Frame capture failed")
                    time.sleep(0.01)
                    continue

                frame = resize_frame(frame, self._width, self._height, keep_aspect=False)
                frame_id += 1
                self._fps_counter.tick()
                fps = self._fps_counter.fps

                # ── Detect → Track → Analyse → Visualise ──────────────────
                detections          = self._detector.detect(frame)
                tracks              = self._tracker.update(detections, frame)
                analytics, annotated = self._engine.process(tracks, frame, fps)
                annotated           = self._visualizer.render(annotated, tracks)

                # ── Persist events (not every frame — only when something happens) ──
                now = analytics.timestamp

                for v in analytics.new_violations:
                    self._db.log_violation(ViolationEvent(
                        worker_id      = v.track_id,
                        violation_type = v.violation_type,
                        timestamp      = v.timestamp,
                    ))

                for c in analytics.line_crossings:
                    self._db.log_crossing(LineCrossingEvent(
                        line_id   = c.line_id,
                        direction = c.direction,
                        worker_id = c.track_id,
                        timestamp = c.timestamp,
                    ))

                if analytics.crowd_alert:
                    self._db.log_alert(AlertEvent(
                        alert_type  = "crowd_density",
                        description = f"{analytics.person_count} persons in frame",
                        timestamp   = now,
                    ))

                # Occupancy snapshot every N seconds
                if now - self._last_occupancy_snap >= OCCUPANCY_INTERVAL:
                    self._db.log_occupancy(OccupancySnapshot(
                        zone_id      = "main",
                        people_count = analytics.person_count,
                        timestamp    = now,
                    ))
                    self._last_occupancy_snap = now

                # ── Stream ────────────────────────────────────────────────
                self.latest_frame_jpeg = frame_to_jpeg(annotated, self._jpeg_quality)
                self.latest_analytics  = analytics

                self.stats.update({
                    "fps":             round(fps, 1),
                    "frame_count":     frame_id,
                    "person_count":    analytics.person_count,
                    "violation_count": analytics.violation_count,
                    "compliant_count": analytics.compliant_count,
                })

                if self._video_writer:
                    self._video_writer.write(annotated)

        except Exception:
            log_exception(log, "Pipeline crashed")
        finally:
            self._shutdown(cap if "cap" in dir() else None)

    # ── Init / shutdown ───────────────────────────────────────────────────

    def _init_components(self) -> None:
        log.info("Initialising pipeline components…")
        self._detector   = Detector(self._cfg)
        self._detector.warmup()
        self._tracker    = Tracker(self._cfg)
        self._engine     = Engine(
            frame_shape=(self._height, self._width),
            config=self._cfg,
        )
        self._visualizer = Visualizer(self._cfg)
        self._db         = Database()

        if self._save_video:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self._cfg.stream.fourcc)
            self._video_writer = cv2.VideoWriter(
                str(self._output_path), fourcc,
                self._cfg.stream.fps,
                (self._width, self._height),
            )
        log.info("All components ready")

    def _shutdown(self, cap) -> None:
        log.info("Pipeline shutdown…")
        if cap:
            cap.release()
        if self._video_writer:
            self._video_writer.release()
        if self._engine:
            self._engine.heatmap.save(Path("outputs/heatmaps/final_heatmap.png"))
        if self._db:
            self._db.close()
        log.info("Pipeline shutdown complete")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
