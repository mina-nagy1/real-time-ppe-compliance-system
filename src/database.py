"""
database.py
-----------
Analytics and event persistence layer.

What this stores
----------------
violation_events   — who, what, when, where. "Worker #7 had no helmet at 14:32"
line_crossings     — directional counts per counting line
zone_occupancy     — periodic snapshots of people count per zone
alerts             — crowd density and compliance alert log

What this does NOT store
------------------------
- Frame images or raw video
- Per-frame detection arrays
- Redundant track state every frame

Useful queries this enables
---------------------------
  "How many violations happened today?"
  "Which worker violated most?"
  "Peak crowd density this hour?"
  "Which zone had most crossings?"
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from src.logger import get_logger

log = get_logger(__name__)

DB_PATH = Path("outputs/metrics/ppe_events.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Event dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ViolationEvent:
    worker_id:      int
    violation_type: str        # "no_helmet" | "no_vest" | "no_gloves"
    camera_id:      str = "cam_0"
    timestamp:      float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class LineCrossingEvent:
    line_id:   str
    direction: str             # "in" | "out"
    worker_id: int
    camera_id: str = "cam_0"
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class OccupancySnapshot:
    """Logged every N seconds — not every frame."""
    zone_id:      str
    people_count: int
    camera_id:    str = "cam_0"
    timestamp:    float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AlertEvent:
    alert_type:  str           # "crowd_density" | "compliance_violation"
    description: str
    camera_id:   str = "cam_0"
    timestamp:   float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ── Database ──────────────────────────────────────────────────────────────────

class Database:
    """
    Lightweight event store. Writes are batched to avoid per-event I/O overhead.

    Usage
    -----
    db = Database()
    db.log_violation(ViolationEvent(worker_id=7, violation_type="no_helmet"))
    db.log_occupancy(OccupancySnapshot(zone_id="main", people_count=12))

    # Analytics queries:
    db.violations_today()
    db.top_violators(limit=5)
    db.peak_occupancy(zone_id="main")
    """

    def __init__(self, path: Path = DB_PATH, batch_size: int = 20) -> None:
        self._path       = path
        self._batch_size = batch_size
        self._lock       = threading.Lock()

        self._viol_buf:     list[ViolationEvent]    = []
        self._cross_buf:    list[LineCrossingEvent] = []
        self._occ_buf:      list[OccupancySnapshot] = []
        self._alert_buf:    list[AlertEvent]        = []

        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
        log.info("Database ready", extra={"path": str(path)})

    # ── Schema ────────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        self._conn.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS violation_events (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      REAL    NOT NULL,
            worker_id      INTEGER NOT NULL,
            violation_type TEXT    NOT NULL,
            camera_id      TEXT    NOT NULL DEFAULT 'cam_0'
        );

        CREATE TABLE IF NOT EXISTS line_crossings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL    NOT NULL,
            line_id   TEXT    NOT NULL,
            direction TEXT    NOT NULL,
            worker_id INTEGER,
            camera_id TEXT    NOT NULL DEFAULT 'cam_0'
        );

        CREATE TABLE IF NOT EXISTS zone_occupancy (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    REAL    NOT NULL,
            zone_id      TEXT    NOT NULL,
            people_count INTEGER NOT NULL,
            camera_id    TEXT    NOT NULL DEFAULT 'cam_0'
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            alert_type  TEXT    NOT NULL,
            description TEXT,
            camera_id   TEXT    NOT NULL DEFAULT 'cam_0'
        );

        CREATE INDEX IF NOT EXISTS idx_viol_ts     ON violation_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_viol_worker ON violation_events(worker_id);
        CREATE INDEX IF NOT EXISTS idx_cross_line  ON line_crossings(line_id);
        CREATE INDEX IF NOT EXISTS idx_occ_zone    ON zone_occupancy(zone_id, timestamp);
        """)
        self._conn.commit()

    # ── Log methods ───────────────────────────────────────────────────────

    def log_violation(self, event: ViolationEvent) -> None:
        with self._lock:
            self._viol_buf.append(event)
            self._maybe_flush()

    def log_crossing(self, event: LineCrossingEvent) -> None:
        with self._lock:
            self._cross_buf.append(event)
            self._maybe_flush()

    def log_occupancy(self, snapshot: OccupancySnapshot) -> None:
        """Call every N seconds, not every frame."""
        with self._lock:
            self._occ_buf.append(snapshot)
            self._maybe_flush()

    def log_alert(self, event: AlertEvent) -> None:
        with self._lock:
            self._alert_buf.append(event)
            self._maybe_flush()

    # ── Flush ─────────────────────────────────────────────────────────────

    def _maybe_flush(self) -> None:
        total = len(self._viol_buf) + len(self._cross_buf) + \
                len(self._occ_buf) + len(self._alert_buf)
        if total >= self._batch_size:
            self._flush_unsafe()

    def flush(self) -> None:
        with self._lock:
            self._flush_unsafe()

    def _flush_unsafe(self) -> None:
        if not any([self._viol_buf, self._cross_buf, self._occ_buf, self._alert_buf]):
            return
        try:
            cur = self._conn.cursor()

            if self._viol_buf:
                cur.executemany(
                    "INSERT INTO violation_events (timestamp, worker_id, violation_type, camera_id) "
                    "VALUES (?, ?, ?, ?)",
                    [(e.timestamp, e.worker_id, e.violation_type, e.camera_id)
                     for e in self._viol_buf],
                )

            if self._cross_buf:
                cur.executemany(
                    "INSERT INTO line_crossings (timestamp, line_id, direction, worker_id, camera_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(e.timestamp, e.line_id, e.direction, e.worker_id, e.camera_id)
                     for e in self._cross_buf],
                )

            if self._occ_buf:
                cur.executemany(
                    "INSERT INTO zone_occupancy (timestamp, zone_id, people_count, camera_id) "
                    "VALUES (?, ?, ?, ?)",
                    [(s.timestamp, s.zone_id, s.people_count, s.camera_id)
                     for s in self._occ_buf],
                )

            if self._alert_buf:
                cur.executemany(
                    "INSERT INTO alerts (timestamp, alert_type, description, camera_id) "
                    "VALUES (?, ?, ?, ?)",
                    [(e.timestamp, e.alert_type, e.description, e.camera_id)
                     for e in self._alert_buf],
                )

            self._conn.commit()
            self._viol_buf.clear()
            self._cross_buf.clear()
            self._occ_buf.clear()
            self._alert_buf.clear()

        except sqlite3.Error as exc:
            log.error("DB flush failed", extra={"error": str(exc)})
            self._conn.rollback()

    # ── Analytics queries ─────────────────────────────────────────────────

    def violations_today(self) -> list[dict]:
        """All violations in the last 24 hours."""
        since = time.time() - 86400
        cur = self._conn.execute(
            "SELECT * FROM violation_events WHERE timestamp > ? ORDER BY timestamp DESC",
            (since,),
        )
        return [dict(r) for r in cur.fetchall()]

    def top_violators(self, limit: int = 5) -> list[dict]:
        """Workers with the most violations, all time."""
        cur = self._conn.execute(
            "SELECT worker_id, COUNT(*) as total, "
            "GROUP_CONCAT(DISTINCT violation_type) as types "
            "FROM violation_events "
            "GROUP BY worker_id ORDER BY total DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def violations_by_type(self) -> list[dict]:
        """Count of each violation type."""
        cur = self._conn.execute(
            "SELECT violation_type, COUNT(*) as count "
            "FROM violation_events GROUP BY violation_type ORDER BY count DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    def peak_occupancy(self, zone_id: str = "main") -> dict | None:
        """Highest recorded people count for a zone."""
        cur = self._conn.execute(
            "SELECT * FROM zone_occupancy WHERE zone_id = ? "
            "ORDER BY people_count DESC LIMIT 1",
            (zone_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def crossing_counts(self) -> list[dict]:
        """Total in/out per counting line."""
        cur = self._conn.execute(
            "SELECT line_id, direction, COUNT(*) as count "
            "FROM line_crossings GROUP BY line_id, direction"
        )
        return [dict(r) for r in cur.fetchall()]

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]

    def hourly_violation_summary(self) -> list[dict]:
        """Violations grouped by hour — for trend charts."""
        cur = self._conn.execute(
            "SELECT strftime('%Y-%m-%d %H:00', datetime(timestamp, 'unixepoch')) as hour, "
            "COUNT(*) as count FROM violation_events "
            "GROUP BY hour ORDER BY hour DESC LIMIT 24"
        )
        return [dict(r) for r in cur.fetchall()]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self.flush()
        self._conn.close()
        log.info("Database closed")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
