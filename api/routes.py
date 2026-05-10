"""
routes.py
---------
All FastAPI REST endpoints. server.py mounts this router.
No business logic here — delegates to pipeline and database.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import (
    HealthResponse,
    StatsResponse,
    ViolationRecord,
    LineCrossingCount,
    StreamSourceUpdate,
)
from src.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api")

_pipeline = None
_database = None


def inject(pipeline, database) -> None:
    global _pipeline, _database
    _pipeline = pipeline
    _database = database


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


# ── Pipeline control ──────────────────────────────────────────────────────────

@router.post("/pipeline/start")
def start_pipeline():
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")
    if _pipeline.is_running:
        return {"status": "already_running"}
    _pipeline.start()
    return {"status": "started"}


@router.post("/pipeline/stop")
def stop_pipeline():
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")
    _pipeline.stop()
    return {"status": "stopped"}


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")
    return StatsResponse(**_pipeline.stats)


# ── Data queries — match actual database.py method names ─────────────────────

@router.get("/violations")
def get_violations():
    """All violations in the last 24 hours."""
    if _database is None:
        raise HTTPException(503, "Database not initialised")
    return _database.violations_today()


@router.get("/violations/top")
def get_top_violators(limit: int = 5):
    """Workers with the most violations."""
    if _database is None:
        raise HTTPException(503, "Database not initialised")
    return _database.top_violators(limit=limit)


@router.get("/violations/by-type")
def get_violations_by_type():
    """Count per violation type."""
    if _database is None:
        raise HTTPException(503, "Database not initialised")
    return _database.violations_by_type()


@router.get("/violations/hourly")
def get_hourly_summary():
    """Violations grouped by hour — for trend charts."""
    if _database is None:
        raise HTTPException(503, "Database not initialised")
    return _database.hourly_violation_summary()


@router.get("/crossings")
def get_crossings():
    """Total in/out counts per counting line."""
    if _database is None:
        raise HTTPException(503, "Database not initialised")
    return _database.crossing_counts()


@router.get("/occupancy/peak")
def get_peak_occupancy(zone_id: str = "main"):
    """Peak people count recorded for a zone."""
    if _database is None:
        raise HTTPException(503, "Database not initialised")
    result = _database.peak_occupancy(zone_id=zone_id)
    if result is None:
        return {"zone_id": zone_id, "people_count": 0}
    return result


@router.get("/alerts")
def get_alerts(limit: int = 20):
    """Recent crowd and compliance alerts."""
    if _database is None:
        raise HTTPException(503, "Database not initialised")
    return _database.recent_alerts(limit=limit)


# ── Live config patching ──────────────────────────────────────────────────────

@router.post("/config/source")
def update_source(body: StreamSourceUpdate):
    """Hot-swap video source — pipeline restarts automatically."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")
    _pipeline.stop()
    _pipeline._source = body.source
    _pipeline.start()
    return {"status": "restarted", "source": body.source}
