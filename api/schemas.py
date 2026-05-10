"""
schemas.py
----------
Pydantic models for all API request/response bodies and WebSocket messages.
Single source of truth shared by backend routes and frontend JS.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


# ── REST response models ──────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    running:         bool
    fps:             float
    frame_count:     int
    person_count:    int
    violation_count: int
    compliant_count: int


class ViolationRecord(BaseModel):
    timestamp:      float
    track_id:       int
    violation_type: str
    cx:             float
    cy:             float


class LineCrossingCount(BaseModel):
    line_id:   str
    direction: str
    count:     int


class FrameStatRecord(BaseModel):
    timestamp:       float
    fps:             float
    person_count:    int
    violation_count: int


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"


# ── WebSocket push message ────────────────────────────────────────────────────

class WSMessage(BaseModel):
    """
    Pushed to all connected WebSocket clients on every analytics flush.
    Frontend parses this to update Chart.js graphs in real time.
    """
    type:             str = "analytics"
    timestamp:        float
    fps:              float
    person_count:     int
    violation_count:  int
    compliant_count:  int
    crowd_alert:      bool
    line_counts:      dict[str, dict[str, int]]
    new_violations:   list[dict[str, Any]]


# ── Config override (for live config patching via API) ────────────────────────

class StreamSourceUpdate(BaseModel):
    source: Any     # int or str


class ConfidenceUpdate(BaseModel):
    conf_threshold: float


class TrackerUpdate(BaseModel):
    name: str       # "bytetrack" | "deepocsort" | "botsort"
