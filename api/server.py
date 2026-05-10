"""
server.py
---------
FastAPI application entry point.

Endpoints
---------
GET  /                    — dark-mode HTML dashboard
GET  /stream              — MJPEG live video stream
WS   /ws                  — WebSocket analytics push
GET  /api/...             — REST API (see routes.py)

Run with:
    python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from api.routes import router, inject
from api.websocket import ws_manager
from api.schemas import WSMessage
from src.config_loader import get_config
from src.database import Database
from src.logger import get_logger
from src.pipeline import Pipeline

log = get_logger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PPE Compliance System",
    description="Real-time PPE detection, tracking and segmentation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared instances ──────────────────────────────────────────────────────────

cfg      = get_config()
pipeline = Pipeline(cfg)
database = Database()

inject(pipeline, database)
app.include_router(router)

# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    pipeline.start()
    log.info("Server started — pipeline running")


@app.on_event("shutdown")
async def shutdown():
    pipeline.stop()
    database.close()
    log.info("Server shutdown complete")


# ── MJPEG stream ──────────────────────────────────────────────────────────────

async def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpeg = pipeline.latest_frame_jpeg
        if jpeg:
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg + b"\r\n"
            )
        await asyncio.sleep(1 / cfg.stream.fps)


@app.get("/stream")
async def video_stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            analytics = pipeline.latest_analytics
            if analytics:
                msg = WSMessage(
                    timestamp       = analytics.timestamp,
                    fps             = pipeline.stats.get("fps", 0.0),
                    person_count    = analytics.person_count,
                    violation_count = analytics.violation_count,
                    compliant_count = analytics.compliant_count,
                    crowd_alert     = analytics.crowd_alert,
                    line_counts     = analytics.line_counts,
                    new_violations  = [
                        {"track_id": v.track_id, "type": v.violation_type}
                        for v in analytics.new_violations
                    ],
                )
                await ws_manager.broadcast(msg.model_dump())
            await asyncio.sleep(0.25)   # push at ~4 Hz, not 30 Hz
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# ── Dashboard ─────────────────────────────────────────────────────────────────

DASHBOARD_PATH = Path("dashboard/index.html")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    if DASHBOARD_PATH.exists():
        return HTMLResponse(content=DASHBOARD_PATH.read_text())
    return HTMLResponse(content="<h1>Dashboard not found</h1><p>Run from project root.</p>")
