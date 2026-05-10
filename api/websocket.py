"""
websocket.py
------------
WebSocket connection manager. Pushes FrameAnalytics to all connected clients.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import WebSocket
from src.logger import get_logger

log = get_logger(__name__)


class ConnectionManager:
    """Tracks open WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)
        log.info("WebSocket client connected", extra={"total": len(self._active)})

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._active:
            self._active.remove(ws)
        log.info("WebSocket client disconnected", extra={"total": len(self._active)})

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """Send JSON payload to all connected clients. Dead connections are removed."""
        dead: list[WebSocket] = []
        message = json.dumps(payload)
        for ws in self._active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._active)


# Singleton used by server.py and routes.py
ws_manager = ConnectionManager()
