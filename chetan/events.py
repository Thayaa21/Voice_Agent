"""
Real-time Event Broadcaster
============================
WebSocket event stream for the live dashboard.
Uses FastAPI WebSocket connections.
"""

import json
import logging
from datetime import datetime
from typing import Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Connected dashboard WebSocket clients
_clients: set[WebSocket] = set()


async def register(websocket: WebSocket):
    """Register a new dashboard client and keep it alive."""
    _clients.add(websocket)
    logger.info("Dashboard client connected. Total: %d", len(_clients))
    try:
        # Keep alive until disconnected
        while True:
            try:
                await websocket.receive_text()
            except Exception:
                break
    finally:
        _clients.discard(websocket)
        logger.info("Dashboard client disconnected. Total: %d", len(_clients))


async def broadcast(event_type: str, data: dict[str, Any] = None):
    """Send an event to all connected dashboard clients."""
    global _clients
    if not _clients:
        return

    payload = json.dumps({
        "type":      event_type,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "data":      data or {},
    })

    dead = set()
    for ws in list(_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)

    _clients -= dead
