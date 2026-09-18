"""BACKEND_BRIEF.md §7: WS /ws/screening/{session_id}, progressive
ScreeningEvent stream with a replay buffer so a reconnecting client receives
every event it missed, in order (§5.4's "emit events as they land" and the
"officers' tablets drop connections" note)."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.auth.dependencies import get_officer_from_ws_cookie
from app.contracts.wire import to_wire
from app.pipeline.bus import event_bus
from app.storage.db import async_session_factory

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/screening/{session_id}")
async def ws_screening(websocket: WebSocket, session_id: str) -> None:
    # WS auth: a WebSocket handshake is a plain HTTP request before the
    # protocol upgrade, so the browser attaches the session cookie exactly
    # as on any other same-site request -- read it via websocket.cookies
    # (see app/auth/dependencies.py's get_officer_from_ws_cookie docstring).
    # No Authorization header is available here, which is why this can't
    # reuse the Depends(require_officer) HTTP dependency directly.
    async with async_session_factory() as auth_db:
        officer = await get_officer_from_ws_cookie(websocket, auth_db)
    if officer is None:
        await websocket.close(code=4401)
        return

    await websocket.accept()

    # The pipeline may not have opened the session's bus yet if the WS
    # connects before the POST's background task starts publishing --
    # poll briefly rather than 404ing a client that connected first.
    bus = event_bus.get(session_id)
    for _ in range(50):
        if bus is not None:
            break
        await asyncio.sleep(0.1)
        bus = event_bus.get(session_id)
    if bus is None:
        await websocket.send_json({"stage": "error", "message": "unknown session"})
        await websocket.close()
        return

    backlog, queue = bus.subscribe()
    try:
        for event in backlog:
            await websocket.send_json(to_wire(event))
        while True:
            event = await queue.get()
            if event is None:  # sentinel: session closed
                break
            await websocket.send_json(to_wire(event))
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(queue)
