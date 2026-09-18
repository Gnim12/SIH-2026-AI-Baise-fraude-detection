"""Per-session event bus with a replay buffer (BACKEND_BRIEF.md §7: "WS:
replay buffer per session so a client reconnecting mid-screening receives
every event it missed, in order. Officers' tablets drop connections.").

In-process asyncio only (BACKEND_BRIEF.md §2: "No Celery in v1") -- one
`SessionBus` per active session, held in `EventBus._sessions`, discarded once
nothing subscribes to it and the pipeline has finished. Every event a branch
publishes is appended to the replay log *and* fanned out to any currently
connected subscriber queues, so `stream.py` can hand a fresh subscriber the
full backlog then continue live.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.contracts.events import ScreeningEvent


@dataclass
class SessionBus:
    session_id: str
    replay: list[ScreeningEvent] = field(default_factory=list)
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _done: bool = False

    async def publish(self, event: ScreeningEvent) -> None:
        self.replay.append(event)
        for q in list(self._subscribers):
            await q.put(event)

    def close(self) -> None:
        self._done = True
        for q in list(self._subscribers):
            q.put_nowait(None)  # sentinel: no more events

    def subscribe(self) -> tuple[list[ScreeningEvent], "asyncio.Queue"]:
        """Returns (backlog, live_queue). Caller replays the backlog first,
        then reads live_queue until it yields None (session closed)."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return list(self.replay), q

    def unsubscribe(self, q: "asyncio.Queue") -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)


class EventBus:
    """Process-wide registry of SessionBus instances, one per in-flight or
    recently-finished screening session."""

    def __init__(self, *, max_finished_sessions: int = 256) -> None:
        self._sessions: dict[str, SessionBus] = {}
        self._max_finished_sessions = max_finished_sessions

    def open(self, session_id: str) -> SessionBus:
        bus = SessionBus(session_id=session_id)
        self._sessions[session_id] = bus
        return bus

    def get(self, session_id: str) -> SessionBus | None:
        return self._sessions.get(session_id)

    async def publish(self, session_id: str, event: ScreeningEvent) -> None:
        bus = self._sessions.get(session_id)
        if bus is None:
            raise KeyError(f"no open session bus for {session_id!r}")
        await bus.publish(event)

    def close(self, session_id: str) -> None:
        bus = self._sessions.get(session_id)
        if bus is not None:
            bus.close()


# One process-wide bus, imported by the orchestrator and the WS route.
event_bus = EventBus()
