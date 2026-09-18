"""Per-run event bus with a replay buffer, keyed by run id.

Same in-process asyncio pattern as app/pipeline/bus.py (used by the older
BACKEND_BRIEF.md session pipeline) -- one bus per in-flight run, a replay log
so a client reconnecting mid-run receives every event it missed in order,
transport-agnostic (the API layer decides WS vs SSE; B1 kept WS per the
existing app/api/stream.py convention rather than adding a parallel SSE
route).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from app.api.schemas import StageEvent

_QueueItem = Optional[StageEvent]  # None is the "run finished" sentinel


@dataclass
class RunBus:
    run_id: str
    replay: list[StageEvent] = field(default_factory=list)
    _subscribers: list["asyncio.Queue[_QueueItem]"] = field(default_factory=list)
    _done: bool = False

    async def publish(self, event: StageEvent) -> None:
        self.replay.append(event)
        for q in list(self._subscribers):
            await q.put(event)

    def close(self) -> None:
        self._done = True
        for q in list(self._subscribers):
            q.put_nowait(None)  # sentinel: run finished

    def subscribe(self) -> tuple[list[StageEvent], "asyncio.Queue[_QueueItem]"]:
        q: "asyncio.Queue[_QueueItem]" = asyncio.Queue()
        self._subscribers.append(q)
        return list(self.replay), q

    def unsubscribe(self, q: "asyncio.Queue[_QueueItem]") -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)


class RunEventBus:
    def __init__(self) -> None:
        self._runs: dict[str, RunBus] = {}

    def open(self, run_id: str) -> RunBus:
        bus = RunBus(run_id=run_id)
        self._runs[run_id] = bus
        return bus

    def register(self, run_id: str, bus: RunBus) -> RunBus:
        """Like `open`, but for a caller (app/api/b1_analyse.py) that needs a
        RunBus subclass -- e.g. one that also persists each event to the DB
        before fanning it out to subscribers."""
        self._runs[run_id] = bus
        return bus

    def get(self, run_id: str) -> RunBus | None:
        return self._runs.get(run_id)

    async def publish(self, run_id: str, event: StageEvent) -> None:
        bus = self._runs.get(run_id)
        if bus is None:
            raise KeyError(f"no open run bus for {run_id!r}")
        await bus.publish(event)

    def close(self, run_id: str) -> None:
        bus = self._runs.get(run_id)
        if bus is not None:
            bus.close()


run_event_bus = RunEventBus()
