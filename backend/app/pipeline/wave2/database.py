"""Database branch (wave 2) -- STUB. Real watchlist/blacklist/gallery/graph
queries (BACKEND_BRIEF.md §6.6) land in M5. Returns an empty, clean graph."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.contracts import GraphResult, Signal
from app.pipeline.context import PipelineContext

BUDGET_MS = 400


@dataclass
class DatabaseResult:
    graph: GraphResult
    signals: list[Signal] = field(default_factory=list)


async def branch_database(fields, ctx: PipelineContext) -> DatabaseResult:
    await asyncio.sleep(BUDGET_MS / 1000)
    return DatabaseResult(graph=GraphResult(prior_encounters=[], conflicts=0, impossible_travel=False), signals=[])
