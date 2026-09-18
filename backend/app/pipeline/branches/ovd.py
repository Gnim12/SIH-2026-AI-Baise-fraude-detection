"""OVD (video-sweep angular reflectance) branch -- STUB, optional. Only run
when a video sweep was captured (BACKEND_BRIEF.md §5.3: `if sweep is not
None`). Sleeps its BUDGET_MS and returns no findings."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.contracts import Signal
from app.pipeline.context import PipelineContext

BUDGET_MS = 900


@dataclass
class OvdResult:
    signals: list[Signal] = field(default_factory=list)


async def branch_ovd(sweep, ctx: PipelineContext) -> OvdResult:
    await asyncio.sleep(BUDGET_MS / 1000)
    return OvdResult(signals=[])
