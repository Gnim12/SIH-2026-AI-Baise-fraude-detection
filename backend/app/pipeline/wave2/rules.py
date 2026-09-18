"""Rules engine (wave 2) -- STUB. Real deterministic checks (check digits,
ISO-3166, date logic, country regex, visa windows -- BACKEND_BRIEF.md §6.5)
land in M2/M3. Returns no findings for now."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.contracts import Signal
from app.pipeline.context import PipelineContext

BUDGET_MS = 300


@dataclass
class RulesResult:
    signals: list[Signal] = field(default_factory=list)


async def branch_rules(ocr_result, ctx: PipelineContext) -> RulesResult:
    await asyncio.sleep(BUDGET_MS / 1000)
    return RulesResult(signals=[])
