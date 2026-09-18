"""Template branch (one-class normality per country/version) -- STUB until
M4. Sleeps its BUDGET_MS and returns no anomaly, matching case-01's genuine
scenario."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.contracts import Signal
from app.pipeline.context import PipelineContext

BUDGET_MS = 500


@dataclass
class TemplateResult:
    signals: list[Signal] = field(default_factory=list)


async def branch_template(document_rectified, ctx: PipelineContext) -> TemplateResult:
    await asyncio.sleep(BUDGET_MS / 1000)
    return TemplateResult(signals=[])
