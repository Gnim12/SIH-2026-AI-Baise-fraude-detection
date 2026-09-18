"""convergence: groups every signal emitted so far into findings
(app/findings/convergence.py). Not itself a gate -- it always 'passes' if it
runs to completion; its output is what gate-2 and decision consume."""
from __future__ import annotations

from app.api.schemas import StageState
from app.findings.convergence import group_signals
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext, StageResult


class ConvergenceStage:
    id = StageId.CONVERGENCE

    async def run(self, ctx: StageContext) -> StageResult:
        findings = group_signals(ctx.all_signals)
        return StageResult(
            state=StageState.PASSED,
            detail=f"{len(findings)} finding(s) grouped from {len(ctx.all_signals)} signal(s).",
            artefacts={"findings": findings},
        )
