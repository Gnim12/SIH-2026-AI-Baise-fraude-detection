"""decision: synthesizes the final categorical verdict from convergence's
findings. No scalar score anywhere -- if this ever computes a weighted sum,
that is a bug, not a feature (per the B1 spec's explicit instruction).

Only reached when gate-1 passed: gate-1 blocks on failure, so a gate-1
failure cascades 'decision' itself to not-evaluated (see runner.py's
_cascade_not_evaluated) and the caller must synthesize RECAPTURE_SUGGESTED
directly from gate-1's failure instead of from this stage's output -- see
the MISMATCH note in app/api/schemas.py's ScreeningResult docstring
(findings/types.ts's Verdict has no 'recapture' member).
"""
from __future__ import annotations

from app.api.schemas import Finding, Severity, StageState, Verdict
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext, StageResult

_SEVERITY_RANK = {Severity.CRITICAL: 3, Severity.HIGH: 2, Severity.MODERATE: 1, Severity.INFORMATIONAL: 0}


def _highest_severity(findings: list[Finding]) -> Severity | None:
    significant = [f for f in findings if f.severity != Severity.INFORMATIONAL]
    if not significant:
        return None
    return max((f.severity for f in significant), key=lambda s: _SEVERITY_RANK[s])


def derive_verdict(findings: list[Finding]) -> tuple[Verdict, str]:
    converged_high = any(
        f.severity in (Severity.HIGH, Severity.CRITICAL) and len(f.signals) > 1 for f in findings
    )
    highest = _highest_severity(findings)

    if converged_high or highest == Severity.CRITICAL:
        top = max(
            (f for f in findings if f.severity != Severity.INFORMATIONAL),
            key=lambda f: _SEVERITY_RANK[f.severity],
        )
        return Verdict.FLAGGED, top.hypothesis
    if highest in (Severity.HIGH, Severity.MODERATE):
        top = max(
            (f for f in findings if f.severity != Severity.INFORMATIONAL),
            key=lambda f: _SEVERITY_RANK[f.severity],
        )
        return Verdict.REVIEW_REQUIRED, top.hypothesis
    return (
        Verdict.CLEARED,
        "No adverse findings from the checks this build can perform (MRZ decode, VIZ decode, "
        "MRZ↔VIZ cross-check). Wave 2 checks were not run -- see coverage.",
    )


class DecisionStage:
    id = StageId.DECISION

    async def run(self, ctx: StageContext) -> StageResult:
        findings = ctx.artefacts.get(StageId.CONVERGENCE, {}).get("findings", [])
        verdict, reason = derive_verdict(findings)
        return StageResult(
            state=StageState.PASSED, detail=reason,
            artefacts={"verdict": verdict, "reason": reason},
        )
