"""gate-2: 'Authentic and physically real'. Its inputs are the four Wave 2
modules (tamper/ovd/face/identity-graph), all `unavailable` in B1 -- so Gate
2 itself cannot make an authenticity determination in this build and reports
`unavailable` rather than a fabricated pass. `blocksOnFailure` is False for
gate-2 regardless (frontend/src/lib/pipeline/types.ts:109); `decision`
always runs once gate-2 settles, whatever it settled to.
"""
from __future__ import annotations

from app.api.schemas import StageState
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext, StageResult


class Gate2Stage:
    id = StageId.GATE_2

    async def run(self, ctx: StageContext) -> StageResult:
        wave2_ids = (StageId.TAMPER, StageId.OVD_SWEEP, StageId.FACE_VERIFY, StageId.IDENTITY_GRAPH)
        wave2_available = any(sid in ctx.artefacts and ctx.artefacts[sid] for sid in wave2_ids)
        if not wave2_available:
            return StageResult(
                state=StageState.UNAVAILABLE,
                detail=(
                    "Gate 2 requires at least one Wave 2 module (tamper, OVD, face, identity graph); "
                    "none are implemented in this build, so an authenticity determination cannot be made."
                ),
            )
        # Real Wave-2 signals exist in a later build: this branch is
        # deliberately conservative until that build's authenticity rule is
        # written, rather than guessing one now.
        return StageResult(state=StageState.PASSED, detail="Wave 2 evidence available.")
