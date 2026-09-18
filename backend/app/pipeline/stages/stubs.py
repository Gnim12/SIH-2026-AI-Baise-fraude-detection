"""The four Wave 2 stages this build cannot perform: tamper forensics, OVD
sweep verification, face verification, identity-graph lookup. Each reports
`unavailable`, never `passed`/`failed` -- this system cannot perform these
checks at all in B1, which is a materially different statement from a gate
having blocked them (`not-evaluated`). See app/api/schemas.py's StageState
docstring.
"""
from __future__ import annotations

from app.api.schemas import StageState
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext, StageResult


class _UnavailableStage:
    id: StageId
    _detail: str

    async def run(self, ctx: StageContext) -> StageResult:
        return StageResult(state=StageState.UNAVAILABLE, detail=self._detail)


class TamperStage(_UnavailableStage):
    id = StageId.TAMPER
    _detail = "Tamper forensics is not implemented in this build."


class OvdSweepStage(_UnavailableStage):
    id = StageId.OVD_SWEEP
    _detail = "OVD sweep verification is not implemented in this build."


class FaceVerifyStage(_UnavailableStage):
    id = StageId.FACE_VERIFY
    _detail = "Face verification is not implemented in this build."


class IdentityGraphStage(_UnavailableStage):
    id = StageId.IDENTITY_GRAPH
    _detail = "Identity graph lookup is not implemented in this build."
