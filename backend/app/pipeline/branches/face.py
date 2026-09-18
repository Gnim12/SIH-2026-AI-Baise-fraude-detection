"""Face branch -- STUB until M4 (BACKEND_BRIEF.md build order: SCRFD + PAD +
ArcFace land there). Sleeps its BUDGET_MS and returns a hardcoded but valid
result matching case-01's genuine scenario: live, matched, above threshold."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.contracts import FaceResult, Signal
from app.pipeline.context import PipelineContext

BUDGET_MS = 600


@dataclass
class FaceBranchResult:
    face: FaceResult
    signals: list[Signal] = field(default_factory=list)


async def branch_face(
    document_rectified, live_frame, ctx: PipelineContext, *, capture_method: str = "live",
) -> FaceBranchResult:
    await asyncio.sleep(BUDGET_MS / 1000)
    # capture_method is plumbed through but not yet acted on: real PAD
    # (passive anti-spoofing) lands at M4 alongside SCRFD/ArcFace. Once it
    # does, an "upload" capture_method should force pad_verdict to
    # "not_run" rather than the hardcoded "live" below, since a still photo
    # was never subjected to liveness detection at all.
    face = FaceResult(
        status="MATCH", similarity=0.93, threshold=0.38,
        pad_verdict="live", ghost_portrait_consistent=True,
        capture_method=capture_method,
    )
    return FaceBranchResult(face=face, signals=[])
