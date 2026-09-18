"""Wires the B1 stage registry to the DAG runner and assembles the final
ScreeningResult wire object. This is the module app/api routes call.
"""
from __future__ import annotations

import datetime
from typing import Any, Optional

import numpy as np

from app.api.schemas import GraphResult, ScreeningResult, StageState, Verdict
from app.findings.convergence import group_signals
from app.pipeline.events import RunBus
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext, run_dag
from app.pipeline.stages.convergence import ConvergenceStage
from app.pipeline.stages.decision import DecisionStage
from app.pipeline.stages.gate1 import Gate1Stage
from app.pipeline.stages.gate2 import Gate2Stage
from app.pipeline.stages.mrz_read import MrzReadStage
from app.pipeline.stages.stubs import FaceVerifyStage, IdentityGraphStage, OvdSweepStage, TamperStage
from app.pipeline.stages.viz_read import VizReadStage


def default_stages() -> dict[StageId, Any]:
    return {
        s.id: s
        for s in (
            MrzReadStage(), VizReadStage(), Gate1Stage(), TamperStage(), OvdSweepStage(),
            FaceVerifyStage(), IdentityGraphStage(), ConvergenceStage(), Gate2Stage(), DecisionStage(),
        )
    }


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


async def run_analysis(
    run_id: str,
    document_image: np.ndarray,
    *,
    document_id: str = "document",
    mrz_ground_truth: Optional[list[str]] = None,
    bus: Optional[RunBus] = None,
    timeout_s: float = 30.0,
) -> ScreeningResult:
    ctx = StageContext(
        run_id=run_id,
        inputs={
            "document_image": document_image, "document_id": document_id,
            "mrz_ground_truth": mrz_ground_truth,
        },
        timeout_s=timeout_s,
    )
    outcome = await run_dag(default_stages(), ctx, bus=bus)
    records = outcome.stages

    mrz_schema = records[StageId.MRZ_READ].artefacts.get("mrz_schema")
    cross_check_table = records[StageId.GATE_1].artefacts.get("cross_check_table", [])

    convergence_rec = records[StageId.CONVERGENCE]
    if convergence_rec.state == StageState.PASSED:
        findings = convergence_rec.artefacts.get("findings", [])
    else:
        # Gate 1 failed and cascaded convergence to not-evaluated -- still
        # group whatever signals mrz-read/viz-read/gate-1 emitted, so the
        # officer sees *why* Gate 1 failed, not an empty findings list.
        findings = group_signals(outcome.all_signals)

    decision_rec = records[StageId.DECISION]
    if decision_rec.state == StageState.PASSED:
        verdict = decision_rec.artefacts["verdict"]
        reason = decision_rec.artefacts["reason"]
    else:
        # decision was cascaded to not-evaluated (gate-1 failed). No verdict
        # member in findings/types.ts maps to "recapture suggested" -- see
        # app/api/schemas.py's ScreeningResult MISMATCH note. review-required
        # is the closest existing member; flagged is not used here because
        # B1 makes no fraud determination.
        gate1_rec = records[StageId.GATE_1]
        verdict = Verdict.REVIEW_REQUIRED
        reason = gate1_rec.detail or "Gate 1 did not pass; recapture suggested."

    return ScreeningResult(
        session_id=run_id, verdict=verdict, reason=reason, findings=findings,
        coverage=outcome.coverage, mrz=mrz_schema, cross_check=cross_check_table,
        identity_graph=GraphResult(ran=False, has_prior_encounter=False, enrolment_count=0),
        completed_at=_now_iso(),
    )
