"""The DAG runner -- Milestone B1's core deliverable.

Resolves execution order from `app/pipeline/registry.py` (never hardcoded
here), dispatches every stage whose dependencies are all settled together via
`asyncio.gather`, and propagates a blocking gate's failure to every
transitive dependent as `not-evaluated` -- a plain graph walk over
`registry.dependents_of`, not an `if` special-casing Wave 2. Adding a new
stage to the registry (a Wave 3, say) inherits blocking behaviour for free
and requires no change to this file; that is the acceptance bar the B1 spec
sets for this module.

No scalar risk score is computed or threaded through anywhere in this file.
The existing BACKEND_BRIEF.md fusion risk sum (app/pipeline/orchestrator.py)
is a separate, older pipeline; B1's `decision` stage (app/pipeline/stages/
decision.py) derives a categorical verdict only, from gate outcomes and
finding severity.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from app.api.schemas import MismatchRow, Modality, Signal, StageCoverage, StageEvent, StageState
from app.pipeline import registry as registry_module
from app.pipeline.events import RunBus
from app.pipeline.registry import StageDefinition, StageId, dependents_of

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclass
class StageContext:
    """Threaded through every stage. `artefacts` holds every stage settled so
    far this run (not just direct dependencies) -- how gate-1 gets at
    mrz-read's and viz-read's raw reader output without re-running them, and
    how convergence (whose registry dependsOn is the four wave-2 stages, not
    gate-1) can still see the MRZ/VIZ/Gate-1 signals it groups into findings,
    since by DAG topology every upstream stage is already terminal by the
    time a downstream stage dispatches."""

    run_id: str
    artefacts: dict[StageId, dict[str, Any]] = field(default_factory=dict)
    all_signals: list[Signal] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)  # document bytes, ground truth, etc.
    model_pins: dict[str, str] = field(default_factory=dict)
    timeout_s: float = DEFAULT_TIMEOUT_S


@dataclass
class StageResult:
    state: StageState
    detail: Optional[str] = None
    signals: list[Signal] = field(default_factory=list)
    artefacts: dict[str, Any] = field(default_factory=dict)
    mismatch_rows: Optional[list[MismatchRow]] = None


class Stage(Protocol):
    id: StageId

    async def run(self, ctx: StageContext) -> StageResult: ...


def stage_timeout_signal(stage_id: StageId, timeout_s: float) -> Signal:
    return Signal(
        signal_id="STAGE_TIMEOUT", modality=Modality.CROSS_CHECK, label="Stage timed out",
        detail=f"Stage {stage_id.value} did not complete within {timeout_s:.0f}s.",
        emitted_at=_now_iso(), model_pin="n/a",
    )


def stage_exception_signal(stage_id: StageId, exc: BaseException) -> Signal:
    # The client sees a signal, not a stack trace; the traceback is only logged.
    return Signal(
        signal_id="STAGE_EXCEPTION", modality=Modality.CROSS_CHECK, label="Stage failed unexpectedly",
        detail=f"Stage {stage_id.value} raised {type(exc).__name__}: {exc}",
        emitted_at=_now_iso(), model_pin="n/a",
    )


@dataclass
class _StageRecord:
    definition: StageDefinition
    state: StageState = StageState.WAITING
    detail: Optional[str] = None
    signals: list[Signal] = field(default_factory=list)
    artefacts: dict[str, Any] = field(default_factory=dict)
    mismatch_rows: Optional[list[MismatchRow]] = None
    blocked_by: Optional[StageId] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_ms: Optional[float] = None


TERMINAL_STATES = {
    StageState.PASSED, StageState.FAILED, StageState.NOT_EVALUATED, StageState.UNAVAILABLE,
}


@dataclass
class RunOutcome:
    run_id: str
    stages: dict[StageId, _StageRecord]
    all_signals: list[Signal]

    @property
    def coverage(self) -> list[StageCoverage]:
        # One entry per registered stage, per the B1 spec -- a run is never
        # partially reported as clean.
        modality_by_stage: dict[StageId, Modality] = {
            StageId.MRZ_READ: Modality.MRZ, StageId.VIZ_READ: Modality.VIZ, StageId.GATE_1: Modality.CROSS_CHECK,
            StageId.TAMPER: Modality.TAMPER, StageId.OVD_SWEEP: Modality.OVD, StageId.FACE_VERIFY: Modality.FACE,
            StageId.IDENTITY_GRAPH: Modality.IDENTITY_GRAPH, StageId.CONVERGENCE: Modality.CROSS_CHECK,
            StageId.GATE_2: Modality.CROSS_CHECK, StageId.DECISION: Modality.CROSS_CHECK,
        }
        out: list[StageCoverage] = []
        for stage_id, rec in self.stages.items():
            ran = rec.state in (StageState.PASSED, StageState.FAILED)
            out.append(StageCoverage(
                modality=modality_by_stage[stage_id],
                ran=ran,
                blocked_by=rec.blocked_by.value if rec.blocked_by else None,
                unavailable=rec.state == StageState.UNAVAILABLE,
            ))
        return out


def _cascade_not_evaluated(
    stages: dict[StageId, _StageRecord], failed_gate_id: StageId,
) -> list[StageId]:
    """BFS over the full transitive dependent closure of a failed blocking
    gate, mirroring frontend/src/lib/pipeline/reducer.ts's cascadeNotEvaluated
    exactly: every stage still WAITING and reachable from the gate through
    dependsOn edges (any depth, any node type -- not just gates) becomes
    not-evaluated, blockedBy the *original* failed gate, never 'passed' or
    'failed'."""
    changed: list[StageId] = []
    queue = list(dependents_of(failed_gate_id))
    seen: set[StageId] = set()
    while queue:
        stage_id = queue.pop(0)
        if stage_id in seen:
            continue
        seen.add(stage_id)
        rec = stages[stage_id]
        if rec.state == StageState.WAITING:
            rec.state = StageState.NOT_EVALUATED
            rec.blocked_by = failed_gate_id
            changed.append(stage_id)
        queue.extend(dependents_of(stage_id))
    return changed


async def _run_one_stage(stage: Stage, ctx: StageContext, timeout_s: float) -> StageResult:
    try:
        return await asyncio.wait_for(stage.run(ctx), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("stage %s timed out after %.0fs", stage.id.value, timeout_s)
        return StageResult(
            state=StageState.FAILED, detail=f"Timed out after {timeout_s:.0f}s.",
            signals=[stage_timeout_signal(stage.id, timeout_s)],
        )
    except Exception as exc:  # noqa: BLE001 -- a stage exception is `failed`, never a crashed run.
        logger.exception("stage %s raised", stage.id.value)
        logger.debug("stage %s traceback:\n%s", stage.id.value, traceback.format_exc())
        return StageResult(
            state=StageState.FAILED, detail=f"{type(exc).__name__}: {exc}",
            signals=[stage_exception_signal(stage.id, exc)],
        )


async def run_dag(
    stages_by_id: dict[StageId, Stage],
    ctx: StageContext,
    *,
    bus: Optional[RunBus] = None,
) -> RunOutcome:
    """Dispatch every registered stage. Stages whose dependencies are all
    settled dispatch together via asyncio.gather; a blocking gate's failure
    cascades not-evaluated to its full transitive dependent closure before
    the next wave is computed, so those stages never even dispatch."""
    records: dict[StageId, _StageRecord] = {
        d.id: _StageRecord(definition=d) for d in registry_module.STAGE_REGISTRY
    }
    all_signals: list[Signal] = []

    def ready_batch() -> list[StageId]:
        return [
            sid for sid, rec in records.items()
            if rec.state == StageState.WAITING
            and all(records[dep].state in TERMINAL_STATES for dep in rec.definition.depends_on)
        ]

    async def emit(build_event: "Callable[[], StageEvent]") -> None:
        # Deferred construction: skip building/validating a StageEvent
        # entirely when nothing is subscribed (also matters for a synthetic
        # stage id outside the closed StageId union in tests -- see
        # test_adding_a_stage_requires_no_runner_change).
        if bus is not None:
            await bus.publish(build_event())

    while True:
        batch = ready_batch()
        if not batch:
            break

        async def dispatch(stage_id: StageId) -> None:
            rec = records[stage_id]
            rec.started_at = _now_iso()
            rec.state = StageState.RUNNING
            started_at = rec.started_at

            def _started_event(stage_id: StageId = stage_id, started_at: str = started_at) -> StageEvent:
                return StageEvent(type="stage.started", stage_id=stage_id, state=StageState.RUNNING, at=started_at)

            await emit(_started_event)
            stage = stages_by_id[stage_id]
            stage_ctx = StageContext(
                run_id=ctx.run_id,
                artefacts={
                    sid: r.artefacts for sid, r in records.items() if r.state in TERMINAL_STATES
                },
                all_signals=list(all_signals),
                inputs=ctx.inputs, model_pins=ctx.model_pins, timeout_s=ctx.timeout_s,
            )
            t0 = time.monotonic()
            result = await _run_one_stage(stage, stage_ctx, ctx.timeout_s)
            duration_ms = round((time.monotonic() - t0) * 1000, 1)

            rec.state = result.state
            rec.detail = result.detail
            rec.signals = result.signals
            rec.artefacts = result.artefacts
            rec.mismatch_rows = result.mismatch_rows
            rec.ended_at = _now_iso()
            rec.duration_ms = duration_ms

        await asyncio.gather(*(dispatch(sid) for sid in batch))

        for stage_id in batch:
            rec = records[stage_id]
            all_signals.extend(rec.signals)

            def _settled_event(rec: _StageRecord = rec, stage_id: StageId = stage_id) -> StageEvent:
                return StageEvent(
                    type="stage.settled", stage_id=stage_id, state=rec.state, at=rec.ended_at or _now_iso(),
                    detail=rec.detail, signal_ids=[s.signal_id for s in rec.signals] or None,
                    mismatch_rows=rec.mismatch_rows, duration_ms=rec.duration_ms,
                )

            await emit(_settled_event)
            if rec.definition.is_gate and rec.definition.blocks_on_failure and rec.state == StageState.FAILED:
                cascaded = _cascade_not_evaluated(records, stage_id)
                for cid in cascaded:

                    def _cascaded_event(cid: StageId = cid, stage_id: StageId = stage_id) -> StageEvent:
                        return StageEvent(
                            type="stage.settled", stage_id=cid, state=StageState.NOT_EVALUATED,
                            at=_now_iso(), blocked_by=stage_id,
                        )

                    await emit(_cascaded_event)

    return RunOutcome(run_id=ctx.run_id, stages=records, all_signals=all_signals)
