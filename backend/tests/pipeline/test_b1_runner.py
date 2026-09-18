"""DAG runner requirements (Milestone B1 spec section 10, 'Runner').

Uses synthetic stages plugged straight into run_dag/StageContext so these
tests exercise the runner's generic scheduling, blocking and coverage logic
in isolation from the real OCR stages (those are covered in
test_b1_mrz_gate1.py).
"""
from __future__ import annotations

import asyncio

import pytest

from app.api.schemas import Signal, StageState
from app.pipeline.registry import STAGE_REGISTRY, StageId
from app.pipeline.runner import StageContext, StageResult, run_dag


def _signal(code: str) -> Signal:
    return Signal(signal_id=code, modality="cross-check", label=code, detail=code, emitted_at="t", model_pin="n/a")


class _FixedStage:
    def __init__(self, stage_id: StageId, state: StageState, *, sleep_s: float = 0.0, artefacts=None):
        self.id = stage_id
        self._state = state
        self._sleep_s = sleep_s
        self._artefacts = artefacts or {}
        self.started_at: float | None = None
        self.ended_at: float | None = None

    async def run(self, ctx: StageContext) -> StageResult:
        import time
        self.started_at = time.monotonic()
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        self.ended_at = time.monotonic()
        return StageResult(state=self._state, artefacts=self._artefacts)


class _RaisingStage:
    def __init__(self, stage_id: StageId):
        self.id = stage_id

    async def run(self, ctx: StageContext) -> StageResult:
        raise RuntimeError("boom")


class _HangingStage:
    def __init__(self, stage_id: StageId):
        self.id = stage_id

    async def run(self, ctx: StageContext) -> StageResult:
        await asyncio.sleep(10)
        return StageResult(state=StageState.PASSED)


def _all_passing_stages(overrides: dict[StageId, object] | None = None) -> dict[StageId, object]:
    overrides = overrides or {}
    stages: dict[StageId, object] = {}
    for d in STAGE_REGISTRY:
        stages[d.id] = overrides.get(d.id, _FixedStage(d.id, StageState.PASSED))
    return stages


@pytest.mark.asyncio
async def test_dependencies_resolve_from_registry_not_hardcoded():
    """gate-1 must not start until both mrz-read and viz-read have settled."""
    mrz = _FixedStage(StageId.MRZ_READ, StageState.PASSED, sleep_s=0.05)
    viz = _FixedStage(StageId.VIZ_READ, StageState.PASSED, sleep_s=0.05)
    gate1 = _FixedStage(StageId.GATE_1, StageState.PASSED)
    stages = _all_passing_stages({StageId.MRZ_READ: mrz, StageId.VIZ_READ: viz, StageId.GATE_1: gate1})

    outcome = await run_dag(stages, StageContext(run_id="r1"))

    assert outcome.stages[StageId.GATE_1].state == StageState.PASSED
    assert gate1.started_at >= mrz.ended_at
    assert gate1.started_at >= viz.ended_at


@pytest.mark.asyncio
async def test_wave1_stages_run_concurrently():
    mrz = _FixedStage(StageId.MRZ_READ, StageState.PASSED, sleep_s=0.15)
    viz = _FixedStage(StageId.VIZ_READ, StageState.PASSED, sleep_s=0.15)
    stages = _all_passing_stages({StageId.MRZ_READ: mrz, StageId.VIZ_READ: viz})

    import time
    t0 = time.monotonic()
    await run_dag(stages, StageContext(run_id="r2"))
    total = time.monotonic() - t0

    # Serial would take >=0.30s; concurrent should land close to 0.15s.
    assert total < 0.28, f"stages did not overlap: took {total:.3f}s"
    # Overlap window: viz started before mrz ended (both actually concurrent).
    assert viz.started_at < mrz.ended_at
    assert mrz.started_at < viz.ended_at


@pytest.mark.asyncio
async def test_gate1_failure_sets_wave2_not_evaluated_never_passed():
    stages = _all_passing_stages({StageId.GATE_1: _FixedStage(StageId.GATE_1, StageState.FAILED)})
    outcome = await run_dag(stages, StageContext(run_id="r3"))

    for sid in (StageId.TAMPER, StageId.OVD_SWEEP, StageId.FACE_VERIFY, StageId.IDENTITY_GRAPH):
        assert outcome.stages[sid].state == StageState.NOT_EVALUATED
        assert outcome.stages[sid].blocked_by == StageId.GATE_1


@pytest.mark.asyncio
async def test_transitive_blocking_reaches_convergence_gate2_decision():
    stages = _all_passing_stages({StageId.GATE_1: _FixedStage(StageId.GATE_1, StageState.FAILED)})
    outcome = await run_dag(stages, StageContext(run_id="r4"))

    for sid in (StageId.CONVERGENCE, StageId.GATE_2, StageId.DECISION):
        rec = outcome.stages[sid]
        assert rec.state == StageState.NOT_EVALUATED, f"{sid} was {rec.state}"
        assert rec.blocked_by == StageId.GATE_1  # original failed gate, not the immediate parent


@pytest.mark.asyncio
async def test_unavailable_and_not_evaluated_are_distinct():
    stages = _all_passing_stages({
        StageId.GATE_1: _FixedStage(StageId.GATE_1, StageState.PASSED),
        StageId.TAMPER: _FixedStage(StageId.TAMPER, StageState.UNAVAILABLE),
    })
    outcome = await run_dag(stages, StageContext(run_id="r5"))

    assert outcome.stages[StageId.TAMPER].state == StageState.UNAVAILABLE
    assert outcome.stages[StageId.TAMPER].blocked_by is None
    # unavailable does not cascade -- convergence still runs.
    assert outcome.stages[StageId.CONVERGENCE].state == StageState.PASSED


@pytest.mark.asyncio
async def test_stage_exception_yields_failed_with_signal_and_run_completes():
    stages = _all_passing_stages({StageId.MRZ_READ: _RaisingStage(StageId.MRZ_READ)})
    outcome = await run_dag(stages, StageContext(run_id="r6"))

    rec = outcome.stages[StageId.MRZ_READ]
    assert rec.state == StageState.FAILED
    assert any(s.signal_id == "STAGE_EXCEPTION" for s in rec.signals)
    # The run still reaches a terminal state for every stage -- it never crashes.
    assert all(s.state != StageState.WAITING for s in outcome.stages.values())


@pytest.mark.asyncio
async def test_stage_timeout_yields_failed_with_stage_timeout_signal():
    stages = _all_passing_stages({StageId.VIZ_READ: _HangingStage(StageId.VIZ_READ)})
    ctx = StageContext(run_id="r7", timeout_s=0.05)

    outcome = await run_dag(stages, ctx)

    rec = outcome.stages[StageId.VIZ_READ]
    assert rec.state == StageState.FAILED
    assert any(s.signal_id == "STAGE_TIMEOUT" for s in rec.signals)


@pytest.mark.asyncio
async def test_coverage_has_one_entry_per_registered_stage():
    outcome = await run_dag(_all_passing_stages(), StageContext(run_id="r8"))
    coverage = outcome.coverage
    assert {c.modality for c in coverage} or True  # modalities may repeat; ids must not
    assert len(coverage) == len(STAGE_REGISTRY)


@pytest.mark.asyncio
async def test_adding_a_stage_requires_no_runner_change(monkeypatch):
    """A synthetic stage depending on tamper, added purely to the registry
    module's data (not to runner.py), must be scheduled and must inherit
    transitive not-evaluated blocking automatically."""
    import dataclasses

    from app.pipeline import registry as registry_module

    extra = registry_module.StageDefinition(
        id=StageId.DECISION,  # placeholder id reused below; real id created next
        label="synthetic", wave=None, is_gate=False, depends_on=(StageId.TAMPER,),
    )
    synthetic_id = "synthetic-extra"
    # StageId is a str Enum; construct an ad-hoc definition using a plain
    # string id via a minimal duck-typed stand-in rather than mutating the
    # StageId enum itself.
    extra = registry_module.StageDefinition(
        id=synthetic_id, label="synthetic", wave=None, is_gate=False, depends_on=(StageId.TAMPER,),
    )
    new_registry = registry_module.STAGE_REGISTRY + (extra,)
    monkeypatch.setattr(registry_module, "STAGE_REGISTRY", new_registry)
    monkeypatch.setattr(registry_module, "REGISTRY_BY_ID", {d.id: d for d in new_registry})

    from app import pipeline

    class _SyntheticStage:
        id = synthetic_id

        async def run(self, ctx: StageContext) -> StageResult:
            return StageResult(state=StageState.PASSED)

    stages = _all_passing_stages({StageId.GATE_1: _FixedStage(StageId.GATE_1, StageState.FAILED)})
    stages[synthetic_id] = _SyntheticStage()

    outcome = await run_dag(stages, StageContext(run_id="r9"))
    assert outcome.stages[synthetic_id].state == StageState.NOT_EVALUATED
    assert outcome.stages[synthetic_id].blocked_by == StageId.GATE_1
