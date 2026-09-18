"""B1b: POST /api/sessions/{id}/analyse, WS /api/sessions/{id}/events,
GET /api/sessions/{id}/result.

Wires app/pipeline/analyse.py's run_analysis() -- complete and tested since
B1a, but not called from any route until this milestone -- to the new HTTP/WS
surface. See this module's WS handler docstring for the wire-shape
translation this requires between app/api/schemas.py's StageEvent and the
frontend's frontend/src/lib/pipeline/reducer.ts PipelineEvent union.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ScreeningResult, StageEvent, StageState
from app.config import settings
from app.pipeline.analyse import run_analysis
from app.pipeline.events import RunBus, run_event_bus
from app.storage import b1_repositories as repo
from app.storage.b1_models import B1StageEvent
from app.storage.db import async_session_factory, get_session

from .b1_errors import ApiError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["b1-analyse"])

REQUIRED_ARTEFACT_KINDS = ("document-still",)
HEARTBEAT_INTERVAL_S = 20.0

# Guards the check-then-create sequence in start_analysis_route: two POSTs
# for the same session arriving close together (a double-click, or React
# StrictMode's dev-mode double effect invocation) would otherwise both see
# "no run in flight" and each create their own run, running the pipeline
# twice in parallel. Single-process only, matching this build's no-Celery,
# in-process-asyncio architecture (BACKEND_BRIEF.md §2) -- a multi-worker
# deployment would need a DB-level constraint instead.
_analyse_locks: dict[str, asyncio.Lock] = {}


def _analyse_lock(session_id: str) -> asyncio.Lock:
    lock = _analyse_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _analyse_locks[session_id] = lock
    return lock


class PersistingRunBus(RunBus):
    """Persists every event to `b1_stage_event` (append-only, monotonic
    sequence) before fanning it out to WS subscribers, so a process restart
    still has the full ordered log for replay and B2's audit chain."""

    def __init__(self, run_id: str, session_id: str) -> None:
        super().__init__(run_id=run_id)
        self._session_id = session_id
        # Wave-1 stages (mrz-read, viz-read) dispatch concurrently via
        # asyncio.gather and publish independently -- without this lock, two
        # concurrent `select(max(sequence))`-then-insert round trips race and
        # can compute the same "next" sequence number for both.
        self._sequence_lock = asyncio.Lock()

    async def publish(self, event: StageEvent) -> None:
        async with self._sequence_lock:
            async with async_session_factory() as db:
                await repo.append_stage_event(
                    db, self.run_id, event.stage_id.value, event.state.value, event.detail, event.at,
                )
        await super().publish(event)


def _decode_document_image(raw: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


async def _run_and_persist(
    run_id: str, session_id: str, document_bytes: bytes, mrz_ground_truth: Optional[list[str]],
) -> None:
    bus = run_event_bus.register(run_id, PersistingRunBus(run_id, session_id))
    image = _decode_document_image(document_bytes)

    try:
        if image is None:
            raise ValueError("stored document-still artefact is not a decodable image")
        result = await run_analysis(
            run_id, image, mrz_ground_truth=mrz_ground_truth, bus=bus, timeout_s=settings.b1_stage_timeout_s,
        )
        async with async_session_factory() as db:
            await repo.settle_run(db, run_id, session_id, result)
    except Exception:
        # A stage failing is reported as a stage result, never an API error
        # (B1b spec §3) -- this branch is only for a genuine crash in the
        # runner/persistence layer itself. Log the real traceback with the
        # run/session id. Marking the run "settled" here (as an earlier
        # version of this handler did) is wrong: GET /result treats
        # `result_json is None` as still in-flight regardless of `status`,
        # so a crashed run would poll as 409 "in progress" forever. Use a
        # distinct "errored" status so GET /result and the WS route can each
        # surface a real failure instead of hanging.
        logger.exception("run %s for session %s failed unexpectedly", run_id, session_id)
        async with async_session_factory() as db:
            run = await repo.get_run(db, run_id)
            if run is not None:
                run.status = "errored"
                run.ended_at = datetime.datetime.now(datetime.timezone.utc)
                await db.commit()
    finally:
        bus.close()


@router.post("/{session_id}/analyse")
async def start_analysis_route(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """B1d closed the `mrzGroundTruth` HTTP back door that used to live on
    this route's request body (no trained CRNN checkpoint exists yet, B1c,
    so it was a dev/test-only affordance to let a caller supply the answer
    the system is meant to determine). `run_analysis()`'s own
    `mrz_ground_truth` parameter still exists for tests, but the only way to
    reach it now is in-process -- see tests/api/_b1_helpers.py -- never over
    the network. This route always passes `None`."""
    session = await repo.get_session(db, session_id)
    if session is None:
        raise ApiError(404, "session_not_found", f"No session found with id {session_id!r}.")

    async with _analyse_lock(session_id):
        in_flight = await repo.get_in_flight_run(db, session_id)
        if in_flight is not None:
            return {"runId": in_flight.id}

        artefacts = await repo.list_artefacts(db, session_id)
        present_kinds = {a.kind for a in artefacts}
        for required in REQUIRED_ARTEFACT_KINDS:
            if required not in present_kinds:
                raise ApiError(
                    422, "missing_artefact",
                    f"Cannot analyse: {required} has not been captured for this session.",
                    {"missingKind": required},
                )

        document_artefact = next(a for a in artefacts if a.kind == "document-still")
        from pathlib import Path
        document_bytes = Path(document_artefact.stored_path).read_bytes()

        run = await repo.create_run(db, session_id)
        asyncio.create_task(_run_and_persist(run.id, session_id, document_bytes, None))
        return {"runId": run.id}


def _translate_event(event: B1StageEvent | StageEvent) -> Optional[dict[str, object]]:
    """StageEvent ('stage.started'|'stage.settled', unified `state`) ->
    frontend PipelineEvent shape (hyphenated type, outcome baked into it, no
    `state` field) -- see frontend/src/lib/pipeline/reducer.ts.

    `not-evaluated` is never forwarded: the frontend derives the cascade
    itself client-side (reducer.ts's cascadeNotEvaluated) the moment it sees
    the triggering gate's own stage-failed event, and has no wire event for
    it at all. `unavailable` has no frontend event either -- this build adds
    one (`stage-unavailable`) since without it the four Wave 2 stub stages
    would sit at 'waiting' forever in a real run (see the B1b report)."""
    if isinstance(event, B1StageEvent):
        stage_id, state, at, detail = event.stage_id, event.state, event.occurred_at.isoformat(), event.detail
        signal_ids = None
        mismatch_rows = None
    else:
        stage_id, state, at, detail = event.stage_id.value, event.state.value, event.at, event.detail
        signal_ids = [s for s in (event.signal_ids or [])] or None
        mismatch_rows = [m.model_dump(mode="json", by_alias=True) for m in (event.mismatch_rows or [])] or None

    if state == StageState.RUNNING.value:
        return {"type": "stage-started", "stageId": stage_id, "at": at}
    if state == StageState.PASSED.value:
        return {"type": "stage-passed", "stageId": stage_id, "at": at, "detail": detail, "signalIds": signal_ids}
    if state == StageState.FAILED.value:
        return {
            "type": "stage-failed", "stageId": stage_id, "at": at, "detail": detail,
            "signalIds": signal_ids, "mismatchRows": mismatch_rows,
        }
    if state == StageState.UNAVAILABLE.value:
        return {"type": "stage-unavailable", "stageId": stage_id, "at": at, "detail": detail}
    return None  # not-evaluated / waiting: never sent over the wire


@router.websocket("/{session_id}/events")
async def session_events_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    async with async_session_factory() as db:
        run = await repo.get_latest_run(db, session_id)

    if run is None:
        await websocket.send_json({"type": "run.error", "message": "No analysis has been started for this session."})
        await websocket.close()
        return

    bus = run_event_bus.get(run.id) if run.status == "running" else None

    if bus is None:
        # Either the run already settled, or this process restarted since
        # the run started (the in-memory bus is gone) -- replay purely from
        # the durable log, then stop (no more live events will ever arrive).
        async with async_session_factory() as db:
            rows = await repo.list_stage_events(db, run.id)
        try:
            for row in rows:
                translated = _translate_event(row)
                if translated is not None:
                    await websocket.send_json(translated)
            async with async_session_factory() as db:
                run = await repo.get_run(db, run.id)
            if run is not None and run.status == "errored":
                await websocket.send_json({
                    "type": "run.error", "runId": run.id,
                    "message": "Analysis could not complete due to an internal error.",
                })
            else:
                await websocket.send_json({
                    "type": "run.settled", "runId": run.id if run else None,
                    "verdict": run.verdict if run else None,
                })
        except WebSocketDisconnect:
            pass
        await websocket.close()
        return

    backlog, queue = bus.subscribe()
    try:
        for event in backlog:
            translated = _translate_event(event)
            if translated is not None:
                await websocket.send_json(translated)

        while True:
            try:
                queue_item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL_S)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
                continue

            if queue_item is None:  # sentinel: run settled (or errored)
                async with async_session_factory() as db:
                    settled_run = await repo.get_run(db, run.id)
                if settled_run is not None and settled_run.status == "errored":
                    await websocket.send_json({
                        "type": "run.error", "runId": run.id,
                        "message": "Analysis could not complete due to an internal error.",
                    })
                else:
                    await websocket.send_json({
                        "type": "run.settled", "runId": run.id,
                        "verdict": settled_run.verdict if settled_run else None,
                    })
                break

            translated = _translate_event(queue_item)
            if translated is not None:
                await websocket.send_json(translated)
    except WebSocketDisconnect:
        pass  # disconnection never affects the run itself
    finally:
        bus.unsubscribe(queue)


@router.get("/{session_id}/result")
async def get_result_route(session_id: str, db: AsyncSession = Depends(get_session)) -> ScreeningResult:
    session = await repo.get_session(db, session_id)
    if session is None:
        raise ApiError(404, "session_not_found", f"No session found with id {session_id!r}.")

    run = await repo.get_latest_run(db, session_id)
    if run is None:
        raise ApiError(404, "no_run", "No analysis has been started for this session.")

    if run.status == "errored":
        raise ApiError(
            500, "run_failed",
            "Analysis could not complete due to an internal error. Start a new analysis.",
            {"runId": run.id},
        )

    if run.status != "settled" or run.result_json is None:
        raise ApiError(
            409, "run_in_progress", f"Analysis is still {run.status}. Poll again once it settles.",
            {"runId": run.id, "status": run.status},
        )

    return ScreeningResult.model_validate(run.result_json)
