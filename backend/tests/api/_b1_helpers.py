"""B1d: test-only in-process ground-truth injection.

POST /api/sessions/{id}/analyse no longer accepts `mrzGroundTruth` over HTTP
(that back door is closed -- see app/api/b1_analyse.py). A test that needs a
known MRZ to exercise a genuine Gate-1 pass calls this helper instead, which
drives the exact same `run_analysis()` entry point the HTTP route uses, but
in-process: it is not reachable over the network at all.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from app.api import b1_analyse
from app.storage import b1_repositories as repo
from app.storage.db import async_session_factory


def analyse_in_process(
    session_id: str, document_bytes: bytes, mrz_ground_truth: Optional[list[str]] = None,
) -> str:
    """Runs a full analysis synchronously (blocks until settled) and returns
    the run id. Bypasses HTTP entirely for both the run creation and the
    ground-truth injection."""

    async def _run() -> str:
        async with async_session_factory() as db:
            run = await repo.create_run(db, session_id)
        await b1_analyse._run_and_persist(run.id, session_id, document_bytes, mrz_ground_truth)
        return run.id

    return asyncio.run(_run())
