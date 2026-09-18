"""B2: canonical serialisation, genesis, linkage, tamper detection,
concurrency, and empty-chain verification for app/audit/chain.py."""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.audit import chain
from app.main import app
from app.storage.b2_models import B2AuditEntry
from app.storage.db import async_session_factory


def _unique_terminal(label: str) -> str:
    """A fresh terminal id per test invocation -- tests/_test_screening.db
    is a real file that persists across separate `pytest` runs (not reset
    per-session), so a fixed literal like "TERM-CHAIN-1" would accumulate
    leftover rows from every previous run and break count-based assertions
    on a rerun."""
    return f"TERM-{label}-{uuid.uuid4().hex[:8]}"


def test_canonical_serialisation_is_deterministic_regardless_of_key_order():
    payload_a = {"b": 1, "a": {"z": 2, "y": 3}, "c": [3, 2, 1]}
    payload_b = {"a": {"y": 3, "z": 2}, "c": [3, 2, 1], "b": 1}
    assert chain.canonical_json(payload_a) == chain.canonical_json(payload_b)


def test_canonical_serialisation_is_deterministic_across_float_noise():
    payload_a = {"threshold": 0.75}
    payload_b = {"threshold": 0.7500000000000001}
    assert chain.canonical_json(payload_a) == chain.canonical_json(payload_b)


def test_canonical_serialisation_stable_across_repeated_calls():
    payload = {"entryId": "AUD-1", "notes": "hello", "divergenceReason": None, "thresholds": {"a": 0.5}}
    first = chain.canonical_json(payload)
    for _ in range(5):
        assert chain.canonical_json(payload) == first


def _make_payload(**overrides):
    base = dict(
        entry_id="AUD-TEST", session_id="SESSION-1", run_id="RUN-1", previous_hash=chain.GENESIS_PREVIOUS_HASH,
        sealed_at="2026-01-01T00:00:00+00:00", officer_id="OFF-1", terminal_id="TERM-1",
        evidence_hashes=[], signal_ids=[], finding_dispositions=[], model_pins={}, thresholds={},
        recommendation="cleared", officer_action="clear", divergence="none", divergence_reason=None,
        notes="ok",
    )
    base.update(overrides)
    return chain.build_entry_payload(**base)


def test_genesis_previous_hash_is_64_zero_hex_with_sha256_prefix():
    assert chain.GENESIS_PREVIOUS_HASH == f"sha256:{'0' * 64}"


def test_entry_hash_changes_if_any_field_changes():
    payload = _make_payload()
    h1 = chain.compute_entry_hash(payload)
    payload2 = _make_payload(notes="different")
    h2 = chain.compute_entry_hash(payload2)
    assert h1 != h2


async def _append_decision_entry(db, *, terminal_id: str, session_id: str) -> chain.SealResult:
    result = await chain.append(
        db, entry_type="decision", terminal_id=terminal_id, officer_id="OFF-1", session_id=session_id,
        run_id="RUN-1", evidence_hashes=[{"kind": "document-still", "sha256": "sha256:abc"}],
        signal_ids=["SIG-1"], finding_dispositions=[{"findingId": "F-1", "disposition": "accepted"}],
        model_pins={"m": "1.0"}, thresholds={"t": 0.5}, recommendation="cleared", officer_action="clear",
        divergence="none", divergence_reason=None, notes="ok",
    )
    await db.commit()
    return result


def test_each_entry_references_its_predecessor():
    with TestClient(app):
        async def _run():
            terminal_id = _unique_terminal("CHAIN-1")
            async with async_session_factory() as db:
                first = await _append_decision_entry(db, terminal_id=terminal_id, session_id="S1")
                second = await _append_decision_entry(db, terminal_id=terminal_id, session_id="S2")
            assert first.previous_hash == chain.GENESIS_PREVIOUS_HASH
            assert second.previous_hash == first.entry_hash

        asyncio.run(_run())


def test_altering_a_stored_field_breaks_verification_and_names_first_bad_link():
    with TestClient(app):
        async def _run():
            terminal_id = _unique_terminal("CHAIN-2")
            async with async_session_factory() as db:
                first = await _append_decision_entry(db, terminal_id=terminal_id, session_id="S1")
                await _append_decision_entry(db, terminal_id=terminal_id, session_id="S2")

            async with async_session_factory() as db:
                row = await db.get(B2AuditEntry, first.entry_id)
                row.payload_json = {**row.payload_json, "notes": "TAMPERED"}
                await db.commit()

            async with async_session_factory() as db:
                result = await chain.verify(db, terminal_id=terminal_id)
            assert result.intact is False
            assert result.broken_entry_id == first.entry_id

        asyncio.run(_run())


def test_concurrent_seals_produce_a_single_linear_chain_no_fork():
    with TestClient(app):
        async def _run():
            terminal_id = _unique_terminal("CONCURRENT")
            async with async_session_factory() as db0:
                await db0.execute(select(B2AuditEntry).where(B2AuditEntry.terminal_id == terminal_id))

            async def _seal(i: int):
                async with async_session_factory() as db:
                    return await _append_decision_entry(db, terminal_id=terminal_id, session_id=f"S-{i}")

            results = await asyncio.gather(*(_seal(i) for i in range(10)))

            async with async_session_factory() as db:
                rows_result = await db.execute(
                    select(B2AuditEntry).where(B2AuditEntry.terminal_id == terminal_id)
                    .order_by(B2AuditEntry.sequence.asc())
                )
                rows = list(rows_result.scalars().all())

            assert len(rows) == 10
            sequences = [r.sequence for r in rows]
            assert sequences == list(range(1, 11))  # no gaps, no duplicates -- a linear chain
            expected_previous = chain.GENESIS_PREVIOUS_HASH
            for row in rows:
                assert row.previous_hash == expected_previous
                expected_previous = row.entry_hash

            async with async_session_factory() as db:
                verify_result = await chain.verify(db, terminal_id=terminal_id)
            assert verify_result.intact is True
            assert len(results) == 10

        asyncio.run(_run())


def test_verification_of_an_empty_chain_reports_intact_not_broken():
    with TestClient(app):
        async def _run():
            async with async_session_factory() as db:
                result = await chain.verify(db, terminal_id=_unique_terminal("NEVER-USED"))
            assert result.intact is True
            assert result.entries_checked == 0

        asyncio.run(_run())
