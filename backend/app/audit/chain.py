"""Milestone B2: the real hash-linked audit chain, replacing B1's fixture
(frontend `src/lib/audit/hash.ts`'s FNV-1a "stub digest" -- explicitly
labelled TODO there).

Three properties, enforced here:

- Append-only: this module exposes no update or delete path. `AuditChain`
  only ever inserts a new `B2AuditEntry` row; nothing in this codebase
  updates or deletes one (see app/storage/b2_models.py's docstring).
- Hash-linked: `entry_hash = sha256(canonical_json(payload))`, where
  `payload.previous_hash` is the prior entry's `entry_hash` for the same
  terminal. `verify()` walks the chain from genesis and recomputes every
  hash, reporting the first mismatched link by entry id.
- Complete at seal time: the payload below is exactly what
  `POST /api/sessions/{id}/decision` and `PUT /api/config/thresholds`
  gather *before* calling `append()` -- nothing is reconstructed later.

Canonical serialisation is written and tested before the hasher (this
module's docstring order mirrors the spec's own ordering requirement): if
the same entry could serialise two ways, verification would fail randomly
and nobody would find out why. `canonical_json()` is deliberately the only
function in this file that turns a payload into bytes -- `seal_entry()` and
`verify()` both call it, so there is exactly one encoding in the codebase,
never two that could drift apart.

Chain scope: one chain **per terminal**, not per case (spec §1's "Chain
scope") -- cross-case ordering on the same terminal is what makes insertion
or deletion of a single case's entry detectable. `terminal_id` is therefore
part of `AuditChain.append()`'s locking key, and `GENESIS_PREVIOUS_HASH` is
what the first entry on any given terminal references.

Concurrency: `append()` is guarded by a per-terminal `asyncio.Lock`
(`_terminal_locks`, same pattern as `app/api/b1_analyse.py`'s
`_analyse_locks`) so two officers sealing on the same terminal at the same
moment cannot both read the same "latest hash" and fork the chain. Reading
the latest entry, computing the new hash, and inserting the row all happen
inside the lock.
"""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.b2_models import B2AuditEntry

# frontend/src/lib/audit/types.ts's GENESIS_HASH -- the wire format for both
# sides is "sha256:" + 64 hex zeros. Every terminal's first entry references
# this exact string as its previousHash.
GENESIS_PREVIOUS_HASH = f"sha256:{'0' * 64}"


def _round_floats(value: Any) -> Any:
    """Fixed float formatting: every float in the payload is rounded to 6
    decimal places before serialisation, so a value that arrives as
    0.7500000000000001 on one process and 0.75 on another (both genuinely
    equal thresholds, differing only in float noise) canonicalises
    identically. Recurses through dicts/lists; every other type passes
    through unchanged. `bool` is checked before `int`/`float` because
    `isinstance(True, int)` is true in Python and would otherwise be
    rounded into `1.0`."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v) for v in value]
    return value


def canonical_json(payload: dict[str, Any]) -> bytes:
    """The one deterministic encoding: recursively sorted keys (via
    `sort_keys=True`, which sorts every nested dict, not just the top
    level), fixed float formatting (`_round_floats` above), UTF-8 bytes,
    and explicit `null` for every `None` (Python's `json` module already
    emits `null` for `None` rather than omitting the key, so a payload built
    with every key always present -- see `build_decision_payload` /
    `build_config_change_payload` below -- serialises identically whether a
    field's value is absent or explicitly null; there is no third state).
    No whitespace variance either: `separators=(",", ":")` removes the
    default space-after-separator, which is itself a source of two
    equally-valid-looking encodings of the same data."""
    canonical = _round_floats(payload)
    text = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8")


def compute_entry_hash(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return f"sha256:{digest}"


@dataclass
class SealResult:
    entry_id: str
    entry_hash: str
    previous_hash: str
    sealed_at: str
    payload: dict[str, Any] = field(repr=False)


# One lock per terminal -- guards the read-latest / compute-hash / insert
# sequence in `append()` so concurrent seals on the same terminal cannot
# both observe the same previousHash and fork the chain. Single-process
# only, matching this build's no-Celery in-process-asyncio architecture
# (same caveat as app/api/b1_analyse.py's `_analyse_locks`).
_terminal_locks: dict[str, asyncio.Lock] = {}


def _terminal_lock(terminal_id: str) -> asyncio.Lock:
    lock = _terminal_locks.get(terminal_id)
    if lock is None:
        lock = asyncio.Lock()
        _terminal_locks[terminal_id] = lock
    return lock


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


async def _latest_hash_for_terminal(db: AsyncSession, terminal_id: str) -> str:
    result = await db.execute(
        select(B2AuditEntry.entry_hash)
        .where(B2AuditEntry.terminal_id == terminal_id)
        .order_by(B2AuditEntry.sequence.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return row if row is not None else GENESIS_PREVIOUS_HASH


async def peek_latest_hash(db: AsyncSession, terminal_id: str) -> str:
    """Read-only lookup of the hash a *new* entry on `terminal_id` would
    reference as its `previousHash` right now. Backs the pre-seal audit
    preview (spec §5) so the client shows the real chain head instead of a
    client-only placeholder. Does not take `_terminal_lock` -- a decision
    sealed between this read and the officer's actual submit simply means
    the preview was briefly stale, which is fine for a preview; `append()`
    itself always re-reads under the lock, so the sealed entry is never
    wrong even if the preview was."""
    return await _latest_hash_for_terminal(db, terminal_id)


async def _next_sequence_for_terminal(db: AsyncSession, terminal_id: str) -> int:
    result = await db.execute(
        select(B2AuditEntry.sequence)
        .where(B2AuditEntry.terminal_id == terminal_id)
        .order_by(B2AuditEntry.sequence.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return (row or 0) + 1


def build_entry_payload(
    *,
    entry_id: str,
    session_id: Optional[str],
    run_id: Optional[str],
    previous_hash: str,
    sealed_at: str,
    officer_id: str,
    terminal_id: str,
    evidence_hashes: list[dict[str, str]],
    signal_ids: list[str],
    finding_dispositions: list[dict[str, Any]],
    model_pins: dict[str, str],
    thresholds: dict[str, float],
    recommendation: Optional[str],
    officer_action: Optional[str],
    divergence: Optional[str],
    divergence_reason: Optional[str],
    notes: Optional[str],
) -> dict[str, Any]:
    """Exactly the field set the spec's hash formula names, every key always
    present (explicit `null`, never an omitted key -- see `canonical_json`'s
    docstring). `entry_type`-specific fields that don't apply (e.g.
    `recommendation`/`officerAction`/`divergence` for a config-change entry)
    are `None`, not left out -- a decision entry and a config-change entry
    both hash the same field set, just with different fields populated."""
    return {
        "entryId": entry_id,
        "sessionId": session_id,
        "runId": run_id,
        "previousHash": previous_hash,
        "sealedAt": sealed_at,
        "officerId": officer_id,
        "terminalId": terminal_id,
        "evidenceHashes": evidence_hashes,
        "signalIds": signal_ids,
        "findingDispositions": finding_dispositions,
        "modelPins": model_pins,
        "thresholds": thresholds,
        "recommendation": recommendation,
        "officerAction": officer_action,
        "divergence": divergence,
        "divergenceReason": divergence_reason,
        "notes": notes,
    }


async def append(
    db: AsyncSession,
    *,
    entry_type: str,
    terminal_id: str,
    officer_id: str,
    session_id: Optional[str],
    run_id: Optional[str],
    evidence_hashes: list[dict[str, str]],
    signal_ids: list[str],
    finding_dispositions: list[dict[str, Any]],
    model_pins: dict[str, str],
    thresholds: dict[str, float],
    recommendation: Optional[str],
    officer_action: Optional[str],
    divergence: Optional[str],
    divergence_reason: Optional[str],
    notes: Optional[str],
) -> SealResult:
    """Appends one entry to `terminal_id`'s chain and returns it. Does not
    commit -- the caller (app/storage/b2_repositories.py's decision/config
    functions) wraps this in the same transaction as the domain row
    (`B2Decision`/`B2ConfigChange`) it belongs to, per spec §2: "a decision
    is only recorded if the audit entry seals -- one transaction, or the
    decision does not exist." Caller must `await db.commit()` afterwards.

    Holds `terminal_id`'s lock for the read-latest-then-insert sequence;
    releasing it only after the insert is added to the session (not after
    commit) is intentional -- the lock's job is to make sequence assignment
    and previousHash linkage race-free within this process, not to hold a
    lock across a DB round trip that isn't needed for that."""
    async with _terminal_lock(terminal_id):
        previous_hash = await _latest_hash_for_terminal(db, terminal_id)
        sequence = await _next_sequence_for_terminal(db, terminal_id)

        entry_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
        sealed_at = _now_iso()
        payload = build_entry_payload(
            entry_id=entry_id, session_id=session_id, run_id=run_id, previous_hash=previous_hash,
            sealed_at=sealed_at, officer_id=officer_id, terminal_id=terminal_id,
            evidence_hashes=evidence_hashes, signal_ids=signal_ids,
            finding_dispositions=finding_dispositions, model_pins=model_pins, thresholds=thresholds,
            recommendation=recommendation, officer_action=officer_action, divergence=divergence,
            divergence_reason=divergence_reason, notes=notes,
        )
        entry_hash = compute_entry_hash(payload)

        row = B2AuditEntry(
            id=entry_id, terminal_id=terminal_id, sequence=sequence, entry_type=entry_type,
            session_id=session_id, run_id=run_id, previous_hash=previous_hash, entry_hash=entry_hash,
            sealed_at=datetime.datetime.fromisoformat(sealed_at), officer_id=officer_id,
            payload_json=payload,
        )
        db.add(row)
        await db.flush()

        return SealResult(
            entry_id=entry_id, entry_hash=entry_hash, previous_hash=previous_hash, sealed_at=sealed_at,
            payload=payload,
        )


@dataclass
class ChainVerificationResult:
    intact: bool
    broken_entry_id: Optional[str] = None
    reason: Optional[str] = None
    entries_checked: int = 0


async def verify(db: AsyncSession, *, terminal_id: Optional[str] = None) -> ChainVerificationResult:
    """Walks the chain (optionally scoped to one terminal) from genesis in
    stored `sequence` order, recomputing each entry's hash from its stored
    payload and checking the previousHash linkage against the prior entry's
    stored entry_hash. Reports the first mismatched link by entry id. An
    empty chain is reported `intact=True` -- "nothing to verify" is not
    "broken" (spec's test list, explicitly)."""
    stmt = select(B2AuditEntry).order_by(B2AuditEntry.terminal_id.asc(), B2AuditEntry.sequence.asc())
    if terminal_id is not None:
        stmt = stmt.where(B2AuditEntry.terminal_id == terminal_id)
    rows = list((await db.execute(stmt)).scalars().all())

    expected_previous_by_terminal: dict[str, str] = {}
    checked = 0
    for row in rows:
        checked += 1
        expected_previous = expected_previous_by_terminal.get(row.terminal_id, GENESIS_PREVIOUS_HASH)
        if row.previous_hash != expected_previous:
            return ChainVerificationResult(
                intact=False, broken_entry_id=row.id,
                reason=f"{row.id} does not reference the hash of the preceding entry on terminal {row.terminal_id}.",
                entries_checked=checked,
            )
        recomputed = compute_entry_hash(row.payload_json)
        if recomputed != row.entry_hash:
            return ChainVerificationResult(
                intact=False, broken_entry_id=row.id,
                reason=f"{row.id} has been altered since it was sealed.",
                entries_checked=checked,
            )
        expected_previous_by_terminal[row.terminal_id] = row.entry_hash

    return ChainVerificationResult(intact=True, entries_checked=checked)
