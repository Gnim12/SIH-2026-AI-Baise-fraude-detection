"""The signal ID registry: every code a B1 stage can emit, with the
modality and default severity used to build a Finding out of it in
convergence.py. Not exhaustive of every code any stage could ever emit
(STAGE_TIMEOUT/STAGE_EXCEPTION are runner-level and stay 'informational'
singletons) -- exhaustive of the codes this milestone's grouping rules
actually key off.
"""
from __future__ import annotations

CHECKSUM_CODES = frozenset({
    "MRZ_CHECKSUM_FAIL_LINE1", "MRZ_CHECKSUM_FAIL_LINE2", "MRZ_CHECKSUM_FAIL_COMPOSITE",
    "MRZ_DECODE_UNRECOVERABLE",
})

UNREADABLE_PREFIXES = ("MRZ_VIZ_FIELD_UNREADABLE_", "VIZ_FIELD_NOT_READABLE", "MRZ_BAND_NOT_FOUND")

MISMATCH_PREFIX = "MRZ_VIZ_MISMATCH_"


def is_checksum_failure(code: str) -> bool:
    return code in CHECKSUM_CODES


def is_unreadable(code: str) -> bool:
    return code.startswith(UNREADABLE_PREFIXES)


def is_crosscheck_mismatch(code: str) -> bool:
    return code.startswith(MISMATCH_PREFIX)
