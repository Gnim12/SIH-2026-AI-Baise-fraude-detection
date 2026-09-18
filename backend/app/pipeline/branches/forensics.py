"""Forensics branch -- STUB until M4 (BACKEND_BRIEF.md build order).

Sleeps its BUDGET_MS and returns a hardcoded but valid result matching
case-01's genuine scenario (no findings): empty signals, no derived views yet
(ELA/noise/heatmap/FFT rendering is real forensic work, not fabricated here).

BACKEND_BRIEF.md §5.2, the rule that will bite you: this branch receives the
ORIGINAL, unrectified bytes -- EXIF, JPEG quantization tables and compression
history are destroyed the instant the file is re-encoded. The orchestrator
must call this with `doc.original_bytes`, never `doc.rectified`.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.contracts import Signal
from app.pipeline.context import PipelineContext

BUDGET_MS = 1200


@dataclass
class ForensicsResult:
    views: dict[str, str] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)


async def branch_forensics(original_bytes: bytes, ctx: PipelineContext) -> ForensicsResult:
    await asyncio.sleep(BUDGET_MS / 1000)
    return ForensicsResult(views={}, signals=[])
