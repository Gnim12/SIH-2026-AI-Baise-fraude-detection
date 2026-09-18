"""BACKEND_BRIEF.md §5.1/§6.1: the two blocking gates.

M1 scope note: this milestone's task is the DAG skeleton and the *real* OCR
branch (BACKEND_BRIEF.md build order puts "Quality gate, classifier" in M2).
Both gates here are deliberately simple stand-ins -- quality always passes,
classification always returns a fixed PASSPORT/UTO guess -- so the DAG has
something to gate on end to end. Swapping in the real Laplacian-variance /
glare / DPI quality check and the real CNN+heuristic classifier is M2 work
and does not change either function's signature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityResult:
    ok: bool
    dpi: Optional[float] = None
    reason: Optional[str] = None
    hint: Optional[str] = None


@dataclass
class ClassifyResult:
    doc_type: str
    country: str
    version: str
    confidence: float


async def quality_gate(document_id: str, image_bytes: bytes) -> QualityResult:
    # M2 TODO: Laplacian variance > 120, glare clusters < 4%, DPI >= 250,
    # all four corners found, rectification residual within tolerance
    # (BACKEND_BRIEF.md §6.1). Stub: always passes.
    return QualityResult(ok=True, dpi=300.0)


async def classify(document_id: str, image_bytes: bytes) -> ClassifyResult:
    # M2 TODO: small CNN over MIDV + template matching + aspect-ratio/MRZ
    # heuristic (BACKEND_BRIEF.md §6.1). Stub: always PASSPORT/UTO v1.
    return ClassifyResult(doc_type="PASSPORT", country="UTO", version="v1", confidence=0.95)
