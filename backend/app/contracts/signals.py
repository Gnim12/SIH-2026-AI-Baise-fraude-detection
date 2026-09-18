"""BACKEND_BRIEF.md §4: Region and Signal, ported field-for-field from
FRONTEND_BRIEF.md §3's `Region` / `Signal` TypeScript interfaces."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

Module = Literal[
    "ocr", "validation", "tamper", "face", "ovd", "template",
    "database", "graph", "crossdoc", "system",
]
Severity = Literal["info", "low", "medium", "high", "critical"]


class Region(BaseModel):
    """Normalised 0..1 against the rectified document image. Never pixels."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    w: float
    h: float
    document_id: str


class Signal(BaseModel):
    id: str
    code: str  # e.g. 'MRZ_CHECKDIGIT_DOB' -- see appendix A of the spec
    module: Module
    severity: Severity
    weight: float = 0.0  # contribution to risk; 0 for info-only
    detail: str  # a finished human sentence, written by the backend
    region: Optional[Region] = None  # absent for non-spatial signals (face, watchlist)
    heatmap_url: Optional[str] = None
    convergence_group: Optional[str] = None  # set when >=3 modules agree on the same region
