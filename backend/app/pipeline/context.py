"""Per-document context threaded through every wave-1/wave-2 branch."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DocumentInput:
    document_id: str
    doc_type: str  # frontend DocType, best-effort until GATE 2 classifies for real
    original_bytes: bytes  # untouched -- forensics reads THIS, never the rectified image (§5.2)
    rectified: np.ndarray  # BGR image array, ready for OCR/face/template
    # Dev/test-only: BACKEND_BRIEF.md §1.3 says no trained CRNN checkpoint
    # exists yet, so MRZReader can only run in stub mode, which fabricates
    # logprobs from a *known* ground-truth string rather than actually
    # reading the image. A real, unlabelled capture therefore cannot have its
    # MRZ decoded at all until the CRNN is trained -- branch_ocr degrades
    # gracefully (MRZ_MODEL_UNAVAILABLE) rather than crash when this is None.
    # Tests/demos that want to see a genuine checksum-constrained decode
    # against a real image set this to the known MRZ lines that were
    # rendered into that image.
    mrz_ground_truth: Optional[list[str]] = None


@dataclass
class PipelineContext:
    session_id: str
    lane_id: str
    officer_id: str
    model_versions: dict[str, str] = field(default_factory=dict)
    # BACKEND_BRIEF.md §1.3: no trained CRNN weights exist yet. Set False only
    # via config, never silently -- every branch that reads this must log it,
    # not just the OCR module in isolation (this task's brief, item 3).
    mrz_require_trained_weights: bool = True
