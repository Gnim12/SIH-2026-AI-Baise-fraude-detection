"""BACKEND_BRIEF.md §5.3: failure isolation is the point. A branch that times
out or raises degrades the session's result for that branch; it never crashes
the session and it never silently passes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Degraded:
    branch: str
    reason: str  # 'timeout' | the exception message
