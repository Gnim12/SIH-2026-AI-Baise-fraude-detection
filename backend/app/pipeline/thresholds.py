"""Thresholds, loaded once at import and exposed read-only.

Only `mrz_viz_field_tolerance` is wired into a stage in B1 (gate1.py); the
other five F10 thresholds are outside this milestone's scope (no stage in
the B1 registry consumes them yet) and are not stubbed here to avoid a
threshold that looks configured but does nothing -- flagged in the B1
report as follow-up work, not silently invented.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    mrz_viz_field_tolerance: float = 0.92


DEFAULT_THRESHOLDS = Thresholds()
