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
    # Mean per-character network probability over the decoded MRZ lines below
    # which MRZ_LOW_CONFIDENCE is raised.
    mrz_low_confidence_mean: float = 0.6
    # A recovery is suspect when the network's confidence in its raw reading
    # exceeds its confidence in the checksum-resolved reading by more than
    # this. A confident network that the check digit contradicts strongly is
    # either wrong, or the band was misdetected and the whole line is unreliable.
    mrz_recovery_confidence_margin: float = 0.30


DEFAULT_THRESHOLDS = Thresholds()
