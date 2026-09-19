"""Group signals into findings by hypothesis (B1 grouping rules only --
MRZ and cross-check signals are all that exist until Wave 2 ships real
modules, so most findings are single-signal here and that is correct, not a
shortcut).

Grouping rules:
- MRZ_VIZ_MISMATCH_* (identity-field disagreement) -> one finding.
- Checksum failures (MRZ_CHECKSUM_FAIL_LINE1/2, MRZ_CHECKSUM_FAIL_COMPOSITE,
  MRZ_DECODE_UNRECOVERABLE)
  -> one 'MRZ integrity failure' finding.
- Unreadable fields (MRZ_BAND_NOT_FOUND, MRZ_VIZ_FIELD_UNREADABLE_*,
  VIZ_FIELD_NOT_READABLE) -> one 'Document not fully readable' finding,
  severity moderate.
- Everything else -> its own single-signal finding.

Region joining: a later signal whose region overlaps an existing finding's
region joins that finding instead of starting a parallel one, on IoU > 0 --
crude but correct, and the mechanism a B3 Wave-2 signal implicating the same
photo region needs to converge into an existing MRZ/VIZ finding rather than
duplicate it. Not retrofitted later; this is the actual payoff of writing
the grouper this way now.
"""
from __future__ import annotations

import uuid

from app.api.schemas import Finding, Region, Severity, Signal
from app.findings.signals import is_checksum_failure, is_crosscheck_mismatch, is_unreadable


def _iou(a: Region, b: Region) -> float:
    ax2, ay2, bx2, by2 = a.x + a.w, a.y + a.h, b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _region_overlaps(existing: Region | None, region: Region | None) -> bool:
    if existing is None or region is None:
        return False
    return _iou(existing, region) > 0.0


def _try_join_by_region(findings: list[Finding], signal: Signal) -> bool:
    if signal.region is None:
        return False
    for finding in findings:
        if _region_overlaps(finding.region, signal.region):
            finding.signals.append(signal)
            if signal.modality not in finding.converged_modalities:
                finding.converged_modalities.append(signal.modality)
            return True
    return False


def _new_finding(*, title: str, hypothesis: str, severity: Severity, signal: Signal) -> Finding:
    return Finding(
        finding_id=str(uuid.uuid4()), title=title, hypothesis=hypothesis, severity=severity,
        signals=[signal], converged_modalities=[signal.modality], region=signal.region,
    )


def group_signals(signals: list[Signal]) -> list[Finding]:
    findings: list[Finding] = []
    mismatch_finding: Finding | None = None
    checksum_finding: Finding | None = None
    unreadable_finding: Finding | None = None

    for signal in signals:
        if signal.signal_id in ("STAGE_TIMEOUT", "STAGE_EXCEPTION"):
            findings.append(_new_finding(
                title="Stage did not complete", hypothesis=signal.detail,
                severity=Severity.INFORMATIONAL, signal=signal,
            ))
            continue

        if _try_join_by_region(findings, signal):
            continue

        if is_crosscheck_mismatch(signal.signal_id):
            if mismatch_finding is None:
                mismatch_finding = _new_finding(
                    title="Printed and machine-readable data disagree",
                    hypothesis="One or more fields printed in the visual inspection zone do not match "
                    "the machine-readable zone. Capture quality (glare, blur, fold) causes this far more "
                    "often than a genuine document defect; recapture and re-run first.",
                    severity=Severity.HIGH, signal=signal,
                )
                findings.append(mismatch_finding)
            else:
                mismatch_finding.signals.append(signal)
                if signal.modality not in mismatch_finding.converged_modalities:
                    mismatch_finding.converged_modalities.append(signal.modality)
            continue

        if is_checksum_failure(signal.signal_id):
            if checksum_finding is None:
                checksum_finding = _new_finding(
                    title="MRZ integrity failure",
                    hypothesis="A machine-readable zone check digit does not validate against any "
                    "hypothesis the checksum-constrained decoder considered.",
                    severity=Severity.HIGH, signal=signal,
                )
                findings.append(checksum_finding)
            else:
                checksum_finding.signals.append(signal)
            continue

        if is_unreadable(signal.signal_id):
            if unreadable_finding is None:
                unreadable_finding = _new_finding(
                    title="Document not fully readable",
                    hypothesis="One or more fields could not be read on the machine-readable and/or "
                    "visual inspection zone, so full cross-checking could not be completed.",
                    severity=Severity.MODERATE, signal=signal,
                )
                findings.append(unreadable_finding)
            else:
                unreadable_finding.signals.append(signal)
            continue

        findings.append(_new_finding(
            title=signal.label, hypothesis=signal.detail, severity=Severity.INFORMATIONAL, signal=signal,
        ))

    return findings
