"""OCRReader: the unified MRZ + VIZ contract (BACKEND_BRIEF.md §6.2).

read(document) -> OCRResult runs MRZReader on the MRZ band and VizReader on
the remaining VIZ region, then cross-checks the two (§6.2 step 5): any
mismatch on birth_date, expiry_date, doc_number, surname, nationality, or sex
becomes a high-severity MRZ_VIZ_MISMATCH signal. An unrecoverable MRZ
checksum also becomes a signal here (CHECKSUM_UNRECOVERABLE), not just a
decode status buried in MRZReader's return value -- §1.3 requires this to
surface as a fraud signal at the point downstream consumers actually look.

OCRResult is an OCR-branch-internal aggregate (fields dict keyed
'mrz.<name>'/'viz.<name>', plus mrz_present/viz_field_count bookkeeping the
frontend's OcrEvent doesn't need) built out of the real app/contracts types
(Signal, ExtractedField, MrzInfo/MrzLine/MrzGroup) -- not a frontend contract
itself. The pipeline orchestrator flattens `fields` into the list the OcrEvent
wire type expects.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from app.contracts import ExtractedField, MrzGroup, MrzInfo, MrzLine, Region, Signal

from .mrz import decode as mrz_decode
from .mrz import detect as mrz_detect
from .mrz.infer import MRZReader, MrzReadResult
from .viz import VizField, VizReader, classify_fields

logger = logging.getLogger(__name__)

# name -> (mrz field name, viz field name) for the cross-check in §6.2 step 5.
_CROSSCHECK_FIELDS = [
    ("birth_date", "birth_date"),
    ("expiry_date", "expiry_date"),
    ("doc_number", "doc_number"),
    ("surname", "name"),  # VIZ only has a combined "name" field for now
    ("nationality", "nationality"),
    ("sex", "sex"),
]

_FIELD_LABELS = {
    "doc_number": "Document number",
    "surname": "Surname",
    "given_names": "Given names",
    "nationality": "Nationality",
    "sex": "Sex",
    "birth_date": "Date of birth",
    "expiry_date": "Expiry date",
    "name": "Name",
}


def _label_for(name: str) -> str:
    return _FIELD_LABELS.get(name, name.replace("_", " ").capitalize())


@dataclass
class OCRResult:
    fields: dict[str, ExtractedField] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    mrz: Optional[MrzInfo] = None
    confidence: float = 0.0
    mrz_present: bool = False
    viz_field_count: int = 0


def _normalize_for_compare(field_name: str, value: str) -> str:
    if field_name in ("birth_date", "expiry_date"):
        return value  # already ISO from both readers
    return value.strip().upper().replace("<", " ").strip()


def _dates_close(a: str, b: str) -> bool:
    try:
        return date.fromisoformat(a) == date.fromisoformat(b)
    except ValueError:
        return a == b


@dataclass
class CrossCheckRowResult:
    """One field's MRZ<->VIZ comparison outcome. Status is the frontend's
    CrossCheckStatus: 'match' | 'mismatch' | 'unreadable'."""

    field: str  # the mrz-side field name, e.g. 'surname'
    viz_field: str  # the viz-side field name, e.g. 'name'
    mrz_value: Optional[str]
    viz_value: Optional[str]
    status: str
    region: Optional[Region] = None


@dataclass
class CrossCheckOutcome:
    rows: list[CrossCheckRowResult] = field(default_factory=list)

    @property
    def mismatched_names(self) -> set[str]:
        out: set[str] = set()
        for row in self.rows:
            if row.status == "mismatch":
                out.add(row.field)
                out.add(row.viz_field)
        return out


def crosscheck_mrz_viz(
    mrz_result: MrzReadResult,
    viz_fields: list[VizField],
    *,
    fallback_region: Optional[Region] = None,
) -> CrossCheckOutcome:
    """Compare MRZ and VIZ values field by field (App §6.2 step 5 / Gate 1).

    A field present on only one side is 'unreadable', not silently skipped --
    an officer cannot cross-check a field that was never read on one side,
    and that is worth surfacing distinctly from a genuine mismatch.
    """
    rows: list[CrossCheckRowResult] = []
    if mrz_result.parsed is None:
        return CrossCheckOutcome(rows=rows)

    viz_by_name = {f.name: f for f in viz_fields}

    for mrz_name, viz_name in _CROSSCHECK_FIELDS:
        mrz_field = mrz_result.fields.get(mrz_name)
        viz_field = viz_by_name.get(viz_name)

        if mrz_field is None or viz_field is None:
            rows.append(CrossCheckRowResult(
                field=mrz_name, viz_field=viz_name,
                mrz_value=mrz_field.value if mrz_field else None,
                viz_value=viz_field.value if viz_field else None,
                status="unreadable", region=fallback_region,
            ))
            continue

        mrz_value = _normalize_for_compare(mrz_name, mrz_field.value)
        viz_value = _normalize_for_compare(mrz_name, viz_field.value)

        if mrz_name in ("birth_date", "expiry_date"):
            matches = _dates_close(mrz_value, viz_value)
        elif mrz_name == "surname":
            # VIZ "name" is a combined surname+given-names line; a substring
            # match is the honest bar for a heuristic classifier.
            matches = mrz_value in viz_value or viz_value in mrz_value
        else:
            matches = mrz_value == viz_value

        rows.append(CrossCheckRowResult(
            field=mrz_name, viz_field=viz_name,
            mrz_value=mrz_field.value, viz_value=viz_field.value,
            status="match" if matches else "mismatch",
            region=viz_field.region or fallback_region,
        ))

    return CrossCheckOutcome(rows=rows)


class OCRReader:
    """Ties MRZReader and VizReader together into the OCR branch contract."""

    def __init__(self, mrz_reader: MRZReader, viz_reader: VizReader) -> None:
        self.mrz_reader = mrz_reader
        self.viz_reader = viz_reader

    def read(
        self,
        document: np.ndarray,
        *,
        document_id: str = "document",
        stub_mrz_ground_truth: Optional[list[str]] = None,
    ) -> OCRResult:
        mrz_result = self.mrz_reader.read(document, stub_ground_truth=stub_mrz_ground_truth)

        viz_boxes = self.viz_reader.read_boxes(
            document, mrz_band=mrz_result.band, document_id=document_id
        )
        viz_fields = classify_fields(viz_boxes)

        mismatched_names: set[str] = set()
        signals: list[Signal] = list(self._mrz_signals(mrz_result, document_id))
        crosscheck_signals, mismatched_names = self._crosscheck(mrz_result, viz_fields, document_id)
        signals.extend(crosscheck_signals)

        fields: dict[str, ExtractedField] = {}
        for name, mf in mrz_result.fields.items():
            fields[f"mrz.{name}"] = ExtractedField(
                key=name, label=_label_for(name), value=mf.value,
                confidence=mf.confidence, source="MRZ",
                mismatch=name in mismatched_names or None,
            )
        for vf in viz_fields:
            fields[f"viz.{vf.name}"] = ExtractedField(
                key=vf.name, label=_label_for(vf.name), value=vf.value,
                confidence=vf.confidence, source="VIZ",
                mismatch=vf.name in mismatched_names or None,
            )

        confidence = self._overall_confidence(mrz_result, viz_fields, signals)
        mrz_info = self._build_mrz_info(mrz_result, signals)

        return OCRResult(
            fields=fields,
            signals=signals,
            mrz=mrz_info,
            confidence=confidence,
            mrz_present=mrz_result.band is not None,
            viz_field_count=len(viz_fields),
        )

    @staticmethod
    def _region_of(
        band: Optional[mrz_detect.MrzBand], document_id: str
    ) -> Optional[Region]:
        if band is None:
            return None
        x, y, w, h = band.region
        return Region(x=x, y=y, w=w, h=h, document_id=document_id)

    def _mrz_signals(self, mrz_result: MrzReadResult, document_id: str) -> list[Signal]:
        signals: list[Signal] = []
        region = self._region_of(mrz_result.band, document_id)

        for sig in mrz_result.signals:
            signals.append(Signal(
                id=str(uuid.uuid4()), code=sig.code, module="ocr",
                severity=sig.severity, detail=sig.detail, region=region,
            ))

        if mrz_result.status is mrz_decode.DecodeStatus.UNRECOVERABLE and not any(
            s.code == "MRZ_CHECKDIGIT_UNRECOVERABLE" for s in mrz_result.signals
        ):
            # The band was found and decoded, but the overall MRZ (e.g. the
            # composite check digit) doesn't validate even though no single
            # field-level group was individually unrecoverable -- still a
            # fraud signal, not silence.
            signals.append(Signal(
                id=str(uuid.uuid4()), code="CHECKSUM_UNRECOVERABLE", module="ocr",
                severity="high", weight=30.0,
                detail=(
                    "The machine-readable zone's checksum does not validate under any "
                    "hypothesis the decoder considered; this is treated as a potential "
                    "tamper indicator, not an ordinary OCR error."
                ),
                region=region,
            ))
        elif any(s.code == "MRZ_CHECKDIGIT_UNRECOVERABLE" for s in mrz_result.signals):
            # Surface the fraud-signal code the brief names explicitly (§1.3),
            # in addition to the field-level detail MRZReader already emitted.
            signals.append(Signal(
                id=str(uuid.uuid4()), code="CHECKSUM_UNRECOVERABLE", module="ocr",
                severity="high", weight=30.0,
                detail=(
                    "One or more machine-readable zone fields have a checksum that could "
                    "not be recovered by the checksum-constrained decoder; this is treated "
                    "as a potential tamper indicator, not an ordinary OCR error."
                ),
                region=region,
            ))

        return signals

    def _crosscheck(
        self, mrz_result: MrzReadResult, viz_fields: list[VizField], document_id: str
    ) -> tuple[list[Signal], set[str]]:
        region = self._region_of(mrz_result.band, document_id)
        outcome = crosscheck_mrz_viz(mrz_result, viz_fields, fallback_region=region)
        signals = [
            Signal(
                id=str(uuid.uuid4()), code="MRZ_VIZ_MISMATCH", module="ocr",
                severity="high", weight=30.0,
                detail=(
                    f"The {row.field.replace('_', ' ')} read from the machine-readable "
                    f"zone ({row.mrz_value!r}) does not match the value printed in "
                    f"the visual inspection zone ({row.viz_value!r})."
                ),
                region=row.region,
            )
            for row in outcome.rows if row.status == "mismatch"
        ]
        return signals, outcome.mismatched_names

    @staticmethod
    def _build_mrz_info(
        mrz_result: MrzReadResult, signals: list[Signal]
    ) -> Optional[MrzInfo]:
        parsed = mrz_result.parsed
        if parsed is None or not parsed.lines:
            return None

        # Link an invalid group to the signal that reported it, matching on
        # the group name embedded in MRZReader's detail text (see infer.py's
        # _to_result: "MRZ {name.replace('_', ' ')} check digit...").
        signal_by_group_name = {
            s.code: s.id for s in signals if s.code == "CHECKSUM_UNRECOVERABLE"
        }

        lines: list[MrzLine] = []
        for idx, text in enumerate(parsed.lines):
            groups = [
                MrzGroup(
                    name=g.name, start=g.start, end=g.end,
                    check_digit_index=g.check_digit_index, valid=g.valid,
                    expected=None if g.valid else g.expected,
                    read=None if g.valid else g.read,
                    signal_id=None if g.valid else signal_by_group_name.get("CHECKSUM_UNRECOVERABLE"),
                )
                for g in parsed.checks
                if g.line == idx
            ]
            lines.append(MrzLine(text=text, groups=groups))

        status = "VERIFIED" if mrz_result.status is mrz_decode.DecodeStatus.VERIFIED else "UNRECOVERABLE"
        return MrzInfo(format=parsed.format.value, lines=lines, status=status)

    @staticmethod
    def _overall_confidence(
        mrz_result: MrzReadResult, viz_fields: list[VizField], signals: list[Signal]
    ) -> float:
        if mrz_result.parsed is None:
            return 0.0
        mrz_conf = sum(f.confidence for f in mrz_result.fields.values()) / max(len(mrz_result.fields), 1)
        viz_conf = sum(f.confidence for f in viz_fields) / max(len(viz_fields), 1) if viz_fields else 0.5
        base = 0.6 * mrz_conf + 0.4 * viz_conf
        penalty = 0.15 * sum(1 for s in signals if s.severity in ("high", "critical"))
        return max(0.0, min(1.0, base - penalty))
