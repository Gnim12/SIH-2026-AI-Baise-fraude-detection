"""Round-trip test for app/contracts/: confirms the Python-side models use
snake_case fields that survive a JSON round trip unchanged (BACKEND_BRIEF.md
§4), and separately that the wire converter (app/contracts/wire.py) produces
the camelCase shape the already-built frontend's types/screening.ts expects.
"""
from __future__ import annotations

import json

from app.contracts import (
    ExtractedField,
    MrzGroup,
    MrzInfo,
    MrzLine,
    OcrEvent,
    Region,
    ScreenedDocument,
    ScreeningSession,
    Signal,
)
from app.contracts.wire import to_wire


def _sample_session() -> ScreeningSession:
    region = Region(x=0.1, y=0.2, w=0.3, h=0.4, document_id="doc-1")
    signal = Signal(
        id="sig-1", code="MRZ_CHECKDIGIT_DOB", module="ocr", severity="high",
        weight=30.0, detail="DOB check digit expected 4, read 7.", region=region,
        convergence_group=None,
    )
    mrz_group = MrzGroup(
        name="birth_date", start=13, end=19, check_digit_index=19,
        valid=False, expected="4", read="7", signal_id="sig-1",
    )
    mrz = MrzInfo(
        format="TD3",
        lines=[MrzLine(text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", groups=[]),
               MrzLine(text="L898902C36UTO7408122F1204159ZE184226B<<<<<10", groups=[mrz_group])],
        status="UNRECOVERABLE",
    )
    field = ExtractedField(key="birth_date", label="Date of birth", value="1974-08-12",
                            confidence=0.62, source="MRZ", mismatch=True)
    doc = ScreenedDocument(id="doc-1", type="PASSPORT", country="UTO", version="v1",
                            image_url="/assets/doc-1.jpg", views={"rgb": "/assets/doc-1.jpg"},
                            mrz=mrz, fields=[field], risk=78.0)
    return ScreeningSession(
        session_id="sess-1", lane_id="IGI-T3-LANE-07", officer_id="OFF-2291",
        started_at="2026-09-01T10:00:00Z", band="HOLD", risk=78.0, confidence=0.91,
        abstained=False, documents=[doc], signals=[signal], face=None, graph=None,
        cross_document_signals=[], coverage_flags=["no_biometric"],
        timing_ms={"total": 1620.0}, sealed=False,
        model_versions={"mrz_crnn": "1.3.0", "rapidocr_det": "v5"},
    )


def test_session_roundtrips_through_json_with_snake_case_fields():
    session = _sample_session()
    raw = session.model_dump_json()
    payload = json.loads(raw)

    # snake_case on the wire (Python-side serialization, not the frontend hop).
    assert "session_id" in payload
    assert "lane_id" in payload
    assert "documents" in payload
    assert payload["documents"][0]["image_url"] == "/assets/doc-1.jpg"
    assert payload["signals"][0]["region"]["document_id"] == "doc-1"
    assert payload["documents"][0]["mrz"]["lines"][1]["groups"][0]["check_digit_index"] == 19
    assert payload["model_versions"] == {"mrz_crnn": "1.3.0", "rapidocr_det": "v5"}

    restored = ScreeningSession.model_validate_json(raw)
    assert restored == session


def test_ocr_event_discriminates_on_stage():
    event = OcrEvent(document_id="doc-1", fields=[], mrz=None, signals=[])
    raw = event.model_dump_json()
    payload = json.loads(raw)
    assert payload["stage"] == "ocr"

    from app.contracts import ScreeningEvent
    from pydantic import TypeAdapter

    adapter = TypeAdapter(ScreeningEvent)
    restored = adapter.validate_json(raw)
    assert isinstance(restored, OcrEvent)


def test_wire_converter_produces_camel_case_matching_frontend_types():
    session = _sample_session()
    wire = to_wire(session)

    # matches frontend/src/types/screening.ts ScreeningSession field names
    assert wire["sessionId"] == "sess-1"
    assert wire["laneId"] == "IGI-T3-LANE-07"
    assert wire["documents"][0]["imageUrl"] == "/assets/doc-1.jpg"
    assert wire["signals"][0]["region"]["documentId"] == "doc-1"
    assert wire["documents"][0]["mrz"]["lines"][1]["groups"][0]["checkDigitIndex"] == 19
    assert wire["documents"][0]["mrz"]["lines"][1]["groups"][0]["signalId"] == "sig-1"
    assert wire["coverageFlags"] == ["no_biometric"]
    # snake_case keys must not leak through
    assert "session_id" not in wire
    assert "check_digit_index" not in json.dumps(wire)
