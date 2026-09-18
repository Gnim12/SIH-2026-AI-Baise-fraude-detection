"""BACKEND_BRIEF.md §4 / FRONTEND_BRIEF.md §3: the ScreeningEvent discriminated
union on `stage`, matching the frontend's event union exactly."""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from .session import DocType, ExtractedField, FaceResult, GraphResult, MrzInfo, SystemBand, ViewKey
from .signals import Signal


class ReceivedEvent(BaseModel):
    stage: Literal["received"] = "received"
    session_id: str
    lane_id: str
    officer_id: str


class QualityEvent(BaseModel):
    stage: Literal["quality"] = "quality"
    document_id: str
    ok: bool
    dpi: Optional[float] = None
    reason: Optional[str] = None
    hint: Optional[str] = None


class ClassifiedEvent(BaseModel):
    stage: Literal["classified"] = "classified"
    document_id: str
    type: DocType
    country: str
    version: str
    confidence: float
    image_url: str


class OcrEvent(BaseModel):
    stage: Literal["ocr"] = "ocr"
    document_id: str
    fields: list[ExtractedField]
    mrz: Optional[MrzInfo] = None
    signals: list[Signal]


class FaceEvent(BaseModel):
    stage: Literal["face"] = "face"
    face: FaceResult
    signals: list[Signal]


class DatabaseEvent(BaseModel):
    stage: Literal["database"] = "database"
    graph: GraphResult
    signals: list[Signal]


class ForensicsEvent(BaseModel):
    stage: Literal["forensics"] = "forensics"
    document_id: str
    views: dict[ViewKey, str]
    signals: list[Signal]


class CrossdocEvent(BaseModel):
    stage: Literal["crossdoc"] = "crossdoc"
    signals: list[Signal]


class DecisionEvent(BaseModel):
    stage: Literal["decision"] = "decision"
    band: SystemBand
    risk: Optional[float] = None
    confidence: float
    abstained: bool
    coverage_flags: list[str]
    timing_ms: dict[str, float]


class ErrorEvent(BaseModel):
    stage: Literal["error"] = "error"
    message: str


ScreeningEvent = Annotated[
    Union[
        ReceivedEvent,
        QualityEvent,
        ClassifiedEvent,
        OcrEvent,
        FaceEvent,
        DatabaseEvent,
        ForensicsEvent,
        CrossdocEvent,
        DecisionEvent,
        ErrorEvent,
    ],
    Field(discriminator="stage"),
]
