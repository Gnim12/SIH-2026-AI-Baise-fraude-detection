"""Pydantic v2 contracts mirroring FRONTEND_BRIEF.md §3 (`src/types/screening.ts`)
field for field. See BACKEND_BRIEF.md §4.

Field names here are idiomatic Python snake_case (`document_id`, not
`documentId`) -- that is what "snake_case on the wire" in BACKEND_BRIEF.md §4
means for these model definitions themselves, and it's what the round-trip
test in tests/contracts/test_roundtrip.py exercises directly.

The already-built React frontend (FRONTEND_BRIEF.md) consumes camelCase JSON
verbatim with no case conversion of its own (see src/api/socket.ts: events are
`JSON.parse`d straight into the `ScreeningEvent` union). So the API boundary
(app/api/stream.py, app/api/screening.py, app/api/history.py) converts these
snake_case models to camelCase on the way out with `wire.to_wire_json` --
never bakes camelCase aliasing into the contracts themselves. Internal code,
persistence, and the audit chain all stay snake_case; only the last hop before
the socket/response changes case.
"""
from .events import (
    ClassifiedEvent,
    CrossdocEvent,
    DatabaseEvent,
    DecisionEvent,
    ErrorEvent,
    FaceEvent,
    ForensicsEvent,
    OcrEvent,
    QualityEvent,
    ReceivedEvent,
    ScreeningEvent,
)
from .session import (
    Decision,
    DocType,
    Encounter,
    ExtractedField,
    FaceResult,
    GraphResult,
    MrzGroup,
    MrzInfo,
    MrzLine,
    OfficerDecision,
    ScreenedDocument,
    ScreeningSession,
    SystemBand,
)
from .signals import Region, Signal

__all__ = [
    "Region",
    "Signal",
    "Decision",
    "DocType",
    "SystemBand",
    "ExtractedField",
    "MrzGroup",
    "MrzLine",
    "MrzInfo",
    "ScreenedDocument",
    "FaceResult",
    "Encounter",
    "GraphResult",
    "OfficerDecision",
    "ScreeningSession",
    "ReceivedEvent",
    "QualityEvent",
    "ClassifiedEvent",
    "OcrEvent",
    "FaceEvent",
    "DatabaseEvent",
    "ForensicsEvent",
    "CrossdocEvent",
    "DecisionEvent",
    "ErrorEvent",
    "ScreeningEvent",
]
