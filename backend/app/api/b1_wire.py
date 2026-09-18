"""Wire (response) models for the session/artefact endpoints of the B1b HTTP
surface -- app/api/schemas.py owns the pipeline/result contracts (mirrors the
frontend's lib/pipeline + findings types exactly); these are new resources
that contract doesn't cover, so they get their own small camelCase models
here rather than bare dicts, for the same OpenAPI/mypy benefit.
"""
from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


def _camel_model() -> ConfigDict:
    return ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ArtefactOut(BaseModel):
    model_config = _camel_model()
    kind: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    captured_at: datetime.datetime
    duration_ms: Optional[float] = None
    frame_count: Optional[int] = None
    angular_coverage_deg: Optional[float] = None


class SessionOut(BaseModel):
    model_config = _camel_model()
    session_id: str
    created_at: datetime.datetime
    document_class: Optional[str] = None
    status: str
    artefacts: list[ArtefactOut]


class ArtefactUploadOut(BaseModel):
    model_config = _camel_model()
    artefact: ArtefactOut
    notice: Optional[str] = None
