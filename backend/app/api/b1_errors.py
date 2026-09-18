"""Uniform error envelope for the B1b HTTP surface:
`{"error": {"code": ..., "message": ..., "detail": {...}}}`.

Every route in app/api/b1_*.py raises `ApiError` instead of a bare
`HTTPException` so the shape is consistent; `register` installs the exception
handler that renders it. Messages state what the caller can do next; no
traceback, internal path, or model internal ever reaches the body -- the
traceback is logged server-side with whatever session/run id is available
(see each raise site).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(
        self, status_code: int, code: str, message: str, detail: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail or {}


def _envelope(code: str, message: str, detail: dict[str, Any]) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "detail": detail}}


def register(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.detail),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Stage failures are reported as stage results, not API errors (see
        # the B1b spec's §3); anything reaching here is a genuine bug in the
        # HTTP layer itself. Log the real traceback, tell the caller nothing
        # more than "something went wrong".
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_envelope(
                "internal_error",
                "Something went wrong processing this request. Try again; if it persists, contact support.",
                {},
            ),
        )
