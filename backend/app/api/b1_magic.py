"""Magic-byte content sniffing for artefact uploads (B1b spec: "Validate by
content sniffing, not by extension or the client-supplied content type").

No third-party sniffing library is in requirements.txt; the five accepted
types have short, well-known signatures, so a small explicit table is more
auditable than adding a dependency for this.
"""
from __future__ import annotations

from typing import Optional

ACCEPTED_IMAGE_PDF = ("image/jpeg", "image/png", "application/pdf")
ACCEPTED_VIDEO = ("video/mp4", "video/webm")
ACCEPTED_TYPES = ACCEPTED_IMAGE_PDF + ACCEPTED_VIDEO


def sniff_content_type(head: bytes) -> Optional[str]:
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "video/mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    return None


def is_video(content_type: str) -> bool:
    return content_type in ACCEPTED_VIDEO
