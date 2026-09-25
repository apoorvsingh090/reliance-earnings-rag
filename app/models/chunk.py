"""Chunk contract — the retrieval unit for all unstructured text.

Every chunk carries page/section provenance. Financial numbers that already
live in the structured database (XBRL tables, annexure data tables) are
deliberately NOT chunked — vectors hold narrative, not authoritative figures.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

from app.models.document import ReportType

ChunkType = Literal["text", "table", "descriptor"]


class Chunk(BaseModel):
    chunk_id: str
    document_id: str = Field(description="SHA256 of the source document")
    page_number: int | None = Field(description="1-indexed PDF page; None for HTML docs")
    printed_label: str = ""
    section: str = ""
    chunk_type: ChunkType = "text"
    text: str

    company: str = "Reliance Industries Limited"
    ticker: str = "RELIANCE"
    financial_year: str = ""
    quarter: str = ""
    period: str = ""
    document_type: str = ""
    report_type: ReportType = ReportType.NOT_APPLICABLE
    source_location: str = ""

    @staticmethod
    def make_id(document_id: str, page: int | None, section: str, idx: int, text: str) -> str:
        key = f"{document_id}|{page}|{section}|{idx}|{hashlib.sha256(text.encode()).hexdigest()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
