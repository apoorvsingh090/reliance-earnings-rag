"""Document metadata model — the contract every ingested file must satisfy.

Reporting basis (report_type) must come from the explicit document spec,
never inferred from a filename. The model enforces this by requiring the
field on every record.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    FINANCIAL_RESULTS = "financial_results"
    XBRL = "xbrl"
    INVESTOR_PRESENTATION = "investor_presentation"
    MEDIA_RELEASE = "media_release"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    ANNUAL_REPORT = "annual_report"
    OTHER = "other"


class ReportType(str, Enum):
    CONSOLIDATED = "consolidated"
    STANDALONE = "standalone"
    NOT_APPLICABLE = "not_applicable"


class DocumentSpec(BaseModel):
    """Static, human-curated description of a source document (input)."""

    company: str = "Reliance Industries Limited"
    ticker: str = "RELIANCE"
    financial_year: str
    quarter: str
    period: str  # e.g. "Q1 FY27"
    document_type: DocumentType
    report_type: ReportType
    source_url: str
    filename: str
    publication_date: str  # YYYY-MM-DD

    @field_validator("publication_date")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v


class DocumentRecord(BaseModel):
    """A downloaded + validated document stored on disk (output)."""

    company: str
    ticker: str
    financial_year: str
    quarter: str
    period: str
    document_type: DocumentType
    report_type: ReportType
    source_url: str
    filename: str
    publication_date: str

    # Provenance added by the pipeline
    document_hash: str = Field(description="SHA256 of the stored file bytes")
    file_size: int = Field(ge=1, description="Bytes on disk")
    local_path: str = Field(description="Path relative to project root")
    ingestion_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def manifest_entry(self) -> dict:
        return {
            "filename": self.filename,
            "document_type": self.document_type.value,
            "quarter": self.quarter,
            "period": self.period,
            "reporting_basis": self.report_type.value,
            "source_url": self.source_url,
            "sha256": self.document_hash,
            "file_size": self.file_size,
            "publication_date": self.publication_date,
            "company": self.company,
            "ticker": self.ticker,
        }

    @classmethod
    def from_spec(
        cls, spec: DocumentSpec, document_hash: str, file_size: int, local_path: Path | str
    ) -> "DocumentRecord":
        return cls(
            **spec.model_dump(),
            document_hash=document_hash,
            file_size=file_size,
            local_path=str(local_path),
        )
