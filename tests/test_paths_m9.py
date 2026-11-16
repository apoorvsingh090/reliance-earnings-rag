"""Milestone 9 tests: registry paths stay portable across OSes."""

from __future__ import annotations

from pathlib import Path

from app.config import PROJECT_ROOT
from app.extraction.pipeline import _resolve_raw_path as resolve_extraction
from app.models.document import DocumentRecord, DocumentSpec
from app.retrieval.pipeline import _resolve_raw_path as resolve_retrieval


def _spec() -> DocumentSpec:
    return DocumentSpec(
        company="Reliance Industries Limited", ticker="RELIANCE",
        financial_year="FY27", quarter="Q1", period="Q1 FY27",
        document_type="xbrl", report_type="consolidated",
        source_url="user-upload://x.html", filename="X.html",
        publication_date="2026-07-17")


def test_from_spec_normalizes_windows_separators():
    rec = DocumentRecord.from_spec(
        _spec(), document_hash="ab" * 32, file_size=50000,
        local_path="data\\raw\\reliance\\X.html")
    assert rec.local_path == "data/raw/reliance/X.html"
    assert Path(rec.local_path).name == "X.html"


def test_resolve_raw_path_accepts_legacy_backslash_records():
    legacy = "data\\raw\\reliance\\X.html"
    expected = PROJECT_ROOT / "data/raw/reliance/X.html"
    assert resolve_extraction(legacy) == expected
    assert resolve_retrieval(legacy) == expected


def test_resolve_raw_path_prefers_existing_absolute():
    p = PROJECT_ROOT / "pytest.ini"
    assert resolve_extraction(str(p)) == p
