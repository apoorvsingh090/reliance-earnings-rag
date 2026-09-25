"""Milestone 7 tests: user-upload registration (isolated tmp dirs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.ingestion.pipeline import register_local_file
from app.ingestion.validate import ValidationError
from app.llm.router import route
from app.models.document import DocumentSpec


def _isolate(monkeypatch, tmp_path: Path):
    settings = get_settings()
    monkeypatch.setattr(settings, "raw_dir", tmp_path / "raw" / "reliance")
    monkeypatch.setattr(settings, "processed_dir", tmp_path / "processed")
    monkeypatch.setattr(settings, "extracted_dir", tmp_path / "extracted")
    return settings


def _spec(**overrides) -> DocumentSpec:
    base = dict(company="Reliance Industries Limited", ticker="RELIANCE",
                financial_year="FY27", quarter="Q1", period="Q1 FY27",
                document_type="media_release", report_type="consolidated",
                source_url="user-upload://test.pdf", filename="TEST_upload.pdf",
                publication_date="2026-07-17")
    base.update(overrides)
    return DocumentSpec(**base)


def _pdf_bytes(n: int = 50) -> bytes:
    return b"%PDF-1.4\n" + b"Reliance earnings commentary line.\n" * n


def test_upload_registers_and_is_idempotent(monkeypatch, tmp_path: Path):
    _isolate(monkeypatch, tmp_path)
    src = tmp_path / "up.pdf"
    src.write_bytes(_pdf_bytes())
    rec1, created1 = register_local_file(_spec(), src)
    assert created1 and rec1.file_size > 100 and rec1.report_type.value == "consolidated"
    rec2, created2 = register_local_file(_spec(), src)
    assert not created2 and rec2.document_hash == rec1.document_hash


def test_upload_html_kind(monkeypatch, tmp_path: Path):
    _isolate(monkeypatch, tmp_path)
    src = tmp_path / "up.html"
    src.write_bytes(b"<html><body>" + b"notes " * 60 + b"</body></html>")
    rec, created = register_local_file(_spec(document_type="xbrl",
                                             filename="TEST_up.html",
                                             source_url="user-upload://up.html"), src)
    assert created and rec.filename == "TEST_up.html"


def test_upload_rejects_broken_files(monkeypatch, tmp_path: Path):
    _isolate(monkeypatch, tmp_path)
    tiny = tmp_path / "tiny.pdf"
    tiny.write_bytes(b"%PDF-1.4 short")
    with pytest.raises(ValidationError):
        register_local_file(_spec(filename="tiny.pdf"), tiny)
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"<html>not a pdf" + b"x" * 5000)
    with pytest.raises(ValidationError):
        register_local_file(_spec(filename="fake.pdf"), fake)


def test_upload_filename_cannot_escape_raw_dir(monkeypatch, tmp_path: Path):
    settings = _isolate(monkeypatch, tmp_path)
    src = tmp_path / "evil.pdf"
    src.write_bytes(_pdf_bytes())
    rec, _ = register_local_file(_spec(filename="../../evil.pdf"), src)
    assert Path(rec.local_path).name == "evil.pdf"
    assert (settings.raw_dir / "evil.pdf").exists()


def test_ui_basis_prefix_convention_routes_standalone():
    assert route("On a standalone basis: What was revenue in Q1 FY27?").report_type == "standalone"
    assert route("What was revenue in Q1 FY27?").report_type == "consolidated"
