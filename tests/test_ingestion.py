"""Milestone 1 tests: specs, hashing, validation, registry idempotency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion import documents as docspecs
from app.ingestion.hashing import sha256_bytes, sha256_file
from app.ingestion.registry import DocumentRegistry, write_manifest
from app.ingestion.validate import ValidationError, validate_file
from app.models.document import DocumentRecord


def test_seed_specs_are_explicit_and_isolated():
    specs = docspecs.Q1FY27_DOCUMENTS
    assert len(specs) == 3
    by_file = {s.filename: s for s in specs}
    # Standalone vs consolidated must be explicit and distinct.
    assert by_file["NSE_Q1FY27_standalone_xbrl.html"].report_type.value == "standalone"
    assert by_file["NSE_Q1FY27_consolidated_xbrl.html"].report_type.value == "consolidated"
    assert (
        by_file["NSE_Q1FY27_standalone_xbrl.html"].source_url
        != by_file["NSE_Q1FY27_consolidated_xbrl.html"].source_url
    )
    # Media release is explicitly consolidated (not inferred from filename).
    assert by_file["RIL_Q1FY27_media_release.pdf"].report_type.value == "consolidated"
    assert by_file["RIL_Q1FY27_media_release.pdf"].document_type.value == "media_release"
    # All Q1 FY27, Reliance.
    for s in specs:
        assert s.ticker == "RELIANCE"
        assert s.period == "Q1 FY27"
        assert s.quarter == "Q1"
        assert s.financial_year == "FY27"


def test_manifest_entry_has_required_keys():
    spec = docspecs.Q1FY27_DOCUMENTS[1]  # consolidated XBRL
    rec = DocumentRecord.from_spec(spec, document_hash="ab" * 32, file_size=12345, local_path="data/x")
    entry = rec.manifest_entry()
    for key in (
        "filename", "document_type", "quarter", "reporting_basis",
        "source_url", "sha256", "file_size",
    ):
        assert key in entry, f"missing manifest key: {key}"
    assert entry["reporting_basis"] == "consolidated"


def test_registry_is_idempotent_by_hash(tmp_path: Path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    spec = docspecs.Q1FY27_DOCUMENTS[0]
    rec = DocumentRecord.from_spec(spec, document_hash="aa" * 32, file_size=50000, local_path="data/y")
    stored1, created1 = reg.upsert(rec)
    stored2, created2 = reg.upsert(rec)  # same bytes ingested twice
    assert created1 is True
    assert created2 is False
    assert len(reg) == 1
    assert stored1.document_hash == stored2.document_hash
    # Same URL but different bytes (re-filed document) => new record, hash is identity.
    rec2 = DocumentRecord.from_spec(spec, document_hash="bb" * 32, file_size=50001, local_path="data/y")
    _, created3 = reg.upsert(rec2)
    assert created3 is True
    assert len(reg) == 2


def test_duplicate_ingestion_does_not_create_duplicate_manifest_rows(tmp_path: Path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    spec = docspecs.Q1FY27_DOCUMENTS[2]
    rec = DocumentRecord.from_spec(spec, document_hash="cc" * 32, file_size=800000, local_path="data/z")
    reg.upsert(rec)
    reg.upsert(rec)
    out = write_manifest(reg.all(), tmp_path / "manifest.json")
    entries = json.loads(out.read_text(encoding="utf-8"))
    assert len(entries) == 1


def test_validate_rejects_truncated_and_wrong_kind(tmp_path: Path):
    tiny = tmp_path / "tiny.pdf"
    tiny.write_bytes(b"%PDF-1.4 short")
    with pytest.raises(ValidationError):
        validate_file(tiny, min_bytes=100_000, expected_kind="pdf")

    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"<html>not a pdf" + b"x" * 5000)
    with pytest.raises(ValidationError):
        validate_file(fake_pdf, min_bytes=100, expected_kind="pdf")

    ok_html = tmp_path / "ok.html"
    ok_html.write_bytes(b"<!DOCTYPE html><html>" + b"x" * 5000)
    res = validate_file(ok_html, min_bytes=100, expected_kind="html")
    assert res.kind == "html"


def test_hashing_known_vector(tmp_path: Path):
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == sha256_bytes(b"abc")
