"""Milestone 1 pipeline: download -> validate -> hash -> register -> manifest.

Idempotent: re-running with the same bytes never creates duplicate records.
The document_hash is the identity; filename/URL are attributes.
M7 adds register_local_file() for user-uploaded documents (no download step).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config import get_settings
from app.ingestion import documents as docspecs
from app.ingestion.downloader import download_file
from app.ingestion.hashing import sha256_file
from app.ingestion.registry import DocumentRegistry, write_manifest
from app.ingestion.validate import validate_file
from app.models.document import DocumentRecord, DocumentSpec

log = logging.getLogger(__name__)


def ingest_document(spec: DocumentSpec, registry: DocumentRegistry) -> tuple[DocumentRecord, bool]:
    settings = get_settings()
    dest = settings.raw_dir / spec.filename
    min_bytes = docspecs.MIN_BYTES.get(spec.filename, 1_000)
    expected_kind = docspecs.EXPECTED_KIND.get(spec.filename, "pdf")

    download_file(spec.source_url, dest, min_bytes=min_bytes)
    validation = validate_file(dest, min_bytes=min_bytes, expected_kind=expected_kind)
    digest = sha256_file(dest)

    # Store local_path relative to project root for portability.
    from app.config import PROJECT_ROOT

    try:
        local_rel = str(dest.relative_to(PROJECT_ROOT))
    except ValueError:
        local_rel = str(dest)

    record = DocumentRecord.from_spec(
        spec, document_hash=digest, file_size=validation.size, local_path=local_rel
    )
    stored, created = registry.upsert(record)
    log.info(
        "%s %s (%d bytes, sha256 %.12s...)",
        "ingested" if created else "duplicate-skip",
        spec.filename,
        validation.size,
        digest,
    )
    return stored, created


def ingest_all(specs: list[DocumentSpec] | None = None) -> list[DocumentRecord]:
    settings = get_settings()
    settings.ensure_dirs()
    registry = DocumentRegistry(settings.registry_path)
    specs = specs if specs is not None else docspecs.Q1FY27_DOCUMENTS
    records: list[DocumentRecord] = []
    for spec in specs:
        record, _ = ingest_document(spec, registry)
        records.append(record)
    write_manifest(registry.all(), settings.manifest_path)
    return records


def load_manifest() -> list[dict]:
    import json

    path = get_settings().manifest_path
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def register_local_file(spec: DocumentSpec, src_path: Path) -> tuple[DocumentRecord, bool]:
    """Register a user-uploaded file (M7). No download; validate+hash+register.

    Returns (record, created). Same bytes twice -> same record, no duplicate.
    """
    from app.config import PROJECT_ROOT

    settings = get_settings()
    settings.ensure_dirs()
    suffix = src_path.suffix.lower()
    expected_kind = "pdf" if suffix == ".pdf" else "html"
    min_bytes = docspecs.MIN_BYTES.get(spec.filename, 100)

    safe_name = Path(spec.filename).name  # no directory escapes
    dest = settings.raw_dir / safe_name
    if not dest.exists():
        shutil.copyfile(src_path, dest)

    validate_file(dest, min_bytes=min_bytes, expected_kind=expected_kind)
    digest = sha256_file(dest)
    try:
        local_rel = str(dest.relative_to(PROJECT_ROOT))
    except ValueError:
        local_rel = str(dest)
    record = DocumentRecord.from_spec(
        spec, document_hash=digest, file_size=dest.stat().st_size, local_path=local_rel
    )
    registry = DocumentRegistry(settings.registry_path)
    stored, created = registry.upsert(record)
    write_manifest(registry.all(), settings.manifest_path)
    log.info("%s %s (sha256 %.12s...)", "registered" if created else "duplicate-skip",
             safe_name, digest)
    return stored, created
