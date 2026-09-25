"""JSON document registry — makes ingestion idempotent without needing a DB.

Keyed by document_hash: ingesting the same bytes twice returns the existing
record instead of creating a duplicate, even if the filename differs.
A secondary index on source_url lets us detect URL re-ingestion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app.models.document import DocumentRecord


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            self._records = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._records = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._records, indent=2), encoding="utf-8")

    def get_by_hash(self, document_hash: str) -> DocumentRecord | None:
        raw = self._records.get(document_hash)
        return DocumentRecord(**raw) if raw else None

    def get_by_url(self, source_url: str) -> DocumentRecord | None:
        for raw in self._records.values():
            if raw["source_url"] == source_url:
                return DocumentRecord(**raw)
        return None

    def upsert(self, record: DocumentRecord) -> tuple[DocumentRecord, bool]:
        """Insert or return existing. Returns (record, created)."""
        existing = self.get_by_hash(record.document_hash)
        if existing is not None:
            return existing, False
        self._records[record.document_hash] = record.model_dump(mode="json")
        self._save()
        return record, True

    def all(self) -> list[DocumentRecord]:
        return [DocumentRecord(**raw) for raw in self._records.values()]

    def __len__(self) -> int:
        return len(self._records)

    def manifest(self) -> list[dict]:
        return [r.manifest_entry() for r in self.all()]


def write_manifest(records: Iterable[DocumentRecord], path: Path) -> Path:
    entries = [r.manifest_entry() for r in records]
    entries.sort(key=lambda e: e["filename"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path
