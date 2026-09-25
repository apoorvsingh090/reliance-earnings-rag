"""Milestone 3 pipeline: chunk registered documents -> embed -> index (idempotent).

Chunks and embeddings are upserted by deterministic ids, so re-running only
fills gaps. The BM25 pickle rebuilds whenever the chunk set changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from app.config import PROJECT_ROOT, get_settings
from app.extraction.pdf import parse_pdf
from app.extraction.xbrl import parse_xbrl
from app.ingestion.registry import DocumentRegistry
from app.models.chunk import Chunk
from app.models.database import (
    count_chunks,
    count_embeddings,
    get_session,
    init_db,
    upsert_chunk,
    upsert_embedding,
)
from app.retrieval.bm25 import BM25Index
from app.retrieval.chunking import chunk_pdf_page, chunk_xbrl, serialize_pdf_tables
from app.retrieval.embeddings import SentenceTransformerProvider

log = logging.getLogger(__name__)


def _resolve_raw_path(local_path: str) -> Path:
    p = Path(local_path)
    return p if p.exists() else PROJECT_ROOT / local_path


def build_chunks_for_record(record) -> list[Chunk]:
    raw_path = _resolve_raw_path(record.local_path)
    chunks: list[Chunk] = []
    if record.document_type.value == "xbrl":
        chunk_xbrl(chunks, record, parse_xbrl(raw_path))
    elif record.document_type.value == "media_release":
        parsed = parse_pdf(raw_path)
        doc = pymupdf.open(str(raw_path))
        try:
            table_cache = {p.pdf_page: serialize_pdf_tables(doc[p.pdf_page - 1]) for p in parsed.pages}
        finally:
            doc.close()
        for page in parsed.pages:
            if page.needs_ocr:
                log.warning("skipping needs_ocr page %s (%s)", page.pdf_page, record.filename)
                continue
            chunk_pdf_page(
                chunks, record, page.pdf_page, page.printed_label,
                page.text, page.headings, table_cache.get(page.pdf_page, []),
            )
    else:
        log.info("no M3 chunker for %s — skipped", record.document_type.value)
    return chunks


def run_indexing(engine=None, provider=None) -> dict:
    settings = get_settings()
    settings.ensure_dirs()
    registry = DocumentRegistry(settings.registry_path)
    records = registry.all()
    if not records:
        raise RuntimeError("registry is empty — run scripts/ingest_q1fy27.py first")

    engine = init_db(engine)
    session = get_session(engine)
    provider = provider or SentenceTransformerProvider(settings.embedding_model)

    summary = {"documents": 0, "chunks_created": 0, "chunks_reused": 0,
               "embedded": 0, "embeddings_reused": 0, "per_document": {}}
    for record in records:
        chunks = build_chunks_for_record(record)
        created = reused = 0
        for chunk in chunks:
            _, is_new = upsert_chunk(session, chunk)
            created, reused = (created + 1, reused) if is_new else (created, reused + 1)
        # Embed only chunks missing this provider's vectors.
        from sqlalchemy import select

        from app.models.database import ChunkEmbeddingRow

        have = {
            r for (r,) in session.execute(
                select(ChunkEmbeddingRow.chunk_id).where(ChunkEmbeddingRow.model == provider.name)
            ).all()
        }
        missing = [c for c in chunks if c.chunk_id not in have]
        embedded = 0
        for i in range(0, len(missing), 32):
            batch = missing[i:i + 32]
            vecs = provider.embed_texts([c.text for c in batch])
            for chunk, vec in zip(batch, vecs):
                _, is_new = upsert_embedding(session, chunk.chunk_id, vec, provider.name)
                if is_new:
                    embedded += 1
        summary["documents"] += 1
        summary["chunks_created"] += created
        summary["chunks_reused"] += reused
        summary["embedded"] += embedded
        summary["embeddings_reused"] += len(chunks) - len(missing)
        summary["per_document"][record.filename] = {
            "chunks": len(chunks), "new": created, "embedded": embedded,
        }

    # Rebuild BM25 when the chunk set changed.
    from sqlalchemy import select

    from app.models.database import ChunkRow

    rows = session.execute(select(ChunkRow.chunk_id, ChunkRow.text)).all()
    ids = [r[0] for r in rows]
    corpus_hash = BM25Index.corpus_hash_for(ids)
    bm25_path = settings.bm25_path
    rebuild = True
    if bm25_path.exists():
        try:
            rebuild = BM25Index.load(bm25_path).corpus_hash != corpus_hash
        except Exception:  # noqa: BLE001 — corrupt pickle rebuilds
            rebuild = True
    if rebuild:
        BM25Index().build(ids, [r[1] for r in rows]).save(bm25_path)
        log.info("BM25 rebuilt: %d chunks -> %s", len(ids), bm25_path)
    summary["bm25_rebuilt"] = rebuild
    summary["total_chunks"] = count_chunks(session)
    summary["total_embeddings"] = count_embeddings(session)
    session.close()
    return summary
