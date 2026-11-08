"""Thin retrievers over the two indexes + a passthrough reranker placeholder."""

from __future__ import annotations

from sqlalchemy import select

from app.models.chunk import Chunk
from app.models.database import ChunkRow
from app.retrieval.interfaces import RetrievedChunk


def _to_chunk(row: ChunkRow) -> Chunk:
    return Chunk(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        page_number=row.page_number,
        printed_label=row.printed_label,
        section=row.section,
        chunk_type=row.chunk_type,  # type: ignore[arg-type]
        text=row.text,
        company=row.company,
        ticker=row.ticker,
        financial_year=row.financial_year,
        quarter=row.quarter,
        period=row.period,
        document_type=row.document_type,
        report_type=row.report_type,  # type: ignore[arg-type]
        source_location=row.source_location,
    )


def _rows_for_ids(session, chunk_ids: list[str]) -> dict[str, ChunkRow]:
    """Batch hydration: one SELECT ... IN for k hits instead of k SELECTs."""
    if not chunk_ids:
        return {}
    stmt = select(ChunkRow).where(ChunkRow.chunk_id.in_(chunk_ids))
    return {r.chunk_id: r for r in session.execute(stmt).scalars().all()}


def _allowed_ids(session, filters: dict | None) -> set[str] | None:
    from app.retrieval.vector_store import filtered_chunk_ids

    return filtered_chunk_ids(session, filters)


class SemanticRetriever:
    name = "semantic"

    def __init__(self, provider, store) -> None:
        self.provider = provider
        self.store = store

    def search(self, query: str, k: int = 8, filters: dict | None = None) -> list[RetrievedChunk]:
        q = self.provider.embed_texts([query])[0]
        hits = self.store.search(q, k=k, filters=filters)
        session = getattr(self.store, "session", None)
        rows = (
            _rows_for_ids(session, [cid for cid, _ in hits])
            if session is not None
            else {}
        )
        out = []
        for cid, score in hits:
            row = rows.get(cid) if rows else self.store.hydrate(cid)
            if row is None:
                continue
            out.append(RetrievedChunk(chunk=_to_chunk(row), score=float(score), source="semantic"))
        return out


class BM25Retriever:
    name = "bm25"

    def __init__(self, index, session) -> None:
        self.index = index
        self.session = session

    def search(self, query: str, k: int = 8, filters: dict | None = None) -> list[RetrievedChunk]:
        allowed = _allowed_ids(self.session, filters)
        hits = self.index.search(query, k=k, allowed_ids=allowed)
        rows = _rows_for_ids(self.session, [cid for cid, _ in hits])
        out = []
        for cid, score in hits:
            row = rows.get(cid)
            if row is None:
                continue
            out.append(RetrievedChunk(chunk=_to_chunk(row), score=float(score), source="bm25"))
        return out


class PassthroughReranker:
    """Keeps retrieval order; M4 replaces this with a real cross-encoder/LLM rerank."""

    name = "passthrough"

    def rerank(self, query: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        return list(candidates[:k])
