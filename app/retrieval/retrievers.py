"""Thin retrievers over the two indexes + a passthrough reranker placeholder."""

from __future__ import annotations

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


class SemanticRetriever:
    name = "semantic"

    def __init__(self, provider, store) -> None:
        self.provider = provider
        self.store = store

    def search(self, query: str, k: int = 8, filters: dict | None = None) -> list[RetrievedChunk]:
        q = self.provider.embed_texts([query])[0]
        out = []
        for cid, score in self.store.search(q, k=k, filters=filters):
            row = self.store.hydrate(cid)
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
        from app.retrieval.vector_store import FILTER_FIELDS

        allowed = None
        if filters:
            from sqlalchemy import select

            stmt = select(ChunkRow.chunk_id)
            for key, value in filters.items():
                if key not in FILTER_FIELDS or value is None:
                    continue
                stmt = stmt.where(getattr(ChunkRow, key) == value)
            allowed = set(self.session.execute(stmt).scalars().all())
        out = []
        for cid, score in self.index.search(query, k=k, allowed_ids=allowed):
            row = self.session.get(ChunkRow, cid)
            if row is None:
                continue
            out.append(RetrievedChunk(chunk=_to_chunk(row), score=float(score), source="bm25"))
        return out


class PassthroughReranker:
    """Keeps retrieval order; M4 replaces this with a real cross-encoder/LLM rerank."""

    name = "passthrough"

    def rerank(self, query: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        return list(candidates[:k])
