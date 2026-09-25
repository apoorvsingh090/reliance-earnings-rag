"""Local vector store: SQLite-backed chunk metadata + numpy brute-force cosine.

Production target is pgvector (same ChunkRow schema; embeddings move to a
halfvec column when DATABASE_URL points at Postgres). This local store keeps
M3 fully runnable without Docker while honoring identical call semantics:
metadata filtering happens in SQL before scoring, never after.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import select

from app.models.database import ChunkEmbeddingRow, ChunkRow

FILTER_FIELDS = ("company", "ticker", "financial_year", "quarter", "period",
                 "document_type", "report_type", "chunk_type")


class LocalVectorStore:
    name = "local-numpy"

    def __init__(self, session) -> None:
        self.session = session

    def _filtered_ids(self, filters: dict | None) -> set[str] | None:
        if not filters:
            return None
        stmt = select(ChunkRow.chunk_id)
        for key, value in filters.items():
            if key not in FILTER_FIELDS or value is None:
                continue
            stmt = stmt.where(getattr(ChunkRow, key) == value)
        return set(self.session.execute(stmt).scalars().all())

    def search(
        self, query_vec: np.ndarray, k: int = 8, filters: dict | None = None
    ) -> list[tuple[str, float]]:
        allowed = self._filtered_ids(filters)
        if allowed is not None and not allowed:
            return []
        rows = self.session.execute(select(ChunkEmbeddingRow)).scalars().all()
        ids: list[str] = []
        mats: list[np.ndarray] = []
        for r in rows:
            if allowed is not None and r.chunk_id not in allowed:
                continue
            ids.append(r.chunk_id)
            mats.append(np.frombuffer(r.embedding, dtype=np.float32))
        if not ids:
            return []
        mat = np.stack(mats)
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        denom = np.linalg.norm(q)
        scores = (mat @ q / (np.linalg.norm(mat, axis=1) * denom + 1e-12)).tolist()
        ranked = sorted(zip(ids, scores), key=lambda t: t[1], reverse=True)
        return ranked[:k]

    def hydrate(self, chunk_id: str) -> ChunkRow | None:
        return self.session.get(ChunkRow, chunk_id)
