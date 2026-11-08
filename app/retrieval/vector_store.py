"""Local vector store: SQLite-backed chunk metadata + numpy brute-force cosine.

Production target is pgvector (same ChunkRow schema; embeddings move to a
halfvec column when DATABASE_URL points at Postgres). This local store keeps
M3 fully runnable without Docker while honoring identical call semantics:
metadata filtering happens in SQL before scoring, never after.

The embedding matrix is loaded once per engine and cached in memory keyed on
(embedding count, total dim); a per-query full table scan + np.stack is the
hot path this module is built to avoid.
"""

from __future__ import annotations

import threading

import numpy as np
from sqlalchemy import func, select

from app.models.database import ChunkEmbeddingRow, ChunkRow

FILTER_FIELDS = ("company", "ticker", "financial_year", "quarter", "period",
                 "document_type", "report_type", "chunk_type")


def filtered_chunk_ids(session, filters: dict | None) -> set[str] | None:
    """Resolve metadata filters to chunk ids in SQL (None = no filter)."""
    if not filters:
        return None
    stmt = select(ChunkRow.chunk_id)
    for key, value in filters.items():
        if key not in FILTER_FIELDS or value is None:
            continue
        stmt = stmt.where(getattr(ChunkRow, key) == value)
    return set(session.execute(stmt).scalars().all())


class LocalVectorStore:
    name = "local-numpy"

    def __init__(self, session) -> None:
        self.session = session

    def _filtered_ids(self, filters: dict | None) -> set[str] | None:
        return filtered_chunk_ids(self.session, filters)

    def search(
        self, query_vec: np.ndarray, k: int = 8, filters: dict | None = None
    ) -> list[tuple[str, float]]:
        allowed = self._filtered_ids(filters)
        if allowed is not None and not allowed:
            return []
        ids, mat, norms = _embedding_matrix(self.session)
        if not ids:
            return []
        if allowed is not None:
            mask = np.fromiter(
                (cid in allowed for cid in ids), dtype=bool, count=len(ids)
            )
            sub_mat = mat[mask]
            if sub_mat.shape[0] == 0:
                return []
            sub_ids = [cid for cid, keep in zip(ids, mask.tolist()) if keep]
            sub_norms = norms[mask]
        else:
            sub_ids, sub_mat, sub_norms = ids, mat, norms
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        denom = np.linalg.norm(q)
        scores = (sub_mat @ q / (sub_norms * denom + 1e-12))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(sub_ids[i], float(scores[i])) for i in order]

    def hydrate(self, chunk_id: str) -> ChunkRow | None:
        return self.session.get(ChunkRow, chunk_id)


_MAT_CACHE: dict[tuple[int, int, int], tuple[list[str], np.ndarray, np.ndarray]] = {}
_MAT_LOCK = threading.Lock()


def _embedding_matrix(session) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Cached (ids, matrix, row-norms) for the engine bound to this session."""
    engine = session.get_bind()
    n = session.execute(
        select(func.count()).select_from(ChunkEmbeddingRow)
    ).scalar() or 0
    dsum = int(session.execute(
        select(func.coalesce(func.sum(ChunkEmbeddingRow.dim), 0))
    ).scalar() or 0)
    key = (id(engine), n, dsum)
    with _MAT_LOCK:
        entry = _MAT_CACHE.get(key)
        if entry is None:
            rows = session.execute(select(ChunkEmbeddingRow)).scalars().all()
            ids = [r.chunk_id for r in rows]
            if rows:
                mats = [
                    np.frombuffer(r.embedding, dtype=np.float32) for r in rows
                ]
                mat = np.stack(mats)
                norms = np.linalg.norm(mat, axis=1)
            else:
                mat = np.zeros((0, 0), dtype=np.float32)
                norms = np.zeros(0, dtype=np.float32)
            entry = (ids, mat, norms)
            _MAT_CACHE.clear()  # keep a single hot matrix in memory
            _MAT_CACHE[key] = entry
        return entry
