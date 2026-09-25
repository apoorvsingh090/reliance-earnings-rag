"""Hybrid retrieval: Reciprocal Rank Fusion over semantic + BM25.

Both sub-retrievers receive the same metadata filters (enforced in SQL /
index-masking before scoring), run independently, and merge by RRF:

    rrf(chunk) = sum over lists of 1 / (RRF_K + rank)

Dedupe is by chunk_id. A Reranker (protocol) refines the fused top-N; the
default passthrough keeps RRF order. M4 fuses; M8/eval motivates a learned
rerank later without touching this seam.
"""

from __future__ import annotations

from app.retrieval.interfaces import RetrievedChunk

RRF_K = 60


class HybridRetriever:
    name = "hybrid"

    def __init__(self, semantic, bm25, reranker=None, rrf_k: int = RRF_K) -> None:
        self.semantic = semantic
        self.bm25 = bm25
        self.reranker = reranker
        self.rrf_k = rrf_k

    def search(
        self, query: str, k: int = 8, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        width = max(k * 3, 10)
        sem_hits = self.semantic.search(query, k=width, filters=filters)
        bm25_hits = self.bm25.search(query, k=width, filters=filters)
        fused = fuse_rrf(
            [("semantic", sem_hits), ("bm25", bm25_hits)], rrf_k=self.rrf_k
        )
        if self.reranker is not None:
            fused = self.reranker.rerank(query, fused, k=max(k * 2, k))
        return fused[:k]


def fuse_rrf(
    ranked_lists: list[tuple[str, list[RetrievedChunk]]], rrf_k: int = RRF_K
) -> list[RetrievedChunk]:
    """Merge ranked lists by RRF. Deterministic: (-score, -sources, chunk_id)."""
    agg: dict[str, dict] = {}
    for source, hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            entry = agg.setdefault(
                hit.chunk.chunk_id, {"chunk": hit.chunk, "score": 0.0, "sources": set()}
            )
            entry["score"] += 1.0 / (rrf_k + rank)
            entry["sources"].add(source)
    fused = [
        RetrievedChunk(chunk=e["chunk"], score=e["score"], source="hybrid",
                       ranks=set(e["sources"]))
        for e in agg.values()
    ]
    fused.sort(key=lambda h: (-h.score, -len(h.ranks), h.chunk.chunk_id))
    return fused
