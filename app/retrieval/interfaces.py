"""Stable boundaries for M4+ extension (spec section 11).

Concrete providers/retrievers live beside these protocols; application code
must depend on the protocols so embedding models, vector backends, and
rerankers stay swappable without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from app.models.chunk import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    source: str  # "semantic" | "bm25" | "hybrid"
    ranks: dict = None  # type: ignore[assignment]  # fusion provenance, e.g. {"semantic","bm25"}

    def __post_init__(self) -> None:
        if self.ranks is None:
            self.ranks = {self.source}


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return (n, dim) float32 row-normalized embeddings."""
        ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(
        self, query: str, k: int = 8, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        ...
