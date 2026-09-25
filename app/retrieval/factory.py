"""Shared retriever construction (hybrid + passthrough rerank)."""

from __future__ import annotations

from app.config import get_settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import SentenceTransformerProvider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.retrievers import BM25Retriever, PassthroughReranker, SemanticRetriever
from app.retrieval.vector_store import LocalVectorStore


def default_retriever(session, provider=None) -> HybridRetriever:
    provider = provider or SentenceTransformerProvider()
    sem = SemanticRetriever(provider, LocalVectorStore(session))
    bm25 = BM25Retriever(BM25Index.load(get_settings().bm25_path), session)
    return HybridRetriever(sem, bm25, reranker=PassthroughReranker())
