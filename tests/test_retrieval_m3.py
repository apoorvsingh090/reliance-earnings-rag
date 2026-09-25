"""Milestone 3 tests: chunking, embeddings, vector store, BM25, idempotency."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, select

from app.config import PROJECT_ROOT, get_settings
from app.models.database import ChunkRow, count_chunks, count_embeddings, get_session, init_db
from app.retrieval.bm25 import BM25Index
from app.retrieval.chunking import MAX_CHARS, split_long
from app.retrieval.embeddings import HashEmbeddingProvider, SentenceTransformerProvider
from app.retrieval.interfaces import EmbeddingProvider, Reranker, Retriever
from app.retrieval.pipeline import run_indexing
from app.retrieval.retrievers import (
    BM25Retriever,
    PassthroughReranker,
    SemanticRetriever,
)
from app.retrieval.vector_store import LocalVectorStore

RAW = PROJECT_ROOT / "data" / "raw" / "reliance"
needs_seed = pytest.mark.skipif(
    not (RAW / "RIL_Q1FY27_media_release.pdf").exists(),
    reason="seed documents not downloaded",
)


def _mem_index(provider=None):
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    summary = run_indexing(engine=engine, provider=provider or HashEmbeddingProvider())
    return engine, summary


# --- splitter -----------------------------------------------------------------

def test_split_long_never_slivers():
    assert split_long("x" * 100) == ["x" * 100]
    assert split_long("y" * MAX_CHARS) == ["y" * MAX_CHARS]
    parts = split_long("a" * 100 + "\n" + "b" * 2000)
    assert 2 <= len(parts) <= 4
    assert all(len(p) <= MAX_CHARS for p in parts)
    long_parts = split_long("z" * 5000)
    assert 3 <= len(long_parts) <= 5
    # Full coverage: every char appears in at least one part.
    covered = set()
    for p in long_parts:
        covered.update(p)
    assert covered == {"z"}


# --- chunk shape ----------------------------------------------------------------

@needs_seed
def test_chunks_have_provenance_and_separate_tables():
    engine, summary = _mem_index()
    session = get_session(engine)
    assert summary["total_chunks"] > 50
    rows = session.execute(select(ChunkRow)).scalars().all()
    types = {r.chunk_type for r in rows}
    assert {"text", "table", "descriptor"} <= types
    for r in rows:
        assert len(r.chunk_id) == 16 and r.document_id and r.source_location
        assert len(r.text.strip()) >= 50
        if r.document_type == "xbrl":
            assert r.page_number is None
        else:
            assert r.page_number is not None and r.page_number >= 1
    session.close()


@needs_seed
def test_annexure_numbers_stay_out_of_vectors():
    engine, _ = _mem_index()
    session = get_session(engine)
    rows = session.execute(select(ChunkRow)).scalars().all()
    annex = [r for r in rows if r.printed_label and "of 14" in r.printed_label]
    assert annex  # annexure pages are chunked (notes/formulae), just not data rows
    for r in annex:
        for tok in ("311,850", "340,257", "28,407", "2,208,747", "880,365", "23,196"):
            assert tok not in r.text, f"leak on p.{r.page_number}: {tok}"
    session.close()


@needs_seed
def test_indexing_idempotent():
    engine, first = _mem_index()
    session = get_session(engine)
    n_chunks, n_emb = count_chunks(session), count_embeddings(session)
    assert n_chunks > 50 and n_emb == n_chunks
    second = run_indexing(engine=engine, provider=HashEmbeddingProvider())
    assert count_chunks(session) == n_chunks
    assert count_embeddings(session) == n_emb
    assert second["chunks_created"] == 0 and second["embedded"] == 0
    assert second["bm25_rebuilt"] is False
    session.close()


# --- providers / protocols -------------------------------------------------------

def test_hash_provider_deterministic_and_compliant():
    p = HashEmbeddingProvider(dim=64)
    assert isinstance(p, EmbeddingProvider)
    a = p.embed_texts(["Jio revenue growth", "Jio revenue growth"])
    b = p.embed_texts(["cnbc retail stores"])
    assert a.shape == (2, 64) and abs(a[0] @ a[1] - 1.0) < 1e-6
    assert abs(a[0] @ b[0] - 1.0) > 1e-3


def test_bm25_search_and_metadata_filter(tmp_path: Path):
    idx = BM25Index().build(
        ["a", "b", "c"],
        ["Jio Platforms revenue grew twelve percent", "Jio retail stores expanded",
         "oil to chemicals margin environment"],
    )
    assert [cid for cid, _ in idx.search("Jio revenue", k=3)][0] == "a"
    masked = idx.search("Jio", k=3, allowed_ids={"b", "c"})
    assert masked and masked[0][0] == "b"  # "a" scores but is masked out
    assert idx.search("zzz_no_match", k=3) == []
    p = tmp_path / "bm25.pkl"
    idx.save(p)
    assert BM25Index.load(p).search("Jio revenue", k=1)[0][0] == "a"


def test_retriever_and_reranker_protocols():
    assert isinstance(SemanticRetriever(None, None), Retriever)  # type: ignore[arg-type]
    assert isinstance(PassthroughReranker(), Reranker)


@needs_seed
def test_vector_store_metadata_filter_returns_empty():
    engine, _ = _mem_index()
    session = get_session(engine)
    store = LocalVectorStore(session)
    provider = HashEmbeddingProvider()
    retriever = SemanticRetriever(provider, store)
    assert retriever.search("revenue", k=3, filters={"quarter": "Q9"}) == []
    hits = retriever.search("ratios res", k=5, filters={"document_type": "xbrl"})
    assert hits and all(h.chunk.document_type == "xbrl" for h in hits)
    session.close()


@needs_seed
def test_semantic_search_finds_jio_narrative():
    engine, _ = _mem_index(provider=SentenceTransformerProvider())  # real model; cached
    session = get_session(engine)
    provider = SentenceTransformerProvider()
    retriever = SemanticRetriever(provider, LocalVectorStore(session))
    hits = retriever.search("Jio Platforms revenue growth", k=5)
    assert hits
    assert any("jio" in h.chunk.text.lower() or "jio" in h.chunk.section.lower() for h in hits)
    assert all(h.source == "semantic" for h in hits)
    session.close()


@needs_seed
def test_bm25_retriever_on_real_index():
    session = get_session()  # file DB index built by scripts/index_q1fy27.py
    settings = get_settings()
    assert settings.bm25_path.exists()
    retriever = BM25Retriever(BM25Index.load(settings.bm25_path), session)
    hits = retriever.search("KGD6 production", k=3)
    assert hits and any("kgd6" in h.chunk.text.lower() for h in hits)
    session.close()
