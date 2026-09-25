"""Milestone 4 tests: RRF fusion, citations, context packs."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from app.config import get_settings
from app.models.chunk import Chunk
from app.models.document import ReportType
from app.models.database import get_session, init_db
from app.retrieval.bm25 import BM25Index
from app.retrieval.citations import (
    Citation,
    cite_chunk,
    cite_fact,
    dedupe_citations,
    format_sources_block,
)
from app.retrieval.context import build_context
from app.retrieval.embeddings import HashEmbeddingProvider, SentenceTransformerProvider
from app.retrieval.hybrid import HybridRetriever, fuse_rrf
from app.retrieval.interfaces import RetrievedChunk
from app.retrieval.pipeline import run_indexing
from app.retrieval.retrievers import BM25Retriever, PassthroughReranker, SemanticRetriever
from app.retrieval.vector_store import LocalVectorStore

needs_seed = pytest.mark.skipif(
    not get_settings().bm25_path.exists(),
    reason="M3 index not built (run scripts/index_q1fy27.py)",
)


def _chunk(cid: str, page: int | None = 1, doctype: str = "media_release") -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id="dochash",
        page_number=page,
        printed_label="Page 1 of 20" if page else "",
        section="sec",
        text=f"narrative text for chunk {cid} " + "about reliance earnings performance. ",
        period="Q1 FY27",
        document_type=doctype,
        report_type=ReportType.CONSOLIDATED,
        source_location=f"loc {cid}",
    )


class _Stub:
    name = "stub"

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.seen_filters = None

    def search(self, query: str, k: int = 8, filters: dict | None = None):
        self.seen_filters = filters
        return [RetrievedChunk(chunk=_chunk(cid), score=1.0, source=self.name)
                for cid in self.ids[:k]]


# --- fusion math (deterministic, no index needed) --------------------------------

def test_rrf_prefers_multi_source_and_dedupes():
    sem = _Stub(["A", "B", "C"])
    bm25 = _Stub(["B", "C", "D"])
    fused = fuse_rrf([("semantic", sem.search("q")), ("bm25", bm25.search("q"))])
    ids = [h.chunk.chunk_id for h in fused]
    assert sorted(ids) == ["A", "B", "C", "D"]  # union, no duplicates
    assert ids[0] == "B"  # rank1+rank1 beats rank1-only and rank2+rank2
    assert all(h.source == "hybrid" for h in fused)
    assert fused[0].ranks == {"semantic", "bm25"}


def test_hybrid_forwards_filters_truncates_and_uses_reranker():
    sem, bm25 = _Stub(["A", "B"]), _Stub(["B", "C"])

    class Reverse:
        name = "reverse"

        def rerank(self, query, candidates, k):
            return list(reversed(candidates))[:k]

    r = HybridRetriever(sem, bm25, reranker=Reverse())
    out = r.search("q", k=2, filters={"quarter": "Q1"})
    assert sem.seen_filters == {"quarter": "Q1"} and bm25.seen_filters == {"quarter": "Q1"}
    assert len(out) == 2

    plain = HybridRetriever(_Stub(["A", "B"]), _Stub(["B", "C"]))
    assert [h.chunk.chunk_id for h in plain.search("q", k=10)] != \
        [h.chunk.chunk_id for h in HybridRetriever(_Stub(["A", "B"]), _Stub(["B", "C"]),
                                                   reranker=Reverse()).search("q", k=10)]


# --- citations --------------------------------------------------------------------

def test_cite_pdf_chunk_has_real_page():
    c = cite_chunk(_chunk("x", page=6))
    assert c.short == "[Q1 FY27 Media Release, p. 6]"
    assert c.page == 6 and c.kind == "narrative"


def test_cite_table_chunk_kind():
    ch = _chunk("x", page=8)
    ch.chunk_type = "table"
    assert cite_chunk(ch).kind == "table"


def test_cite_xbrl_chunk_never_invents_pages():
    ch = _chunk("x", page=None, doctype="xbrl")
    c = cite_chunk(ch)
    assert c.short == "[NSE Q1 FY27 Consolidated XBRL]"
    assert c.page is None and "p." not in c.short


def test_cite_facts():
    from app.models.financial import FinancialFact

    fx = FinancialFact(
        financial_year="FY27", quarter="Q1", period="Q1 FY27", metric="revenue_from_operations",
        value=311850.0, report_type=ReportType.CONSOLIDATED, source_document_id="h",
        source_location="NSE Q1 FY27 consolidated XBRL, P&L table (table 1)",
        extraction_method="xbrl_table",
    )
    assert cite_fact(fx).short == "[NSE Q1 FY27 Consolidated XBRL]"
    assert cite_fact(fx).page is None

    fp = FinancialFact(
        financial_year="FY27", quarter="Q1", period="Q1 FY27", metric="revenue_from_operations",
        value=311850.0, report_type=ReportType.CONSOLIDATED, source_document_id="h",
        source_location="RIL Q1 FY27 media release consolidated annexure, PDF p. 21: Revenue",
        extraction_method="pdf_annexure_table",
    )
    assert cite_fact(fp).short == "[Q1 FY27 Media Release annexure, p. 21]"


def test_dedupe_and_sources_block():
    cs = dedupe_citations([Citation("[B]", "d", 1, "narrative"),
                           Citation("[A]", "d", None, "fact"),
                           Citation("[B]", "d", 1, "narrative")])
    assert [c.short for c in cs] == ["[B]", "[A]"]
    assert [c.index for c in cs] == [1, 2]
    assert "[1] [B]" in format_sources_block(cs)


# --- real index ---------------------------------------------------------------------

def _real_hybrid():
    session = get_session()
    provider = SentenceTransformerProvider()
    sem = SemanticRetriever(provider, LocalVectorStore(session))
    bm25 = BM25Retriever(BM25Index.load(get_settings().bm25_path), session)
    return session, HybridRetriever(sem, bm25, reranker=PassthroughReranker())


@needs_seed
def test_hybrid_search_end_to_end_with_valid_citations():
    session, hybrid = _real_hybrid()
    try:
        hits = hybrid.search("Jio Platforms revenue growth", k=6)
        assert 0 < len(hits) <= 6
        ids = [h.chunk.chunk_id for h in hits]
        assert len(set(ids)) == len(ids)
        assert any("jio" in (h.chunk.text + h.chunk.section).lower() for h in hits)
        for h in hits:
            c = cite_chunk(h.chunk)
            assert "p. None" not in c.short
            if h.chunk.document_type == "xbrl":
                assert c.page is None and "p." not in c.short
            else:
                assert c.page is not None and c.page >= 1
        # Deterministic across runs.
        again = [h.chunk.chunk_id for h in hybrid.search("Jio Platforms revenue growth", k=6)]
        assert again == ids
    finally:
        session.close()


@needs_seed
def test_hybrid_respects_filters():
    session, hybrid = _real_hybrid()
    try:
        assert hybrid.search("revenue", k=5, filters={"quarter": "Q9"}) == []
        hits = hybrid.search("ratios", k=5, filters={"document_type": "xbrl"})
        assert hits and all(h.chunk.document_type == "xbrl" for h in hits)
    finally:
        session.close()


@needs_seed
def test_build_context_attaches_facts_and_citations():
    session, hybrid = _real_hybrid()
    try:
        pack = build_context(
            "What was consolidated revenue in Q1 FY27?",
            hybrid, session, k=4,
            fact_requests=[{"metric": "revenue_from_operations", "period": "Q1 FY27",
                            "report_type": "consolidated"},
                           {"metric": "nope_missing", "period": "Q1 FY27"}],
        )
        assert len(pack.facts) == 1 and pack.facts[0].value == 311850.0
        shorts = [c.short for c in pack.citations]
        assert "[NSE Q1 FY27 Consolidated XBRL]" in shorts  # fact citation present
        assert len(shorts) == len(set(shorts))  # deduped
        assert [c.index for c in pack.citations] == list(range(1, len(pack.citations) + 1))
    finally:
        session.close()
