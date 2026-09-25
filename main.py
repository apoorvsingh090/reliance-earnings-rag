"""FastAPI backend (grows across milestones; M6: + ask with router + answers)."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.financial.engine import (
    MissingFactError,
    MixedBasisError,
    calc_absolute,
    calc_cagr,
    calc_margin,
    calc_pp_change,
    calc_qoq,
    calc_share,
    calc_yoy,
)
from app.financial.query import get_fact
from app.ingestion.pipeline import load_manifest
from app.llm.answers import answer_question
from app.llm.providers import get_llm_provider
from app.models.database import get_session
from app.retrieval.bm25 import BM25Index
from app.retrieval.citations import cite_chunk, format_sources_block
from app.retrieval.context import build_context
from app.retrieval.embeddings import SentenceTransformerProvider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.retrievers import BM25Retriever, PassthroughReranker, SemanticRetriever
from app.retrieval.vector_store import LocalVectorStore

app = FastAPI(title="Reliance Earnings Intelligence RAG", version="0.9.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "milestone": 9, "llm": get_llm_provider() is not None}


@app.get("/documents")
def documents() -> JSONResponse:
    return JSONResponse(content=load_manifest())


@app.get("/facts")
def facts(
    metric: str, period: str, report_type: str = "consolidated", segment: str | None = None
) -> dict:
    """One structured fact with provenance. report_type is explicit (no mixing)."""
    session = get_session()
    try:
        row = get_fact(session, metric, period, report_type, segment)
    finally:
        session.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no fact: {metric} {period} {report_type}")
    return {
        "metric": row.metric,
        "value": row.value,
        "unit": row.unit,
        "currency": row.currency,
        "period": row.period,
        "report_type": row.report_type,
        "segment": row.segment,
        "source_location": row.source_location,
        "extraction_method": row.extraction_method,
    }


@lru_cache(maxsize=1)
def _provider() -> SentenceTransformerProvider:
    return SentenceTransformerProvider()


def _semantic_for(session) -> SemanticRetriever:
    return SemanticRetriever(_provider(), LocalVectorStore(session))


def _bm25_for(session) -> BM25Retriever:
    return BM25Retriever(BM25Index.load(get_settings().bm25_path), session)


@app.get("/search")
def search(q: str, k: int = 8, method: str = "hybrid") -> dict:
    """Retrieval with citations. method: hybrid (default) | semantic | bm25."""
    session = get_session()
    try:
        if method == "semantic":
            retriever = _semantic_for(session)
        elif method == "bm25":
            retriever = _bm25_for(session)
        else:
            sem = SemanticRetriever(_provider(), LocalVectorStore(session))
            retriever = HybridRetriever(sem, _bm25_for(session),
                                        reranker=PassthroughReranker())
        pack = build_context(q, retriever, session, k=k)
        return {
            "query": q,
            "method": method,
            "hits": [
                {"score": h.score, "source": h.source, "page": h.chunk.page_number,
                 "section": h.chunk.section,
                 "citation": cite_chunk(h.chunk).short,
                 "text": h.chunk.text[:500]}
                for h in pack.chunks
            ],
            "sources_used": format_sources_block(pack.citations),
        }
    finally:
        session.close()


@app.get("/calc")
def calc(
    op: str,
    metric: str = "revenue_from_operations",
    current: str = "Q1 FY27",
    prior: str = "Q1 FY26",
    report_type: str = "consolidated",
    segment: str | None = None,
    num: str | None = None,
    steps: int = 4,
) -> dict:
    """Financial calculations (Python, never LLM arithmetic).

    ops: yoy | qoq | margin (num= required) | share (num=part metric) |
         pp_change | absolute | cagr.
    """
    session = get_session()
    try:
        if op == "yoy":
            r = calc_yoy(session, metric, current, prior, report_type, segment)
        elif op == "qoq":
            r = calc_qoq(session, metric, current, prior, report_type, segment)
        elif op == "margin":
            r = calc_margin(session, num or metric, metric, current, report_type, segment)
        elif op == "share":
            r = calc_share(session, num or metric, metric, current, report_type, segment)
        elif op == "pp_change":
            r = calc_pp_change(session, metric, current, prior, report_type, segment)
        elif op == "absolute":
            r = calc_absolute(session, metric, current, prior, report_type, segment)
        elif op == "cagr":
            r = calc_cagr(session, metric, prior, current, steps, report_type, segment)
        else:
            raise HTTPException(status_code=400, detail=f"unknown op: {op}")
    except (MissingFactError, MixedBasisError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, MissingFactError) else 400,
                            detail=str(exc))
    finally:
        session.close()
    if r is None:
        raise HTTPException(status_code=422, detail="calculation undefined (missing/zero input)")
    return r.model_dump()


@app.get("/ask")
def ask(q: str, k: int = 6) -> dict:
    """Full Q&A: route -> facts -> calculations -> hybrid context -> answer.

    Uses the LLM when a key is configured, otherwise the extractive template.
    """
    session = get_session()
    try:
        sem = SemanticRetriever(_provider(), LocalVectorStore(session))
        retriever = HybridRetriever(sem, _bm25_for(session),
                                    reranker=PassthroughReranker())
        answer = answer_question(q, session, retriever, llm=get_llm_provider(), k=k)
        return {
            "question": q,
            "route": answer.route.query_class.value,
            "mode": answer.mode,
            "basis": answer.basis,
            "answer": answer.text,
            "calculations": [c.model_dump() for c in answer.calculations],
        }
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
