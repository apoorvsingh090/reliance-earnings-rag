"""Reliance Earnings Intelligence — Streamlit UI (Milestone 7).

Sidebar: upload + ingest + process documents, question scope, retrieval settings.
Main: Q&A with route/mode/basis, calculations, and expandable sources.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, ".")

import streamlit as st

from app.config import get_settings
from app.ingestion.pipeline import load_manifest, register_local_file
from app.llm.answers import answer_question
from app.llm.providers import get_llm_provider
from app.models.database import count_chunks, count_embeddings, count_facts, get_session
from app.models.document import DocumentSpec
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import SentenceTransformerProvider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.retrievers import BM25Retriever, PassthroughReranker, SemanticRetriever
from app.retrieval.vector_store import LocalVectorStore

st.set_page_config(page_title="Reliance Earnings Intelligence", layout="wide")
st.title("Reliance Earnings Intelligence")
st.caption("Structured facts + hybrid RAG over Reliance earnings filings — every number cited.")

EXAMPLES = [
    "What was consolidated revenue in Q1 FY27?",
    "What was revenue growth YoY?",
    "Compare Jio and Retail revenue growth.",
    "What did management say about capex?",
    "Calculate YoY revenue growth.",
]


@st.cache_resource
def _provider():
    return SentenceTransformerProvider()


@st.cache_resource
def _bm25_index(mtime: float):
    return BM25Index.load(get_settings().bm25_path)


def _retriever(session, method: str):
    sem = SemanticRetriever(_provider(), LocalVectorStore(session))
    bm25_path = get_settings().bm25_path
    bm25 = None
    if bm25_path.exists():
        bm25 = BM25Retriever(_bm25_index(bm25_path.stat().st_mtime), session)
    if method == "bm25" and bm25 is None:
        st.warning("BM25 index not built yet — falling back to semantic.")
        method = "semantic"
    if method == "semantic":
        return sem
    if bm25 is None:  # hybrid without BM25 degrades to semantic
        return sem
    return HybridRetriever(sem, bm25, reranker=PassthroughReranker())


def _counts():
    session = get_session()
    try:
        return {
            "facts": count_facts(session),
            "chunks": count_chunks(session),
            "embeddings": count_embeddings(session),
            "documents": len(load_manifest()),
        }
    finally:
        session.close()


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("Add a document")
    upload = st.file_uploader("Upload PDF / XBRL-HTML", type=["pdf", "html", "htm"])
    company = st.text_input("Company", "Reliance Industries Limited")
    ticker = st.text_input("Ticker", "RELIANCE")
    col_q, col_fy = st.columns(2)
    quarter = col_q.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
    fy = col_fy.text_input("Financial year", "FY27")
    doc_type = st.selectbox("Document type",
                            ["media_release", "financial_results", "xbrl",
                             "investor_presentation", "earnings_transcript",
                             "annual_report", "other"])
    report_type = st.selectbox("Reporting basis", ["consolidated", "standalone",
                                                   "not_applicable"])
    pub_date = st.date_input("Publication date", date(2026, 7, 17))
    if st.button("Ingest document", disabled=upload is None):
        suffix = Path(upload.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getvalue())
            tmp_path = Path(tmp.name)
        spec = DocumentSpec(
            company=company, ticker=ticker or "RELIANCE",
            financial_year=fy, quarter=quarter, period=f"{quarter} {fy}",
            document_type=doc_type, report_type=report_type,
            source_url=f"user-upload://{upload.name}",
            filename=Path(upload.name).name,
            publication_date=pub_date.isoformat(),
        )
        try:
            record, created = register_local_file(spec, tmp_path)
            st.success(f"{'Ingested' if created else 'Already registered'}: "
                       f"{record.filename} ({record.file_size:,} bytes)")
        except Exception as exc:  # noqa: BLE001 — show, don't crash
            st.error(f"Ingestion failed: {exc}")

    if st.button("Process documents (extract + index)"):
        from app.extraction.pipeline import run_extraction
        from app.retrieval.pipeline import run_indexing

        with st.spinner("Extracting facts…"):
            summary = run_extraction()
        st.write(f"Facts in DB: {summary['total_facts']}")
        with st.spinner("Chunking + embedding + BM25… (first run downloads the model)"):
            idx = run_indexing()
        st.write(f"Chunks: {idx['total_chunks']}, embeddings: {idx['total_embeddings']}")
        st.success("Done.")

    st.divider()
    st.header("Question scope")
    scope = st.radio("Reporting basis", ["Auto (from question)", "consolidated", "standalone"],
                     help="Auto lets the router decide; otherwise the basis is prepended "
                          "to your question transparently.")
    method = st.selectbox("Retrieval", ["hybrid", "semantic", "bm25"])
    k = st.slider("Top-k chunks", 3, 12, 6)

    st.divider()
    counts = _counts()
    st.metric("Documents", counts["documents"])
    st.metric("Structured facts", counts["facts"])
    st.metric("Chunks / embeddings", f"{counts['chunks']} / {counts['embeddings']}")

# ------------------------------------------------------------------- main ---
st.subheader("Ask a question")
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(ex, key=f"ex{i}"):
        st.session_state["q"] = ex

question = st.text_input("Ask a question…", key="q",
                         placeholder="What was consolidated revenue in Q1 FY27?")
ask = st.button("Ask", type="primary")

if ask and question:
    effective = question
    if scope != "Auto (from question)":
        basis = scope
        effective = f"On a {basis} basis: {question}"
    session = get_session()
    try:
        with st.spinner("Retrieving facts + commentary…"):
            try:
                answer = answer_question(effective, session, _retriever(session, method),
                                         llm=get_llm_provider(), k=k)
            except Exception as exc:  # noqa: BLE001 — show, don't crash
                st.error(f"Couldn't answer that question: {exc}")
                st.stop()
    finally:
        session.close()
    st.markdown(f"**Route:** `{answer.route.query_class.value}` · **Mode:** {answer.mode} "
                f"· **Basis:** {answer.basis}")
    st.markdown(answer.text)

    if answer.calculations:
        with st.expander("Calculations (engine outputs with inputs)", expanded=False):
            for c in answer.calculations:
                st.write(f"**{c.metric}** = `{c.value} {c.unit}` (basis: {c.basis})")
                st.json(c.inputs)

    if answer.pack.chunks:
        with st.expander("Retrieved chunks (scores + provenance)", expanded=False):
            for h in answer.pack.chunks:
                st.write(f"**{h.score:.3f}** [{h.source}] {h.chunk.source_location}")
                st.caption(h.chunk.text[:400])

st.divider()
st.subheader("Indexed documents")
manifest = load_manifest()
if manifest:
    st.table(manifest)
else:
    st.warning("No documents ingested yet. Run: python scripts/ingest_q1fy27.py")
