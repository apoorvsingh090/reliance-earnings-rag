"""Milestone 6 tests: router, template answers, providers, prompt grounding."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.llm.answers import (
    SYSTEM_PROMPT,
    answer_question,
    build_llm_prompt,
    prior_quarter_period,
    prior_year_period,
)
from app.llm.providers import AnthropicProvider, OpenAIProvider, get_llm_provider
from app.llm.router import QueryClass, route
from app.models.database import get_session
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import HashEmbeddingProvider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.retrievers import BM25Retriever, PassthroughReranker, SemanticRetriever
from app.retrieval.vector_store import LocalVectorStore

needs_index = pytest.mark.skipif(
    not get_settings().bm25_path.exists(), reason="M3 index not built"
)


def _hybrid(session):
    # Hash vectors at the REAL dim: template answers don't depend on semantic
    # quality (facts/calcs come from the DB), and this keeps tests offline.
    sem = SemanticRetriever(HashEmbeddingProvider(dim=384), LocalVectorStore(session))
    bm25 = BM25Retriever(BM25Index.load(get_settings().bm25_path), session)
    return HybridRetriever(sem, bm25, reranker=PassthroughReranker())


# --- router ---------------------------------------------------------------------

def test_spec_examples_classify_exactly():
    assert route("What was consolidated revenue in Q1 FY27?").query_class == QueryClass.FINANCIAL_FACT
    assert route("What was revenue growth YoY?").query_class == QueryClass.CALCULATION
    assert route("Why did margins decline?").query_class == QueryClass.QUALITATIVE
    assert route("Compare Jio and Retail revenue growth.").query_class == QueryClass.SEGMENT_ANALYSIS
    assert route("What did management say about capex?").query_class == QueryClass.GUIDANCE


def test_router_slots_and_defaults():
    r = route("What was standalone revenue in Q1 FY27?")
    assert r.report_type == "standalone" and r.periods == ["Q1 FY27"]
    assert r.metric_hints[0] == "revenue_from_operations"
    r = route("What was revenue in Q1 FY27?")
    assert r.report_type == "consolidated"  # house default
    r = route("What caused EBITDA margin to change between Q2 FY26 and Q1 FY27?")
    assert r.query_class == QueryClass.CROSS_QUARTER
    assert r.periods == ["Q2 FY26", "Q1 FY27"]
    r = route("Compare Jio and Retail revenue growth.")
    assert set(r.segments) == {"digital_services", "retail"}
    assert route("Hello there").query_class == QueryClass.GENERAL
    assert route("How has EBITDA margin changed over the last 5 quarters?").query_class == \
        QueryClass.CROSS_QUARTER


def test_period_helpers():
    assert prior_year_period("Q1 FY27") == "Q1 FY26"
    assert prior_quarter_period("Q1 FY27") == "Q4 FY26"
    assert prior_quarter_period("Q3 FY26") == "Q2 FY26"
    assert prior_year_period("FY26") is None


# --- template answers (offline) ---------------------------------------------------

@needs_index
def test_fact_answer_with_citation():
    session = get_session()
    try:
        a = answer_question("What was consolidated revenue in Q1 FY27?", session, _hybrid(session))
        assert a.mode == "template" and a.basis == "consolidated"
        assert "311,850" in a.text and "[NSE Q1 FY27 Consolidated XBRL]" in a.text
    finally:
        session.close()


@needs_index
def test_standalone_basis_stated():
    session = get_session()
    try:
        a = answer_question("What was standalone revenue in Q1 FY27?", session, _hybrid(session))
        assert "166,013" in a.text and "**standalone** basis" in a.text
        assert "311,850" not in a.text  # no silent mixing
    finally:
        session.close()


@needs_index
def test_calc_answer_shows_work():
    session = get_session()
    try:
        a = answer_question("What was revenue growth YoY?", session, _hybrid(session))
        assert "25.41%" in a.text and "248,660" in a.text
        assert a.calculations and a.calculations[0].basis == "consolidated"
    finally:
        session.close()


@needs_index
def test_missing_fact_is_honest():
    session = get_session()
    try:
        a = answer_question("What was total debt in Q1 FY27?", session, _hybrid(session))
        assert "don't have" in a.text
        assert "311,850" not in a.text
    finally:
        session.close()


@needs_index
def test_qualitative_and_guidance_return_cited_commentary():
    session = get_session()
    try:
        a = answer_question("Why did margins decline?", session, _hybrid(session))
        assert "Management commentary" in a.text and "Sources used" in a.text
        g = answer_question("What did management say about capex?", session, _hybrid(session))
        assert g.route.query_class == QueryClass.GUIDANCE
        assert "Sources used" in g.text
    finally:
        session.close()


@needs_index
def test_cross_quarter_and_segment():
    session = get_session()
    try:
        a = answer_question("Compare operating margin between Q1 FY26 and Q1 FY27",
                            session, _hybrid(session))
        assert "9.5" in a.text and "10.6" in a.text
        s = answer_question("Compare Jio and Retail revenue growth", session, _hybrid(session))
        assert "90,409" in s.text and "46,900" in s.text
    finally:
        session.close()


# --- providers ----------------------------------------------------------------------

def test_openai_provider_parses(monkeypatch):
    import httpx

    corp = {"choices": [{"message": {"content": "hello"}}], "model": "m", "usage": {}}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return corp

    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    ans = OpenAIProvider(api_key="k").complete("sys", "user")
    assert ans.text == "hello"


def test_anthropic_provider_parses(monkeypatch):
    import httpx

    corp = {"content": [{"text": "hi"}, {"text": " there"}], "model": "m", "usage": {}}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return corp

    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    assert AnthropicProvider(api_key="k").complete("sys", "user").text == "hi there"


def test_no_key_means_template_mode(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_llm_provider() is None


def test_prompt_is_grounded_and_llm_failure_falls_back():
    assert "ONLY the provided context" in SYSTEM_PROMPT
    assert "Never mix standalone and consolidated" in SYSTEM_PROMPT

    session = get_session()
    try:
        a = answer_question("What was consolidated revenue in Q1 FY27?", session,
                            _hybrid(session))

        prompt = build_llm_prompt("q", a.route, a.pack, a.calculations)
        assert "311850" in prompt and "NSE Q1 FY27 consolidated XBRL" in prompt

        class Broken:
            name = "broken"

            def complete(self, system, user):
                raise RuntimeError("down")

        b = answer_question("What was consolidated revenue in Q1 FY27?", session,
                            _hybrid(session), llm=Broken())
        assert "311,850" in b.text and "LLM unavailable" in b.text
    finally:
        session.close()
