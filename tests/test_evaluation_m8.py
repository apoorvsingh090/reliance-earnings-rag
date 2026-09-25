"""Milestone 8 tests: dataset contract, scorer logic, and a measured subset run."""

from __future__ import annotations

import json

import pytest

from app.evaluation.evaluate import _load_dataset, evaluate_one, run_dataset
from app.llm.router import QueryClass
from app.models.database import get_session
from app.retrieval.factory import default_retriever

COVERAGE = {
    "direct fact": {"fact"},
    "numerical calculations": {"calculation"},
    "yoy": {"yoy"},
    "qoq": {"qoq"},
    "segment": {"segment"},
    "commentary": {"commentary"},
    "ambiguous": {"ambiguous"},
    "standalone_vs_consolidated": {"standalone_vs_consolidated"},
    "missing": {"missing"},
    "cross_quarter": {"cross_quarter"},
}


def test_dataset_has_20_plus_questions_with_full_coverage():
    dataset = _load_dataset()
    assert len(dataset) >= 20
    cats = {q["category"] for q in dataset}
    for label, required in COVERAGE.items():
        assert required <= cats, f"no questions for {label}"
    for q in dataset:
        assert q["question"] and q["category"] and q["id"]
        assert q["route"] in {c.value for c in QueryClass}
        assert isinstance(q.get("facts", []), list)
        assert isinstance(q.get("citations", []), list)
        assert isinstance(q.get("contains", []), list)
        for f in q.get("facts", []):
            assert {"metric", "period", "report_type", "value"} <= set(f)


def test_scorer_pass_fail_and_absence(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class FakeChunk:
        text: str = "kgd6 production averaged 24.8 mmscmd"
        section: str = ""

    @dataclass
    class FakeHit:
        chunk: object = None

        def __post_init__(self):
            self.chunk = FakeChunk()

    @dataclass
    class FakeFact:
        metric: str = "revenue_from_operations"
        period: str = "Q1 FY27"
        report_type: str = "consolidated"
        segment: object = None
        value: float = 311850.0

    @dataclass
    class FakeCite:
        short: str = "[NSE Q1 FY27 Consolidated XBRL]"

    @dataclass
    class FakePack:
        facts: list
        chunks: list
        citations: list

    @dataclass
    class FakeRoute:
        query_class: object = QueryClass.FINANCIAL_FACT

    @dataclass
    class FakeAnswer:
        text: str = "Revenue **311,850 crore** [NSE Q1 FY27 Consolidated XBRL]"
        route: object = None
        pack: object = None
        calculations: list = None

        def __post_init__(self):
            self.route = FakeRoute()
            self.pack = FakePack([FakeFact()], [FakeHit()], [FakeCite()])
            self.calculations = []

    import app.evaluation.evaluate as EV

    monkeypatch.setattr(EV, "answer_question", lambda *a, **k: FakeAnswer())
    entry = {"id": "T1", "category": "fact", "route": "FINANCIAL_FACT",
             "question": "q",
             "facts": [{"metric": "revenue_from_operations", "period": "Q1 FY27",
                        "report_type": "consolidated", "value": 311850.0}],
             "keywords": ["kgd6"],
             "citations": ["[NSE Q1 FY27 Consolidated XBRL]"],
             "contains": ["311,850"]}
    res = evaluate_one(entry, session=None, retriever=None)
    assert res.passed and res.facts_recall == 1.0 and res.keywords_recall == 1.0

    bad = dict(entry, facts=[{"metric": "revenue_from_operations", "period": "Q1 FY27",
                              "report_type": "consolidated", "value": 1.0}])
    res = evaluate_one(bad, session=None, retriever=None)
    assert not res.passed and res.facts_recall == 0.0

    leak = dict(entry, absent_scoped=["311,850"])
    res = evaluate_one(leak, session=None, retriever=None)
    assert not res.passed and res.absent_ok is False


def test_scorer_survives_answer_crash(monkeypatch):
    import app.evaluation.evaluate as EV

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(EV, "answer_question", boom)
    res = evaluate_one({"id": "T9", "category": "x", "question": "q"}, None, None)
    assert not res.passed and res.notes


def test_measured_subset_passes_with_real_index():
    session = get_session()
    try:
        dataset = [q for q in _load_dataset() if q["id"] in {"F1", "C1", "S1", "N1", "X1", "F2", "B2"}]
        report = run_dataset(dataset=dataset, session=session,
                             retriever=default_retriever(session))
        assert report["passed"] == report["total"] == 7
        assert report["metrics"]["answer_correctness"] == 1.0
        assert json.dumps(report)  # serializable for the saved artifact
    finally:
        session.close()
