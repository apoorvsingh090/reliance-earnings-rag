"""Evaluation runner (spec section 19): retrieval recall, citation accuracy,
numerical accuracy, answer correctness — measured, never claimed.

Each dataset entry declares machine-checkable expectations. Scores are strict
per question (1.0 or the fraction matched); the report aggregates means plus
a pass rate. Template mode keeps runs deterministic and offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.llm.answers import answer_question

DATASET_PATH = Path(__file__).with_name("dataset.json")


@dataclass
class QuestionResult:
    id: str
    category: str
    route_ok: bool | None = None
    facts_recall: float | None = None
    keywords_recall: float | None = None
    citations_acc: float | None = None
    numeric_acc: float | None = None
    absent_ok: bool | None = None
    passed: bool = False
    notes: list[str] = field(default_factory=list)


def _load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scoped_head(text: str) -> str:
    """Facts/calculations live above the commentary section; basis-isolation
    (absent_*) checks apply there, since quoted commentary carries its own
    citations and may legitimately mention other bases."""
    return text.split("**Management commentary")[0]


def evaluate_one(entry: dict, session, retriever, k: int = 6) -> QuestionResult:
    res = QuestionResult(id=entry["id"], category=entry.get("category", "?"))
    try:
        answer = answer_question(entry["question"], session, retriever, llm=None, k=k)
    except Exception as exc:  # noqa: BLE001 — a crash is a failure, not a skip
        res.notes.append(f"answer crashed: {exc}")
        return res

    checks: list[bool] = []

    if entry.get("route"):
        res.route_ok = answer.route.query_class.value == entry["route"]
        checks.append(res.route_ok)
        if not res.route_ok:
            res.notes.append(f"route {answer.route.query_class.value} != {entry['route']}")

    exp_facts = entry.get("facts", [])
    if exp_facts:
        found = 0
        for exp in exp_facts:
            hit = next(
                (f for f in answer.pack.facts
                 if f.metric == exp["metric"] and f.period == exp["period"]
                 and f.report_type == exp["report_type"]
                 and (f.segment or None) == exp.get("segment")),
                None,
            )
            if hit is not None and abs(hit.value - exp["value"]) <= exp.get("tol", 0.51):
                found += 1
            else:
                res.notes.append(f"fact missing/off: {exp['metric']} {exp['period']}")
        res.facts_recall = found / len(exp_facts)
        checks.append(res.facts_recall == 1.0)

    keywords = entry.get("keywords", [])
    if keywords:
        blob = " ".join((h.chunk.text + " " + h.chunk.section) for h in answer.pack.chunks).lower()
        hit = sum(1 for kw in keywords if kw.lower() in blob)
        res.keywords_recall = hit / len(keywords)
        checks.append(res.keywords_recall == 1.0)
        if res.keywords_recall < 1.0:
            res.notes.append("keyword(s) absent from retrieved chunks")

    exp_cites = entry.get("citations", [])
    if exp_cites:
        shorts = [c.short for c in answer.pack.citations]
        hit = sum(1 for c in exp_cites if any(c in s for s in shorts))
        res.citations_acc = hit / len(exp_cites)
        checks.append(res.citations_acc == 1.0)
        if res.citations_acc < 1.0:
            res.notes.append(f"citations missing: have={shorts}")

    exp_nums = entry.get("contains", [])
    if exp_nums:
        hit = sum(1 for n in exp_nums if n in answer.text)
        res.numeric_acc = hit / len(exp_nums)
        checks.append(res.numeric_acc == 1.0)
        if res.numeric_acc < 1.0:
            res.notes.append("expected number(s) absent from answer text")

    scoped_absent = entry.get("absent_scoped", [])
    if scoped_absent:
        head = _scoped_head(answer.text)
        ok = all(a not in head for a in scoped_absent)
        res.absent_ok = ok
        checks.append(ok)
        if not ok:
            res.notes.append("scoped absence violated (basis leak in facts/calcs)")

    res.passed = bool(checks) and all(checks)
    return res


def run_dataset(dataset: list[dict] | None = None, session=None, retriever=None,
                k: int = 6) -> dict:
    from app.models.database import get_session

    own_session = session is None
    session = session or get_session()
    try:
        dataset = dataset if dataset is not None else _load_dataset()
        results = [evaluate_one(e, session, retriever, k) for e in dataset]
    finally:
        if own_session:
            session.close()

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    recall_vals = []
    for r in results:
        parts = [v for v in (r.facts_recall, r.keywords_recall) if v is not None]
        if parts:
            recall_vals.append(sum(parts) / len(parts))
    report = {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "metrics": {
            "retrieval_recall": mean(recall_vals),
            "citation_accuracy": mean([r.citations_acc for r in results]),
            "numerical_accuracy": mean([r.numeric_acc for r in results]),
            "answer_correctness": (sum(1 for r in results if r.passed) / len(results)
                                   if results else None),
        },
        "by_category": {},
        "questions": [
            {"id": r.id, "category": r.category, "passed": r.passed,
             "route_ok": r.route_ok, "facts_recall": r.facts_recall,
             "keywords_recall": r.keywords_recall, "citations_acc": r.citations_acc,
             "numeric_acc": r.numeric_acc, "absent_ok": r.absent_ok, "notes": r.notes}
            for r in results
        ],
    }
    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.passed)
    report["by_category"] = {c: {"passed": sum(v), "total": len(v)} for c, v in cats.items()}
    return report
