"""CLI: run the M8 evaluation set and print measured scores."""

from __future__ import annotations

import json
import logging
import sys

sys.path.insert(0, ".")

from app.config import get_settings
from app.evaluation.evaluate import run_dataset
from app.models.database import get_session
from app.retrieval.factory import default_retriever

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    session = get_session()
    try:
        report = run_dataset(session=session, retriever=default_retriever(session))
    finally:
        session.close()

    print(f"PASSED {report['passed']}/{report['total']}")
    for name, value in report["metrics"].items():
        print(f"  {name}: {value:.3f}" if value is not None else f"  {name}: n/a")
    print("  by category:")
    for cat, counts in sorted(report["by_category"].items()):
        print(f"    {cat}: {counts['passed']}/{counts['total']}")
    for q in report["questions"]:
        if not q["passed"]:
            print(f"  FAIL {q['id']}: {'; '.join(q['notes'])}")

    out = get_settings().processed_dir / "eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
