"""CLI: parse the Q1 FY27 seed documents into the structured fact database."""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

from app.extraction.pipeline import run_extraction

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    summary = run_extraction()
    print(f"Documents: {summary['documents']}, total facts in DB: {summary['total_facts']}")
    for filename, counts in summary["per_document"].items():
        print(f"  - {filename}: +{counts['created']} new, {counts['reused']} reused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
