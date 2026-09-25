"""CLI: chunk the Q1 FY27 documents, embed, and build the BM25 index."""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

from app.retrieval.pipeline import run_indexing

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    summary = run_indexing()
    print(f"Documents: {summary['documents']}, chunks: {summary['total_chunks']}, "
          f"embeddings: {summary['total_embeddings']}, bm25_rebuilt={summary['bm25_rebuilt']}")
    for filename, counts in summary["per_document"].items():
        print(f"  - {filename}: {counts['chunks']} chunks "
              f"(+{counts['new']} new, +{counts['embedded']} embedded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
