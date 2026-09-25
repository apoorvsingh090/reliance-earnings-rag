"""Context builder: the single pack M6 (router + LLM) will consume.

Today it fuses hybrid chunk retrieval with explicitly requested structured
facts (the seam the M6 query router will drive). Every attached item carries
a citation; the pack dedupes them in first-seen order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.financial.query import get_fact
from app.retrieval.citations import Citation, cite_chunk, cite_fact, dedupe_citations
from app.retrieval.interfaces import RetrievedChunk


@dataclass
class ContextPack:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    facts: list = field(default_factory=list)  # FinancialFactRow
    citations: list[Citation] = field(default_factory=list)


def build_context(
    query: str,
    retriever,
    session,
    k: int = 8,
    filters: dict | None = None,
    fact_requests: list[dict] | None = None,
) -> ContextPack:
    pack = ContextPack(query=query)
    pack.chunks = retriever.search(query, k=k, filters=filters)
    citations = [cite_chunk(h.chunk) for h in pack.chunks]
    for req in fact_requests or []:
        row = get_fact(
            session,
            req["metric"],
            req["period"],
            req.get("report_type", "consolidated"),
            req.get("segment"),
        )
        if row is None:
            continue
        pack.facts.append(row)
        citations.append(cite_fact(row))
    pack.citations = dedupe_citations(citations)
    return pack
