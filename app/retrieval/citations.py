"""Citation assembly (spec section 14).

Every factual claim traces to a short, human-readable citation plus the full
stored source location. Rules enforced here, not by the LLM:

- PDF chunks/facts cite the REAL page number: ``[Q1 FY27 Media Release, p. 6]``.
- XBRL/HTML sources have no pages: ``[NSE Q1 FY27 Consolidated XBRL]`` — a page
  number is never invented for them.
- Citations describe SOURCES, never values; the caller decides whether the
  claim is a reported fact, a calculation, or a management statement (M6).

Kinds: "narrative" (media body text), "table" (subsidiary KPI tables),
"filing" (XBRL notes/descriptor), "fact" (structured DB rows).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Citation:
    short: str  # e.g. "[Q1 FY27 Media Release, p. 6]"
    detail: str  # full stored source_location
    page: int | None
    kind: str  # narrative | table | filing | fact
    index: int = 0  # 1-based position in the context pack


def cite_chunk(chunk) -> Citation:
    """chunk: app.models.chunk.Chunk (or ChunkRow with the same attributes)."""
    period = chunk.period or "Q1 FY27"
    if chunk.document_type == "xbrl":
        basis = (chunk.report_type.value if hasattr(chunk.report_type, "value")
                 else str(chunk.report_type)).capitalize()
        short = f"[NSE {period} {basis} XBRL]"
        kind = "filing"
        page = None
    else:
        page = chunk.page_number
        short = f"[{period} Media Release, p. {page}]" if page else f"[{period} Media Release]"
        kind = "table" if chunk.chunk_type == "table" else "narrative"
    return Citation(short=short, detail=chunk.source_location, page=page, kind=kind)


def cite_fact(fact) -> Citation:
    """fact: FinancialFactRow or FinancialFact. Page recovered only if stored."""
    loc = fact.source_location or ""
    report_type = fact.report_type.value if hasattr(fact.report_type, "value") else fact.report_type
    period = fact.period or "Q1 FY27"
    method = (fact.extraction_method or "")
    if method.startswith("xbrl"):
        short = f"[NSE {period} {str(report_type).capitalize()} XBRL]"
        return Citation(short=short, detail=loc, page=None, kind="fact")
    m = re.search(r"PDF p\. (\d+)", loc)
    page = int(m.group(1)) if m else None
    short = f"[{period} Media Release annexure, p. {page}]" if page else f"[{period} Media Release annexure]"
    return Citation(short=short, detail=loc, page=page, kind="fact")


def dedupe_citations(citations: list[Citation]) -> list[Citation]:
    """Dedupe by short form, keep first-seen order, assign 1-based indices."""
    seen: dict[str, Citation] = {}
    for c in citations:
        if c.short not in seen:
            seen[c.short] = c
    out = list(seen.values())
    for i, c in enumerate(out, start=1):
        c.index = i
    return out


def format_sources_block(citations: list[Citation]) -> str:
    """Expandable 'Sources used' rendering for the UI / API."""
    lines = []
    for c in citations:
        lines.append(f"[{c.index}] {c.short} — {c.detail} ({c.kind})")
    return "\n".join(lines)
