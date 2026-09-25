"""Chunk builders: PDF pages/sections/tables + XBRL notes/descriptors.

Core rule (spec section 2): numbers that live in the structured database are
NOT duplicated into vectors.
- XBRL tables (P&L, segments) -> skipped; only the filing descriptor + free-text
  notes ("Textual Information") become chunks.
- PDF annexure pages ("Page Y of 14") -> data rows skipped; headings, notes,
  and ratio formulae kept.
- PDF body pages -> full text (narrative + subsidiary KPIs, which exist nowhere
  else) plus separate table chunks.
"""

from __future__ import annotations

import logging
import re

from app.extraction.pdf import _is_value_token
from app.extraction.xbrl import XBRLParse
from app.models.chunk import Chunk
from app.models.document import DocumentRecord, ReportType

log = logging.getLogger(__name__)

MAX_CHARS = 1500
OVERLAP = 200
MIN_CHUNK_CHARS = 50


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_boilerplate(lines: list[str], printed_label: str) -> list[str]:
    """Drop the repeated per-page header (everything up to the Page-label line)."""
    for i, line in enumerate(lines):
        if printed_label and line.strip() == printed_label:
            return lines[i + 1:]
    # Fallback: header block contains the registered-office address.
    cut = 0
    for i, line in enumerate(lines[:12]):
        if "Registered Office" in line or "Corporate Communications" in line:
            cut = i
    return lines[cut + 1:] if cut else lines


def split_long(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary <= start + max_chars // 2:
                boundary = text.rfind(". ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        # Overlap windows, but always advance: a boundary-shrunk window that is
        # smaller than the overlap would otherwise crawl one char per step.
        start = end - overlap if end - overlap > start else end
    return [p for p in parts if len(p) >= MIN_CHUNK_CHARS]


def _emit(
    chunks: list[Chunk],
    record: DocumentRecord,
    page_number: int | None,
    printed_label: str,
    section: str,
    text: str,
    chunk_type: str = "text",
) -> None:
    text = text.strip()
    if len(text) < MIN_CHUNK_CHARS:
        return
    base = len(chunks)
    for j, part in enumerate(split_long(text)):
        idx = base + j
        chunk_id = Chunk.make_id(record.document_hash, page_number, section, idx, part)
        if record.document_type.value == "xbrl":
            loc = f"NSE Q1 FY27 {record.report_type.value} XBRL, notes"
        elif page_number:
            loc = f"RIL Q1 FY27 media release, PDF p. {page_number}"
            if section:
                loc += f" ({section[:60]})"
        else:
            loc = "RIL Q1 FY27 media release"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=record.document_hash,
                page_number=page_number,
                printed_label=printed_label,
                section=section[:200],
                chunk_type=chunk_type,  # type: ignore[arg-type]
                text=part,
                financial_year=record.financial_year,
                quarter=record.quarter,
                period=record.period,
                document_type=record.document_type.value,
                report_type=record.report_type,
                source_location=loc,
            )
        )


def _is_annexure_page(printed_label: str) -> bool:
    return bool(printed_label) and "of 14" in printed_label


def chunk_pdf_page(
    chunks: list[Chunk],
    record: DocumentRecord,
    page_number: int,
    printed_label: str,
    text: str,
    headings: list[str],
    table_texts: list[str],
) -> None:
    lines = strip_boilerplate(text.splitlines(), printed_label)
    heading_set = {_norm(h).lower() for h in headings}
    section = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            _emit(chunks, record, page_number, printed_label, section, "\n".join(buf))
            buf = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if _norm(s).lower() in heading_set:
            flush()
            section = _norm(s)
            continue
        if _is_annexure_page(printed_label) and (_is_value_token(s) or _looks_like_data_label(s)):
            continue  # structured annexure data row -> DB only, never vectors
        buf.append(s)
    flush()

    if not _is_annexure_page(printed_label):
        for t in table_texts:
            if len(t.strip()) >= MIN_CHUNK_CHARS:
                _emit(chunks, record, page_number, printed_label,
                      section or "tables", t.strip(), chunk_type="table")


def _looks_like_data_label(s: str) -> bool:
    """Annexure row labels end up followed by value lines; catch the common ones."""
    low = _norm(s).lower()
    return low.startswith((
        "value of sales", "less: gst", "revenue from operations", "other income",
        "total income", "cost of materials", "purchases of stock", "changes in inventories",
        "excise duty", "employee benefits", "finance costs", "depreciation",
        "other expenses", "total expenses", "profit before tax", "current tax",
        "deferred tax", "profit after tax", "share of profit", "paid-up equity",
        "other equity", "capital redemption", "net worth", "debt service",
        "interest service", "debt equity", "current ratio", "long-term debt",
        "current liability", "total debts", "debtors turnover", "inventory turnover",
        "operating margin", "net profit margin", "segment value", "segment results",
        "segment assets", "segment liabilities", "- oil", "- retail", "- digital",
        "- others", "- unallocated", "gross value", "earnings per equity",
        "basic (in", "diluted (in",
    ))


def chunk_xbrl(chunks: list[Chunk], record: DocumentRecord, parsed: XBRLParse) -> None:
    descriptor = (
        f"{parsed.company_name or record.company} — {record.period} "
        f"{record.report_type.value} financial results filing (NSE iXBRL). "
        f"Currency {parsed.currency}; amounts in {parsed.amount_unit}; "
        f"{parsed.audit_status or 'Unaudited'}; reporting period "
        f"{parsed.period_start} to {parsed.period_end}."
    )
    _emit(chunks, record, None, "", "filing descriptor", descriptor, chunk_type="descriptor")
    for table in parsed.tables:
        for row in table.rows:
            joined = " ".join(row).strip()
            if len(joined) >= 200 and ("notes" in joined.lower() or "gst" in joined.lower()
                                       or "crore" in joined.lower() or "reserve" in joined.lower()):
                _emit(chunks, record, None, "", "notes to financial results",
                      joined, chunk_type="text")


def serialize_pdf_tables(page) -> list[str]:
    """Best-effort table serializations; empty when the page has no clean tables."""
    try:
        tables = list(page.find_tables())
    except Exception:  # noqa: BLE001
        return []
    out = []
    for t in tables:
        try:
            rows = t.extract()
        except Exception:  # noqa: BLE001
            continue
        lines = [" | ".join((c or "").strip() for c in r).strip(" |") for r in rows]
        text = "\n".join(l for l in lines if l)
        if len(text) >= MIN_CHUNK_CHARS:
            out.append(text)
    return out
