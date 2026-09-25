"""Media-release PDF extraction (PyMuPDF).

Two jobs:
1. Page-aware text capture — every page keeps its real PDF page number, the
   printed label ("Page X of 20" / annexure "Page Y of 14"), headings
   (font-size outliers), and char counts. Tables are intentionally NOT taken
   from find_tables (merged cells make it unreliable for this document);
   the annexure tables have a rigid label-then-values text layout that parses
   deterministically (see extract_annexure_facts).
2. OCR-fallback detection — pages with suspiciously little extractable text
   are flagged (needs_ocr) instead of silently accepted. The fallback itself
   (ocrmypdf/tesseract) is a documented M2-future hook, not a silent pass.

Annexure columns (pp. 21-22, 25-26 of the PDF): Q1 FY27 | Q4 FY26 | Q1 FY26 | FY26.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.models.document import ReportType
from app.models.financial import FinancialFact

log = logging.getLogger(__name__)

MIN_CHARS_PER_PAGE = 200  # below this a page is flagged needs_ocr

# Annexure value-column periods, in order.
ANNEXURE_PERIODS: list[tuple[str, str | None, str, str]] = [
    ("Q1 FY27", "Q1", "FY27", "quarter"),
    ("Q4 FY26", "Q4", "FY26", "quarter"),
    ("Q1 FY26", "Q1", "FY26", "quarter"),
    ("FY26", None, "FY26", "year"),
]


@dataclass
class PDFPage:
    pdf_page: int  # 1-indexed position in the PDF file (citation truth)
    printed_label: str = ""  # e.g. "Page 4 of 20" as printed on the page
    text: str = ""
    headings: list[str] = field(default_factory=list)
    char_count: int = 0
    needs_ocr: bool = False


@dataclass
class PDFParse:
    filename: str
    pages: list[PDFPage] = field(default_factory=list)

    @property
    def flagged_pages(self) -> list[int]:
        return [p.pdf_page for p in self.pages if p.needs_ocr]


def _printed_label(text: str) -> str:
    m = re.search(r"Page\s+\d+\s+of\s+\d+", text)
    return m.group(0) if m else ""


def _headings_for_page(page) -> list[str]:
    """Headings = short lines set materially larger than the body size."""
    try:
        info = page.get_text("dict")
    except Exception:  # noqa: BLE001
        return []
    sizes: dict[float, int] = {}
    lines: list[tuple[float, str, bool]] = []
    for block in info.get("blocks", {} if False else []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            size = round(max(s["size"] for s in spans), 1)
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            bold = any("bold" in s.get("font", "").lower() for s in spans)
            lines.append((size, text, bold))
            sizes[size] = sizes.get(size, 0) + len(text)
    if not lines:
        return []
    body = max(sizes, key=lambda s: sizes[s])
    headings = [
        t for size, t, _ in lines
        if size >= body + 2.0 and len(t) <= 160 and len(t) >= 4
    ]
    # De-duplicate while preserving order.
    seen, out = set(), []
    for h in headings:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out[:20]


def parse_pdf(path: Path) -> PDFParse:
    import pymupdf

    doc = pymupdf.open(str(path))
    result = PDFParse(filename=path.name)
    for i, page in enumerate(doc):
        text = page.get_text()
        char_count = len(text.strip())
        result.pages.append(
            PDFPage(
                pdf_page=i + 1,
                printed_label=_printed_label(text),
                text=text,
                headings=_headings_for_page(page),
                char_count=char_count,
                needs_ocr=char_count < MIN_CHARS_PER_PAGE,
            )
        )
    doc.close()
    if result.flagged_pages:
        log.warning("%s: %d page(s) flagged needs_ocr: %s", path.name,
                    len(result.flagged_pages), result.flagged_pages)
    return result


# ---------------------------------------------------------------------------
# Annexure table parsing (text-line strategy)
# ---------------------------------------------------------------------------

_MISSING_TOKENS = {"-", "–", "—", "–", "nil"}


def _is_value_token(s: str) -> bool:
    s = s.strip().rstrip(",")
    if s.lower() in _MISSING_TOKENS or s == "":
        return False
    return bool(re.fullmatch(r"\(?[\d,]+(?:\.\d+)?\)?", s))


def _to_float(s: str) -> float | None:
    s = s.strip()
    if s.lower() in _MISSING_TOKENS or s == "":
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _collect_values(lines: list[str], start: int, n: int = 4, window: int = 10) -> list[float | None] | None:
    """Collect the next n numeric tokens after a label line.

    Skips multi-line label continuations (non-numeric lines) inside the window.
    Returns None if fewer than n value tokens are found (treat as no-match).
    """
    values: list[float | None] = []
    for line in lines[start + 1: start + 1 + window]:
        s = line.strip()
        if _is_value_token(s):
            values.append(_to_float(s))
            if len(values) == n:
                return values
        elif s.lower() in _MISSING_TOKENS or s in {"", "-"}:
            values.append(None)
            if len(values) == n:
                return values
        # else: label continuation line — skip
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# (match key, canonical metric, unit) — matched against the START of a line.
ANNEXURE_PNL: list[tuple[str, str, str]] = [
    ("value of sales & services (revenue)", "value_of_sales_services", "crore"),
    ("less: gst recovered", "gst_recovered", "crore"),
    ("revenue from operations", "revenue_from_operations", "crore"),
    ("other income", "other_income", "crore"),
    ("total income", "total_income", "crore"),
    ("cost of materials consumed", "cost_of_materials", "crore"),
    ("purchases of stock-in-trade", "purchases_stock_in_trade", "crore"),
    ("changes in inventories", "inventory_change", "crore"),
    ("excise duty", "excise_duty", "crore"),
    ("employee benefits expense", "employee_benefit", "crore"),
    ("finance costs", "finance_costs", "crore"),
    ("depreciation / amortisation and depletion expense", "depreciation", "crore"),
    ("other expenses", "other_expenses", "crore"),
    ("total expenses", "total_expenses", "crore"),
    ("profit before tax", "pbt", "crore"),
    ("current tax", "current_tax", "crore"),
    ("deferred tax", "deferred_tax", "crore"),
    ("profit after tax and share of profit", "pat_total", "crore"),  # must precede "profit after tax"
    ("profit after tax", "pat_continuing", "crore"),
    ("share of profit / (loss) of associates", "share_associates_jv", "crore"),
    ("paid-up equity share capital", "paid_up_capital", "crore"),
    ("other equity excluding revaluation reserve", "other_equity", "crore"),
    ("capital redemption reserve / debenture redemption reserve", "capital_redemption_reserve", "crore"),
    ("net worth (including retained earnings)", "net_worth", "crore"),
    ("debt service coverage ratio", "dscr", "ratio"),
    ("interest service coverage ratio", "iscr", "ratio"),
    ("debt equity ratio", "debt_equity_ratio", "ratio"),
    ("current ratio", "current_ratio", "ratio"),
    ("long-term debt to working capital", "long_term_debt_to_working_capital", "ratio"),
    ("current liability ratio", "current_liability_ratio", "ratio"),
    ("total debts to total assets", "total_debts_to_assets", "ratio"),
    ("debtors turnover", "debtors_turnover", "ratio"),
    ("inventory turnover", "inventory_turnover", "ratio"),
    ("operating margin (%)", "operating_margin", "percent"),
    ("net profit margin (%)", "net_profit_margin", "percent"),
]

# Rows inside the "Earnings per equity share" block; "a)"/"Basic (in ...)"
# sit on separate lines, so matching is gated on the in_eps state flag.

SEGMENT_ROWS: dict[str, tuple[str, str | None]] = {
    "- oil to chemicals (o2c)": ("segment", "o2c"),
    "- oil and gas": ("segment", "oil_and_gas"),
    "- retail": ("segment", "retail"),
    "- digital services": ("segment", "digital_services"),
    "- others": ("segment", "others"),
    "- unallocated": ("segment", "unallocated"),
    "gross value of sales and services": ("segment", "total"),
    "total segment profit before interest, tax and": ("segment", "total"),
    "total segment profit before interest and tax": ("segment", "total"),
    "total segment assets": ("segment", "total"),
    "total segment liabilities": ("segment", "total"),
}

# Reconciliation bridges inside segment pages — derivable, never stored as
# segment facts (the P&L tables remain the single source for company revenue).
SEGMENT_BRIDGES = (
    "value of sales & services",
    "revenue from operations",
    "less:",
)

# Annexure basis titles (the PDF holds consolidated AND standalone side by side).
BASIS_TITLES = {
    "unaudited consolidated financial results": ReportType.CONSOLIDATED,
    "unaudited standalone financial results": ReportType.STANDALONE,
    "consolidated segment information": ReportType.CONSOLIDATED,
    "standalone segment information": ReportType.STANDALONE,
}


def extract_annexure_facts(
    parsed: PDFParse,
    *,
    company: str,
    ticker: str,
    source_document_id: str,
) -> list[FinancialFact]:
    """Parse annexure P&L + segment tables from page text lines."""
    facts: list[FinancialFact] = []
    # Annexure = pages whose printed label says "of 14".
    annex_pages = [p for p in parsed.pages if p.printed_label and "of 14" in p.printed_label]
    if not annex_pages:
        annex_pages = [p for p in parsed.pages if p.pdf_page >= 21]
    lines: list[str] = []
    page_of_line: list[int] = []
    for p in annex_pages:
        for line in p.text.splitlines():
            lines.append(line)
            page_of_line.append(p.pdf_page)

    section: str | None = None  # segment section state: revenue|ebitda|ebit|assets|liab
    basis: ReportType | None = None  # annexure holds consolidated AND standalone
    in_eps = False  # set after the "Earnings per equity share" header line
    i = 0
    while i < len(lines):
        norm = _norm(lines[i])

        # --- reporting-basis titles reset section state (standalone/consol. leak guard) ---
        basis_hit = False
        for title, title_basis in BASIS_TITLES.items():
            if title in norm:
                basis = title_basis
                section = None
                in_eps = False
                basis_hit = True
                break
        if basis_hit:
            i += 1
            continue
        if basis is None:
            i += 1
            continue

        # --- segment section headers (bare titles, no reliable numbering) ---
        # Order matters: EBITDA before EBIT (closing paren keeps them distinct).
        if "segment value of sales" in norm and not norm.startswith("-"):
            section = "segment_revenue"
            i += 1
            continue
        if "segment results (ebitda)" in norm:
            section = "segment_ebitda"
            i += 1
            continue
        if "segment results (ebit)" in norm:
            section = "segment_ebit"
            i += 1
            continue
        if "segment assets" in norm and not norm.startswith("-") and "total" not in norm:
            section = "segment_assets"
            i += 1
            continue
        if "segment liabilities" in norm and not norm.startswith("-") and "total" not in norm:
            section = "segment_liabilities"
            i += 1
            continue

        # --- segment data rows ---
        if section is not None and norm.startswith(SEGMENT_BRIDGES):
            i += 1  # reconciliation bridge — derivable, not stored
            continue
        seg_hit = None
        for key, (kind, code) in SEGMENT_ROWS.items():
            if norm.startswith(key):
                # Avoid the company-level "Value of Sales & Services" P&L row
                # colliding with the segment total: only treat as segment row
                # while inside a segment section.
                if kind == "segment" and code == "total" and section not in {
                    "segment_revenue", "segment_ebitda", "segment_ebit",
                    "segment_assets", "segment_liabilities",
                }:
                    continue
                seg_hit = (code, key)
                break
        if seg_hit and section:
            code, _ = seg_hit
            if norm.startswith("less:"):
                i += 1
                continue
            metric = section
            values = _collect_values(lines, i)
            if values is not None:
                for (p, q, fy, ptype), v in zip(ANNEXURE_PERIODS, values):
                    if v is None:
                        continue
                    facts.append(_fact(company, ticker, fy, q, p, ptype, metric, v,
                                       "crore", code, basis, source_document_id, parsed.filename,
                                       page_of_line[i], lines[i].strip()))
                    # The segment-assets total IS the balance-sheet total, so it
                    # doubles as the company-level total_assets fact.
                    if metric == "segment_assets" and code == "total":
                        facts.append(_fact(company, ticker, fy, q, p, ptype, "total_assets", v,
                                           "crore", None, basis, source_document_id,
                                           parsed.filename, page_of_line[i],
                                           lines[i].strip()))
            i += 1
            continue

        # --- EPS block: "a)"/"Basic (in ...)" sit on separate lines ---
        if "earnings per equity share" in norm:
            in_eps = True
            i += 1
            continue
        if in_eps and (norm.startswith("basic (in") or norm.startswith("diluted (in")):
            metric = "eps_basic" if norm.startswith("basic (in") else "eps_diluted"
            values = _collect_values(lines, i)
            if values is not None:
                for (p, q, fy, ptype), v in zip(ANNEXURE_PERIODS, values):
                    if v is None:
                        continue
                    facts.append(_fact(company, ticker, fy, q, p, ptype, metric, v,
                                       "INR_per_share", None, basis, source_document_id,
                                       parsed.filename, page_of_line[i], lines[i].strip()))
            i += 1
            continue

        # --- headline P&L / ratio rows ---
        pnl_hit = None
        for key, metric, unit in ANNEXURE_PNL:
            if norm.startswith(key):
                # "profit after tax" must not swallow the longer "...and share..." row.
                if key == "profit after tax" and norm.startswith("profit after tax and share"):
                    continue
                pnl_hit = (metric, unit)
                break
        if pnl_hit:
            metric, unit = pnl_hit
            if metric == "paid_up_capital":
                in_eps = False  # EPS block ends where capital begins
            values = _collect_values(lines, i)
            if values is not None:
                for (p, q, fy, ptype), v in zip(ANNEXURE_PERIODS, values):
                    if v is None:
                        continue
                    facts.append(_fact(company, ticker, fy, q, p, ptype, metric, v,
                                       unit, None, basis, source_document_id, parsed.filename,
                                       page_of_line[i], lines[i].strip()))
            i += 1
            continue
        i += 1
    return facts


def _fact(company, ticker, fy, q, period, ptype, metric, value, unit, segment,
          report_type, source_document_id, filename, pdf_page, label) -> FinancialFact:
    basis_note = report_type.value
    return FinancialFact(
        company=company,
        ticker=ticker,
        financial_year=fy,
        quarter=q,
        period=period,
        period_type=ptype,  # type: ignore[arg-type]
        metric=metric,
        value=value,
        unit=unit,
        currency="INR",
        segment=segment,
        report_type=report_type,
        source_document_id=source_document_id,
        source_location=f"RIL Q1 FY27 media release {basis_note} annexure, PDF p. {pdf_page}: {label[:60]}",
        extraction_method="pdf_annexure_table",
    )
