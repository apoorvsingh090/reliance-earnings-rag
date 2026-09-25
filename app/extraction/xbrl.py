"""NSE iXBRL (rendered HTML) parser.

The NSE filings are plain-HTML renderings: 9 tables, no inline ix: tags.
- Table 0: general info (company, dates, stated reporting basis, units)
- Table 1: main P&L (values in Rs Lakhs, per the in-document "Amount in (Lakhs)")
- Table 4: segment disclosure (revenue / result / assets / liabilities)
- Table 6: OCI detail
Everything else (ratio shells, auditor info) is captured as metadata only.

Unit handling: all money cells are divided by 100 to canonical INR crore.
EPS / ratios / face value are stored as-is with their own units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

from app.models.document import ReportType
from app.models.financial import FinancialFact

LAKH_TO_CRORE = 100.0

_MISSING = {"", "-", "–", "—", "nil", "na", "n/a"}


def parse_number(raw: str | None) -> float | None:
    """Parse Indian-grouped numbers, (parenthesised) negatives, ratios, EPS."""
    if raw is None:
        return None
    s = raw.strip().replace("\u00a0", " ").strip()
    if s.lower() in _MISSING:
        return None
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    s = s.replace(",", "").strip()
    if s.lower() in _MISSING or s == "":
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def normalize_label(raw: str) -> str:
    s = raw.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


@dataclass
class XBRLTable:
    index: int
    rows: list[list[str]]  # [particulars, current_period, ytd]


@dataclass
class XBRLParse:
    filename: str
    currency: str = "INR"
    amount_unit: str = "lakhs"  # as stated in the document
    stated_report_basis: str = ""  # "Consolidated"/"Standalone" as printed
    company_name: str = ""
    period_start: str = ""
    period_end: str = ""
    audit_status: str = ""
    tables: list[XBRLTable] = field(default_factory=list)

    @property
    def table(self) -> dict[int, XBRLTable]:
        return {t.index: t for t in self.tables}


def parse_xbrl(path: Path) -> XBRLParse:
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    raw = path.read_text(encoding="utf-8", errors="replace")
    result = XBRLParse(filename=path.name)

    m = re.search(r"Amount in \(([^)]+)\)", raw)
    if m:
        result.amount_unit = m.group(1).strip()
    m = re.search(r"presentation currency</th>\s*<td[^>]*>([^<]+)</td>", raw)
    if m:
        result.currency = m.group(1).strip().upper()

    for i, t in enumerate(soup.find_all("table")):
        rows: list[list[str]] = []
        for r in t.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
            if any(cells):
                rows.append(cells)
        result.tables.append(XBRLTable(index=i, rows=rows))

    # General info lives in table 0 as key/value pairs.
    if result.tables:
        info = {}
        for row in result.tables[0].rows:
            if len(row) >= 2:
                info[normalize_label(row[0])] = row[1].strip()
        result.company_name = info.get("name of company", "")
        result.period_start = info.get("date of start of financial year", "")
        result.period_end = info.get("date of end of financial year", "")
        result.currency = info.get("description of presentation currency", result.currency) or result.currency

    # Stated basis + period dates + audit status come from the P&L header rows.
    if len(result.tables) > 1:
        for row in result.tables[1].rows:
            if len(row) < 3:
                continue
            label = normalize_label(row[0])
            if label.startswith("b") and "end of reporting period" in normalize_label(" ".join(row[:2])):
                result.period_end = row[2].strip() or result.period_end
            joined = normalize_label(" ".join(row[:2]))
            if "nature of report" in joined:
                result.stated_report_basis = row[2].strip()
            if "whether results are audited" in joined:
                result.audit_status = row[2].strip()
            if "date of start of reporting period" in joined:
                result.period_start = row[2].strip() or result.period_start
            if "date of end of reporting period" in joined:
                result.period_end = row[2].strip() or result.period_end
    return result


# (label fragment, canonical metric, kind, excluded substrings) — checked in
# order, first hit wins. Exclusions stop near-miss rows (e.g. the regulatory
# deferral row mentions "deferred tax" without being the tax line).
PNL_MAP: list[tuple[str, str | None, str, tuple[str, ...]]] = [
    ("revenue from operations", "revenue_from_operations", "money", ()),
    ("other income", "other_income", "money", ()),
    ("total income", "total_income", "money", ()),
    ("cost of materials consumed", "cost_of_materials", "money", ()),
    ("purchases of stock-in-trade", "purchases_stock_in_trade", "money", ()),
    ("changes in inventories", "inventory_change", "money", ()),
    ("employee benefit expense", "employee_benefit", "money", ()),
    ("finance costs", "finance_costs", "money", ()),
    ("depreciation, depletion and amortisation expense", "depreciation", "money", ()),
    ("excise duty recovered", "excise_duty", "money", ()),
    ("excise duty", "excise_duty", "money", ()),
    # "Total other expenses" is the derivable aggregate (excise + other) — the
    # sub-row "Other Expenses" below is the canonical, cross-source-consistent
    # metric, so the aggregate is intentionally not stored.
    ("total other expenses", None, "money", ()),
    ("other expenses", "other_expenses", "money", ()),
    ("total expenses", "total_expenses", "money", ()),
    ("total profit before exceptional items and tax", "pbt_before_exceptional", "money", ()),
    ("exceptional items", "exceptional_items", "money", ()),
    ("total profit before tax", "pbt", "money", ()),
    ("current tax", "current_tax", "money", ("discontinued",)),
    ("deferred tax", "deferred_tax", "money", ("regulatory", "deferral", "discontinued")),
    ("total tax expenses", "total_tax", "money", ()),
    ("net profit loss for the period from", "pat_continuing", "money", ()),
    ("share of profit (loss) of associates", "share_associates_jv", "money", ()),
    ("total profit (loss) for period", "pat_total", "money", ("attributable",)),
    ("profit or loss, attributable to owners of parent", "pat_owners", "money", ()),
    ("attributable to non-controlling interests", "pat_nci", "money", ()),
    ("other comprehensive income net of taxes", "oci_net", "money", ()),
    ("total comprehensive income for the period", "total_comprehensive_income", "money", ("attributable",)),
    ("paid-up equity share capital", "paid_up_capital", "money", ()),
    ("face value of equity share capital", "face_value", "share", ()),
    ("basic earnings (loss) per share from continuing operations", "eps_basic", "share", ()),
    ("diluted earnings (loss) per share from continuing operations", "eps_diluted", "share", ()),
    ("debt equity ratio", "debt_equity_ratio", "ratio", ()),
    ("debt service coverage ratio", "dscr", "ratio", ()),
    ("interest service coverage ratio", "iscr", "ratio", ()),
]

SEGMENT_NAMES = {
    "oil to chemicals (o2c)": "o2c",
    "oil and gas": "oil_and_gas",
    "retail": "retail",
    "digital services": "digital_services",
    "others": "others",
}


def _money(value: float | None) -> float | None:
    return None if value is None else value / LAKH_TO_CRORE


def extract_pnl_facts(
    parsed: XBRLParse,
    *,
    company: str,
    ticker: str,
    financial_year: str,
    quarter: str,
    period: str,
    report_type: ReportType,
    source_document_id: str,
) -> list[FinancialFact]:
    facts: list[FinancialFact] = []
    if len(parsed.tables) <= 1:
        return facts
    for row in parsed.tables[1].rows:
        if len(row) < 3:
            continue
        # Layout: [serial, particulars, current-period, ytd]. Join every cell
        # except the two numeric tails so multi-column labels stay intact.
        particulars = normalize_label(" ".join(row[:-2]))
        value_raw = row[-2]  # current-period column (== YTD for Q1)
        for fragment, metric, kind, excludes in PNL_MAP:
            if fragment in particulars and not any(x in particulars for x in excludes):
                if metric is None:
                    break  # intentionally not stored (derivable aggregate)
                value = parse_number(value_raw)
                if value is None:
                    break
                if kind == "money":
                    value = value / LAKH_TO_CRORE
                    unit = "crore"
                elif kind == "share":
                    unit = "INR" if "face value" in fragment else "INR_per_share"
                else:
                    unit = "ratio"
                facts.append(
                    FinancialFact(
                        company=company,
                        ticker=ticker,
                        financial_year=financial_year,
                        quarter=quarter,
                        period=period,
                        period_type="quarter",
                        metric=metric,
                        value=value,
                        unit=unit,
                        currency=parsed.currency,
                        report_type=report_type,
                        source_document_id=source_document_id,
                        source_location=f"NSE Q1 FY27 {report_type.value} XBRL, P&L table (table 1): {row[1].strip()[:60] if len(row) > 1 else ''}",
                        extraction_method="xbrl_table",
                    )
                )
                break
    return facts


def extract_segment_facts(
    parsed: XBRLParse,
    *,
    company: str,
    ticker: str,
    financial_year: str,
    quarter: str,
    period: str,
    report_type: ReportType,
    source_document_id: str,
) -> list[FinancialFact]:
    """Parse the segment disclosure table (table 4) with a section state machine."""
    facts: list[FinancialFact] = []
    if len(parsed.tables) <= 4:
        return facts
    section: str | None = None
    for row in parsed.tables[4].rows:
        # Section headers have only 2 cells (no numeric columns), so header
        # detection must run BEFORE the 3-cell data guard below.
        text = normalize_label(" ".join(row))
        key = re.sub(r"^[\d.\s\-()]+", "", text).strip()
        if key.startswith("segment revenue (income)"):
            section = "segment_revenue"
            continue
        if key.startswith("segment result"):
            section = "segment_result_pbt"
            continue
        if "segment asset - segment liabilities" in text:
            continue  # super-header over sections 3-4; section unchanged
        if key == "segment asset":
            if section == "segment_result_pbt":
                section = "segment_assets"
            continue
        if key == "segment liabilities":
            if section == "segment_assets":
                section = "segment_liabilities"
            continue
        if len(row) < 3:
            continue
        text = normalize_label(" ".join(row[:-2]))
        value_raw = row[-2]
        # Total / net rows.
        if text.startswith("total segment revenue"):
            section = "segment_revenue"
            _append_segment(facts, "segment_revenue", "total", value_raw, parsed, company, ticker,
                            financial_year, quarter, period, report_type, source_document_id)
            continue
        if text.startswith("total profit before tax"):
            _append_segment(facts, "segment_result_pbt", "total", value_raw, parsed, company, ticker,
                            financial_year, quarter, period, report_type, source_document_id)
            continue
        if text.startswith("total segment asset"):
            _append_segment(facts, "segment_assets", "total", value_raw, parsed, company, ticker,
                            financial_year, quarter, period, report_type, source_document_id)
            continue
        if text.startswith("net segment asset"):
            _append_segment(facts, "segment_assets", "net_total", value_raw, parsed, company, ticker,
                            financial_year, quarter, period, report_type, source_document_id)
            continue
        if text.startswith("total segment liabilities"):
            _append_segment(facts, "segment_liabilities", "total", value_raw, parsed, company, ticker,
                            financial_year, quarter, period, report_type, source_document_id)
            continue
        if text.startswith("net segment liabilities"):
            _append_segment(facts, "segment_liabilities", "net_total", value_raw, parsed, company, ticker,
                            financial_year, quarter, period, report_type, source_document_id)
            continue
        if "less: inter segment" in text or "revenue from operations" in text and section == "segment_revenue" and "oil" not in text:
            continue  # eliminations / reconciliations are derivable, not stored
        if "finance cost" in text or "unallocable" in text or "profit before tax" == text:
            continue  # reconciliation bridge rows, not segment facts
        # Numbered segment rows: "<n> <Segment Name>".
        if section in {"segment_revenue", "segment_result_pbt", "segment_assets", "segment_liabilities"}:
            seg = _match_segment(text)
            if seg:
                _append_segment(facts, section, seg, value_raw, parsed, company, ticker,
                                financial_year, quarter, period, report_type, source_document_id)
    return facts


def _match_segment(text: str) -> str | None:
    for name, code in SEGMENT_NAMES.items():
        if name in text:
            return code
    return None


def _append_segment(
    facts: list[FinancialFact],
    metric: str,
    segment: str,
    value_raw: str,
    parsed: XBRLParse,
    company: str,
    ticker: str,
    financial_year: str,
    quarter: str,
    period: str,
    report_type: ReportType,
    source_document_id: str,
) -> None:
    value = parse_number(value_raw)
    if value is None:
        return
    facts.append(
        FinancialFact(
            company=company,
            ticker=ticker,
            financial_year=financial_year,
            quarter=quarter,
            period=period,
            period_type="quarter",
            metric=metric,
            value=value / LAKH_TO_CRORE,
            unit="crore",
            currency=parsed.currency,
            segment=segment,
            report_type=report_type,
            source_document_id=source_document_id,
            source_location=f"NSE Q1 FY27 {report_type.value} XBRL, segment table (table 4): {metric}/{segment}",
            extraction_method="xbrl_table",
        )
    )
