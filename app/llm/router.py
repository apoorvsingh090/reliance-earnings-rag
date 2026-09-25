"""Rule-based query router (spec section 12). Deterministic and offline.

Eight classes: FINANCIAL_FACT, CALCULATION, COMPARISON, QUALITATIVE, GUIDANCE,
CROSS_QUARTER, SEGMENT_ANALYSIS, GENERAL — plus extracted slots (periods,
reporting basis, segments, metric hints) that drive fact lookup in M4/M5.
An LLM classifier can replace route() later behind the same return type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryClass(str, Enum):
    FINANCIAL_FACT = "FINANCIAL_FACT"
    CALCULATION = "CALCULATION"
    COMPARISON = "COMPARISON"
    QUALITATIVE = "QUALITATIVE"
    GUIDANCE = "GUIDANCE"
    CROSS_QUARTER = "CROSS_QUARTER"
    SEGMENT_ANALYSIS = "SEGMENT_ANALYSIS"
    GENERAL = "GENERAL"


@dataclass
class Route:
    query_class: QueryClass
    periods: list[str] = field(default_factory=list)
    report_type: str = "consolidated"
    segments: list[str] = field(default_factory=list)
    metric_hints: list[str] = field(default_factory=list)
    needs_narrative: bool = False


PERIOD_RE = re.compile(r"\bQ([1-4])\s*FY(\d{2})\b", re.I)
FY_RE = re.compile(r"\bFY(\d{2})\b", re.I)

SEGMENT_TERMS: dict[str, str] = {
    "o2c": "o2c", "oil to chemicals": "o2c",
    "oil and gas": "oil_and_gas", "oil & gas": "oil_and_gas", "upstream": "oil_and_gas",
    "kgd6": "oil_and_gas",
    "retail": "retail", "rrvl": "retail", "stores": "retail",
    "jio": "digital_services", "digital": "digital_services", "telecom": "digital_services",
    "jpl": "digital_services", "5g": "digital_services",
}

# word -> canonical metric(s), first = primary.
METRIC_TERMS: dict[str, list[str]] = {
    "revenue": ["revenue_from_operations", "value_of_sales_services"],
    "sales": ["value_of_sales_services", "revenue_from_operations"],
    "income": ["total_income", "other_income"],
    "other income": ["other_income"],
    "ebitda": ["segment_ebitda"],
    "ebit": ["segment_ebit"],
    "profit before tax": ["pbt"],
    "pbt": ["pbt"],
    "pat": ["pat_total", "pat_continuing"],
    "net profit": ["pat_total"],
    "profit": ["pat_total", "pbt"],
    "eps": ["eps_basic", "eps_diluted"],
    "earnings per share": ["eps_basic"],
    "tax": ["total_tax", "current_tax", "deferred_tax"],
    "margin": ["operating_margin", "net_profit_margin"],
    "operating margin": ["operating_margin"],
    "net margin": ["net_profit_margin"],
    "net worth": ["net_worth"],
    "equity": ["other_equity", "net_worth"],
    "assets": ["total_assets", "segment_assets"],
    "debt": ["total_debt", "debt_equity_ratio"],
    "capex": [],
    "subscriber": [],
    "customer": [],
}

GUIDANCE_WORDS = ("guidance", "outlook", "capex", "management say", "management said",
                  "management state", "chairman", "commentary", "quote", "said about",
                  "view on", "expectation")
COMPARE_WORDS = ("compare", "comparison", " vs ", " versus ", "difference between",
                 "relative to", "better or worse")
CALC_WORDS = ("growth", "yoy", "y-o-y", "qoq", "q-o-q", "margin", "%", "percent",
              "change", "increase", "decrease", "decline", "calculate", "cagr",
              "ratio", "contribution", "share of")
WHY_WORDS = ("why", "what caused", "cause of", "reason for", "explain", "how has",
             "what drove", "attribute")
FACT_WORDS = ("what was", "what is", "what were", "how much", "give me", "tell me",
              "show me", "report", "announce", "post", "earn")


def _periods(text: str) -> list[str]:
    found: list[str] = []
    for q, fy in PERIOD_RE.findall(text):
        p = f"Q{q} FY{fy}"
        if p not in found:
            found.append(p)
    span = PERIOD_RE.sub(" ", text)
    for fy in FY_RE.findall(span):
        p = f"FY{fy}"
        if p not in found:
            found.append(p)
    if re.search(r"\blast \d+ quarters?\b", text, re.I) and not found:
        found.append("LAST_N_QUARTERS")
    return found


def _segments(text: str) -> list[str]:
    low = f" {text.lower()} "
    segs: list[str] = []
    for term, code in SEGMENT_TERMS.items():
        if term in low and code not in segs:
            segs.append(code)
    return segs


def _metric_hints(text: str) -> list[str]:
    low = text.lower()
    hints: list[str] = []
    for term in sorted(METRIC_TERMS, key=len, reverse=True):  # longest first
        if term in low:
            for m in METRIC_TERMS[term]:
                if m and m not in hints:
                    hints.append(m)
            break
    return hints


def _basis(text: str) -> str:
    low = text.lower()
    if "standalone" in low:
        return "standalone"
    if "consolidated" in low or "overall" in low or "company as a whole" in low:
        return "consolidated"
    return "consolidated"  # house default for general Reliance questions


def route(question: str) -> Route:
    low = f" {question.lower()} "
    periods = _periods(question)
    segments = _segments(question)
    metric_hints = _metric_hints(question)
    basis = _basis(question)

    if len(periods) >= 2 or "LAST_N_QUARTERS" in periods:
        return Route(QueryClass.CROSS_QUARTER, periods, basis, segments, metric_hints, True)
    if any(w in low for w in WHY_WORDS):
        return Route(QueryClass.QUALITATIVE, periods, basis, segments, metric_hints, True)
    if segments and any(w in low for w in ("compare", "growth", "revenue", "ebitda",
                                           "ebit", "margin", "perform", "vs", "contribution")):
        return Route(QueryClass.SEGMENT_ANALYSIS, periods, basis, segments, metric_hints, True)
    if any(w in low for w in GUIDANCE_WORDS):
        return Route(QueryClass.GUIDANCE, periods, basis, segments, metric_hints, True)
    if any(w in low for w in COMPARE_WORDS):
        return Route(QueryClass.COMPARISON, periods, basis, segments, metric_hints, True)
    if any(w in low for w in CALC_WORDS):
        return Route(QueryClass.CALCULATION, periods, basis, segments, metric_hints, False)
    if any(w in low for w in FACT_WORDS) or metric_hints:
        return Route(QueryClass.FINANCIAL_FACT, periods, basis, segments, metric_hints, False)
    return Route(QueryClass.GENERAL, periods, basis, segments, metric_hints, True)
