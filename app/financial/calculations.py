"""Pure financial math. No DB, no LLM, no guessing.

Every function takes explicit floats (or None) and returns a CalcResult that
carries its inputs and method — or None when the computation is undefined
(missing input, zero divisor). The engine layer resolves DB facts into these.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CalcResult(BaseModel):
    metric: str  # e.g. "revenue_yoy_growth"
    value: float  # rounded to 2 decimals
    unit: str  # "percent" | "pp" | "crore" | "ratio" | "x"
    inputs: dict = Field(default_factory=dict)
    method: str = ""
    basis: str = ""  # report_type the inputs share, or "mixed:<a>+<b>" when explicit
    sources: list[str] = Field(default_factory=list)  # source_locations of input facts


def _result(metric: str, value: float, unit: str, inputs: dict, method: str) -> CalcResult:
    return CalcResult(metric=metric, value=round(float(value), 2), unit=unit,
                      inputs=inputs, method=method)


def yoy_growth(current: float | None, prior_year: float | None) -> CalcResult | None:
    """(current - prior_year) / prior_year * 100."""
    if current is None or prior_year is None or prior_year == 0:
        return None
    return _result("yoy_growth", (current - prior_year) / abs(prior_year) * 100, "percent",
                   {"current": current, "prior_year": prior_year}, "yoy")


def qoq_growth(current: float | None, prior_qtr: float | None) -> CalcResult | None:
    """(current - prior_qtr) / prior_qtr * 100."""
    if current is None or prior_qtr is None or prior_qtr == 0:
        return None
    return _result("qoq_growth", (current - prior_qtr) / abs(prior_qtr) * 100, "percent",
                   {"current": current, "prior_quarter": prior_qtr}, "qoq")


def margin(numerator: float | None, revenue: float | None) -> CalcResult | None:
    """numerator / revenue * 100 (EBITDA/EBIT/PAT margin — caller names it)."""
    if numerator is None or revenue is None or revenue == 0:
        return None
    return _result("margin", numerator / revenue * 100, "percent",
                   {"numerator": numerator, "revenue": revenue}, "margin")


def cagr(start: float | None, end: float | None, periods: int) -> CalcResult | None:
    """(end / start) ** (1 / periods) - 1, in percent."""
    if start is None or end is None or periods <= 0 or start <= 0 or end <= 0:
        return None
    return _result("cagr", ((end / start) ** (1.0 / periods) - 1) * 100, "percent",
                   {"start": start, "end": end, "periods": periods}, "cagr")


def absolute_change(current: float | None, previous: float | None,
                    unit: str = "crore") -> CalcResult | None:
    if current is None or previous is None:
        return None
    return _result("absolute_change", current - previous, unit,
                   {"current": current, "previous": previous}, "absolute_change")


def pp_change(current_pct: float | None, previous_pct: float | None) -> CalcResult | None:
    """Percentage-point change between two percent values."""
    if current_pct is None or previous_pct is None:
        return None
    return _result("pp_change", current_pct - previous_pct, "pp",
                   {"current": current_pct, "previous": previous_pct}, "pp_change")


def share(part: float | None, total: float | None) -> CalcResult | None:
    """Segment contribution: part / total * 100."""
    if part is None or total is None or total == 0:
        return None
    return _result("share", part / total * 100, "percent",
                   {"part": part, "total": total}, "share")


def guidance_vs_actual(actual: float | None, guidance: float | None) -> CalcResult | None:
    """Variance of actual vs guidance, absolute and percent."""
    if actual is None or guidance is None or guidance == 0:
        return None
    variance = actual - guidance
    return _result("guidance_variance", variance / abs(guidance) * 100, "percent",
                   {"actual": actual, "guidance": guidance, "variance_abs": variance},
                   "guidance_vs_actual")
