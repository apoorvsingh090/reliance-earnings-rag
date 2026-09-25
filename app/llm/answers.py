"""Answer pipeline: route -> facts -> calculations -> context -> answer.

Two modes: "llm" when a configured provider is passed, otherwise an honest
extractive "template" mode that needs no API key. Both modes share the same
inputs (M4 context packs + M5 calculations), so template answers are never
invented — and the LLM is instructed to use only supplied numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.financial import engine as E
from app.llm.router import QueryClass, Route, route
from app.retrieval.citations import cite_fact, format_sources_block
from app.retrieval.context import ContextPack, build_context

DISPLAY_NAMES = {
    "revenue_from_operations": "Revenue from operations",
    "value_of_sales_services": "Value of sales & services",
    "total_income": "Total income",
    "other_income": "Other income",
    "pbt": "Profit before tax",
    "pat_total": "Profit after tax (total)",
    "pat_continuing": "Profit after tax (continuing)",
    "eps_basic": "Basic EPS",
    "eps_diluted": "Diluted EPS",
    "operating_margin": "Operating margin",
    "net_profit_margin": "Net profit margin",
    "net_worth": "Net worth",
    "total_assets": "Total assets",
    "segment_revenue": "Segment revenue",
    "segment_ebitda": "Segment EBITDA",
    "segment_ebit": "Segment EBIT",
}

PERIOD_RE = re.compile(r"Q([1-4]) FY(\d{2})")


def display_name(metric: str) -> str:
    if metric in DISPLAY_NAMES:
        return DISPLAY_NAMES[metric]
    return metric.replace("_yoy_growth", " YoY growth").replace("_qoq_growth", " QoQ growth") \
        .replace("_pp_change", " pp change").replace("_absolute_change", " absolute change") \
        .replace("_share_of_", " share of ").replace("_margin_on_", " margin on ") \
        .replace("_cagr", " CAGR").replace("_", " ")


def fmt_value(value: float, unit: str) -> str:
    if unit == "percent":
        return f"{value:,.2f}%"
    if unit == "pp":
        return f"{value:+,.2f}pp"
    if unit in ("INR_per_share", "INR"):
        return f"₹{value:,.2f}"
    if unit == "ratio":
        return f"{value:,.2f}"
    return f"₹{value:,.0f} crore"


def prior_year_period(period: str) -> str | None:
    m = PERIOD_RE.fullmatch(period.strip())
    if not m:
        return None
    return f"Q{m.group(1)} FY{int(m.group(2)) - 1:02d}"


def prior_quarter_period(period: str) -> str | None:
    m = PERIOD_RE.fullmatch(period.strip())
    if not m:
        return None
    q, fy = int(m.group(1)), int(m.group(2))
    if q == 1:
        return f"Q4 FY{fy - 1:02d}"
    return f"Q{q - 1} FY{m.group(2)}"


@dataclass
class Answer:
    text: str
    route: Route
    pack: ContextPack
    calculations: list = field(default_factory=list)  # CalcResult
    mode: str = "template"  # template | llm
    basis: str = "consolidated"


def known_periods(session) -> list[str]:
    from sqlalchemy import select

    from app.models.database import FinancialFactRow

    periods = session.execute(
        select(FinancialFactRow.period).distinct()
    ).scalars().all()

    def key(p: str) -> tuple:
        m = PERIOD_RE.fullmatch(p)
        if m:
            return (2000 + int(m.group(2)), int(m.group(1)))
        fy = re.fullmatch(r"FY(\d{2})", p)
        if fy:
            return (2000 + int(fy.group(1)), 9)
        return (0, 0)

    return sorted(set(periods), key=key)


def _latest(periods: list[str]) -> str:
    return periods[-1] if periods else "Q1 FY27"


def answer_question(question: str, session, retriever, llm=None, k: int = 6) -> Answer:
    r = route(question)
    periods = [p for p in r.periods if p != "LAST_N_QUARTERS"]
    if "LAST_N_QUARTERS" in r.periods or not periods:
        periods = known_periods(session) or ["Q1 FY27"]
    current = _latest([p for p in periods if PERIOD_RE.fullmatch(p)] or periods)
    hints = r.metric_hints
    calcs: list = []
    fact_reqs: list[dict] = []

    # Explicit cross-basis comparison: the user named both bases, so fetch both
    # side by side (each fact stays labelled — never blended).
    low = question.lower()
    both_bases = "standalone" in low and "consolidated" in low
    bases = ["standalone", "consolidated"] if both_bases else [r.report_type]

    def req(metric, period, segment=None):
        for b in bases:
            fact_reqs.append({"metric": metric, "period": period,
                              "report_type": b, "segment": segment})
    try:
        if r.query_class == QueryClass.FINANCIAL_FACT and hints:
            for p in periods:
                req(hints[0], p)
        elif r.query_class == QueryClass.CALCULATION and hints:
            m = hints[0]
            if "margin" in low and ("pp" in low or "change" in low or "decline" in low
                                    or "improv" in low):
                py = prior_year_period(current)
                if py:
                    req(m, current)
                    req(m, py)
                    c = E.calc_pp_change(session, m, current, py, r.report_type)
                    if c:
                        calcs.append(c)
            elif "qoq" in low or "q-o-q" in low or "quarter-on-quarter" in low:
                pq = prior_quarter_period(current)
                if pq:
                    req(m, current)
                    req(m, pq)
                    c = E.calc_qoq(session, m, current, pq, r.report_type)
                    if c:
                        calcs.append(c)
            else:  # default growth reading is YoY
                py = prior_year_period(current)
                if py:
                    req(m, current)
                    req(m, py)
                    c = E.calc_yoy(session, m, current, py, r.report_type)
                    if c:
                        calcs.append(c)
        elif r.query_class == QueryClass.SEGMENT_ANALYSIS:
            segs = r.segments or ["o2c"]
            py = prior_year_period(current)
            req("revenue_from_operations", current)  # denominator for share calcs
            for s in segs:
                req("segment_revenue", current, segment=s)
                if py:
                    req("segment_revenue", py, segment=s)
                    c = E.calc_yoy(session, "segment_revenue", current, py,
                                   r.report_type, segment=s)
                    if c:
                        calcs.append(c)
                c = E.calc_share(session, "segment_revenue", "revenue_from_operations",
                                 current, r.report_type, part_segment=s)
                if c:
                    calcs.append(c)
                if "margin" in low:
                    num = ("segment_ebitda" if "ebitda" in low else
                           "segment_ebit" if "ebit" in low else None)
                    if num:
                        req(num, current, segment=s)
                        m = E.calc_margin(session, num, "segment_revenue", current,
                                          r.report_type, segment=s)
                        if m:
                            calcs.append(m)
                for em in ("segment_ebitda", "segment_ebit"):
                    if em.split("_")[1] in low or "margin" in low or "profit" in low:
                        req(em, current, segment=s)
        elif r.query_class in (QueryClass.COMPARISON, QueryClass.CROSS_QUARTER) and hints:
            m = hints[0]
            for p in periods:
                req(m, p)
            if len(periods) >= 2:
                c = E.calc_absolute(session, m, periods[-1], periods[0], r.report_type)
                if c:
                    calcs.append(c)
        elif hints and periods:
            req(hints[0], periods[-1])
    except E.MissingFactError:
        pass  # template layer reports the absence honestly

    pack = build_context(question, retriever, session, k=k,
                         filters={"ticker": "RELIANCE"}, fact_requests=fact_reqs)
    basis = "standalone+consolidated" if both_bases else r.report_type
    answer = Answer(text="", route=r, pack=pack, calculations=calcs, basis=basis)
    if llm is not None:
        answer.text = _llm_answer(question, r, pack, calcs, llm, basis=answer.basis)
        answer.mode = "llm"
    else:
        answer.text = _template_answer(question, r, pack, calcs, basis=answer.basis)
    return answer


# ---------------------------------------------------------------------------
# Template (offline) rendering — extractive, never invented
# ---------------------------------------------------------------------------

def _fact_line(row) -> str:
    c = cite_fact(row)
    seg = f" ({row.segment})" if row.segment else ""
    return (f"{display_name(row.metric)}{seg} ({row.report_type}) in {row.period}: "
            f"**{fmt_value(row.value, row.unit)}** {c.short}")


def _template_answer(question: str, r: Route, pack: ContextPack, calcs: list,
                     basis: str | None = None) -> str:
    lines: list[str] = []
    basis = basis or r.report_type
    if "+" in basis:
        basis_note = ("You asked for an explicit comparison: figures below show "
                      "**both bases side by side**, each labelled — never blended.")
    else:
        basis_note = f"All figures below are on a **{basis}** basis."
    if pack.facts:
        lines.append("**Reported facts**")
        lines.extend(f"- {_fact_line(f)}" for f in pack.facts)
    if calcs:
        lines.append("**Calculated** (Python engine, inputs shown)")
        for c in calcs:
            inputs = ", ".join(f"{kk}={vv:,.2f}" for kk, vv in c.inputs.items()
                               if isinstance(vv, (int, float)))
            lines.append(f"- {display_name(c.metric)}: **{fmt_value(c.value, c.unit)}** "
                         f"[{inputs}] (basis: {c.basis})")
    if not pack.facts and not calcs:
        if r.metric_hints:
            lines.append(
                f"I don't have **{display_name(r.metric_hints[0])}** for the requested "
                f"period on a **{r.report_type}** basis in the indexed filings — "
                "so I won't guess. Try a listed metric (revenue, PAT, EPS, margins) "
                "or add another quarter.")
        else:
            lines.append("I couldn't map that question to a structured metric, "
                         "so here is the most relevant commentary from the filings:")
    if pack.chunks:
        lines.append("**Management commentary (extracts)**")
        for h in pack.chunks[:3]:
            snippet = " ".join(h.chunk.text.split())[:280]
            c = cite_chunk_of(h)
            said = ("According to the filing,"
                    if h.chunk.document_type == "xbrl"
                    else "According to the media release,")
            lines.append(f'- {said} "{snippet}…" {c}')
    lines.append(basis_note)
    if pack.citations:
        lines.append("**Sources used**")
        lines.append(format_sources_block(pack.citations))
    return "\n".join(lines)


def cite_chunk_of(h) -> str:
    from app.retrieval.citations import cite_chunk

    return cite_chunk(h.chunk).short


# ---------------------------------------------------------------------------
# LLM rendering — strict grounding prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You answer questions about Reliance Industries earnings using ONLY the provided context.
Rules, no exceptions:
- Use ONLY the numbers given in REPORTED FACTS and CALCULATIONS. Never invent, round beyond what is shown, or pull figures from memory.
- Every numeric claim ends with its citation exactly as shown, e.g. [NSE Q1 FY27 Consolidated XBRL] or [Q1 FY27 Media Release, p. 6].
- State the reporting basis once up front (consolidated unless the question says standalone). Never mix standalone and consolidated numbers in one figure; for explicit comparisons show both labelled.
- Distinguish three kinds of claims: reported facts ("Reported revenue was ..."), calculated values ("YoY growth calculates to ... from ..."), and management statements ("Management attributed ...", "According to the media release, ...").
- Never present management's explanation as independently verified fact. If a management statement conflicts with reported data, flag it.
- If the context lacks the answer, say so plainly instead of guessing.
Structure: 1) one-line conclusion 2) supporting figures with citations 3) management commentary where relevant 4) basis note."""


def build_llm_prompt(question: str, r: Route, pack: ContextPack, calcs: list) -> str:
    parts = [f"Question ({r.query_class.value}, basis: {r.report_type}): {question}", "",
             "REPORTED FACTS:"]
    for f in pack.facts:
        seg = f" segment={f.segment}" if f.segment else ""
        parts.append(f"- {f.metric}{seg} = {f.value} {f.unit} [{f.period}, {f.report_type}] "
                     f"src: {f.source_location}")
    parts.append("")
    parts.append("CALCULATIONS (authoritative, do not recompute):")
    for c in calcs:
        parts.append(f"- {c.metric} = {c.value} {c.unit} from {c.inputs} (basis {c.basis})")
    if not calcs:
        parts.append("(none)")
    parts += ["", "RETRIEVED COMMENTARY:"]
    for h in pack.chunks[:6]:
        parts.append(f"- [{h.chunk.source_location}] {h.chunk.text[:600]}")
    return "\n".join(parts)


def _llm_answer(question: str, r: Route, pack: ContextPack, calcs: list, llm,
                basis: str | None = None) -> str:
    prompt = build_llm_prompt(question, r, pack, calcs)
    try:
        return llm.complete(SYSTEM_PROMPT, prompt).text
    except Exception as exc:  # noqa: BLE001 — fall back, never fail the question
        fallback = _template_answer(question, r, pack, calcs, basis=basis)
        return fallback + f"\n(LLM unavailable: {exc})"
