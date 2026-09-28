"""Reliance Earnings Dashboard — analytics over the structured-facts DB.

Complements streamlit_app.py (the Q&A UI): this page needs no LLM key and no
embedding model — it reads the SQLite facts DB + eval report directly.

Run:  streamlit run dashboard.py
Data: built by scripts/ingest_q1fy27.py -> extract -> index (see README).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd
import streamlit as st

from app.config import get_settings
from app.financial.query import get_fact, list_facts
from app.models.database import (
    count_chunks,
    count_embeddings,
    count_facts,
    get_session,
)

st.set_page_config(page_title="Reliance Earnings Dashboard", layout="wide")
st.title("Reliance Earnings Dashboard")
st.caption("Structured facts from NSE filings + media release — standalone and consolidated never mixed.")

SEGMENTS = ["o2c", "retail", "digital_services", "oil_and_gas", "others"]
SEG_LABELS = {
    "o2c": "O2C (Refining + Petrochem)",
    "retail": "Retail",
    "digital_services": "Digital (Jio)",
    "oil_and_gas": "Oil & Gas",
    "others": "Others",
}
TREND_PERIODS = ["Q1 FY26", "Q4 FY26", "Q1 FY27"]
KPI_METRICS = [
    ("revenue_from_operations", "Revenue", "₹ cr"),
    ("total_income", "Total income", "₹ cr"),
    ("pat_total", "PAT (total)", "₹ cr"),
    ("eps_basic", "EPS (basic)", "₹"),
    ("operating_margin", "Operating margin", "%"),
    ("net_profit_margin", "Net margin", "%"),
]


def inr_fmt(x: float) -> str:
    neg = x < 0
    x = abs(int(round(x)))
    s = str(x)
    if len(s) <= 3:
        out = s
    else:
        out, s = s[-3:], s[:-3]
        while len(s) > 2:
            out, s = s[-2:] + "," + out, s[:-2]
        out = (s + "," + out) if s else out
    return ("-" if neg else "") + out


def fmt_value(metric: str, v: float) -> str:
    if metric in ("eps_basic", "eps_diluted"):
        return f"₹{v:,.2f}"
    if "margin" in metric or "ratio" in metric:
        return f"{v:.1f}%"
    return f"₹{inr_fmt(v)} cr"


@st.cache_data(ttl=600)
def _facts(period: str, basis: str) -> dict:
    session = get_session()
    try:
        rows = list_facts(session, period=period, report_type=basis)
    finally:
        session.close()
    out: dict = {}
    for r in rows:  # de-dupe: same fact extracted from 2 docs
        out.setdefault((r.metric, r.segment), r)
    return out


@st.cache_data(ttl=600)
def _corpus():
    session = get_session()
    try:
        counts = {
            "facts": count_facts(session),
            "chunks": count_chunks(session),
            "embeddings": count_embeddings(session),
        }
    finally:
        session.close()
    s = get_settings()
    manifest, eval_report = [], {}
    if s.manifest_path.exists():
        manifest = json.loads(s.manifest_path.read_text())
    eval_path = s.processed_dir / "eval_report.json"
    if eval_path.exists():
        eval_report = json.loads(eval_path.read_text())
    return counts, manifest, eval_report


def _get(facts: dict, metric: str, segment=None):
    r = facts.get((metric, segment))
    return r.value if r else None


# ------------------------------------------------------------------ guard ---
try:
    probe = _facts("Q1 FY27", "consolidated")
    if not probe:
        raise RuntimeError("empty")
except Exception:
    st.error("Facts DB not found or empty.")
    st.markdown(
        "Build it first (repo root):\n"
        "```\npython scripts/ingest_q1fy27.py\n"
        "python scripts/extract_q1fy27.py\n"
        "python scripts/index_q1fy27.py\n```\n"
        "For Streamlit Cloud, force-add the built artifacts so they deploy with the app:\n"
        "`git add -f data/processed/reliance_earnings.db data/processed/bm25_index.pkl "
        "data/processed/eval_report.json data/processed/manifest.json data/processed/registry.json`"
    )
    st.stop()

# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("Scope")
    basis = st.selectbox("Reporting basis", ["consolidated", "standalone"])
    period = st.selectbox("Period", ["Q1 FY27", "Q4 FY26", "Q1 FY26", "FY26"])
    st.divider()
    counts, manifest, eval_report = _corpus()
    st.metric("Documents", len(manifest))
    st.metric("Structured facts", counts["facts"])
    st.metric("Chunks / embeddings", f"{counts['chunks']} / {counts['embeddings']}")
    if eval_report:
        st.metric("Eval", f"{eval_report.get('passed')}/{eval_report.get('total')} pass")

facts = _facts(period, basis)
prev_period = {"Q1 FY27": "Q1 FY26", "Q4 FY26": "Q4 FY25", "Q1 FY26": None, "FY26": None}[period]
prev = _facts(prev_period, basis) if prev_period else {}

# -------------------------------------------------------------------- KPIs ---
st.subheader(f"Key figures — {period} ({basis})")
cols = st.columns(len(KPI_METRICS))
pat_owners = _get(facts, "pat_owners")
for col, (metric, label, _unit) in zip(cols, KPI_METRICS):
    v = _get(facts, metric)
    if metric == "pat_total" and pat_owners is not None:
        v, label = pat_owners, "PAT (owners)"
    if v is None:
        col.metric(label, "—")
        continue
    delta = None
    if prev:
        pv = _get(prev, metric if metric != "pat_total" or pat_owners is None else "pat_owners")
        if pv:
            delta = f"{(v / pv - 1) * 100:+.1f}% YoY"
    col.metric(label, fmt_value(metric, v), delta)

# ---------------------------------------------------------------- segments ---
st.divider()
st.subheader(f"Segment mix — {period} ({basis})")
rev = {s: _get(facts, "segment_revenue", s) for s in SEGMENTS}
ebitda = {s: _get(facts, "segment_ebitda", s) for s in SEGMENTS}
if any(rev.values()) or any(ebitda.values()):
    c1, c2 = st.columns(2)
    with c1:
        rev_df = pd.DataFrame(
            {"Revenue (₹ cr)": [rev[s] or 0 for s in SEGMENTS]},
            index=[SEG_LABELS[s] for s in SEGMENTS],
        )
        st.markdown("**Revenue by segment**")
        st.bar_chart(rev_df)
    with c2:
        ebitda_df = pd.DataFrame(
            {"EBITDA (₹ cr)": [ebitda[s] or 0 for s in SEGMENTS]},
            index=[SEG_LABELS[s] for s in SEGMENTS],
        )
        st.markdown("**EBITDA by segment**")
        st.bar_chart(ebitda_df)
    mrows = []
    for s in SEGMENTS:
        r, e = rev[s], ebitda[s]
        mrows.append({
            "Segment": SEG_LABELS[s],
            "Revenue (₹ cr)": inr_fmt(r) if r else "—",
            "EBITDA (₹ cr)": inr_fmt(e) if e else "—",
            "EBITDA margin": f"{e / r * 100:.1f}%" if r and e else "—",
        })
    st.table(pd.DataFrame(mrows).set_index("Segment"))
else:
    st.info(f"No segment facts for {period} on {basis} basis — try consolidated Q1 FY27.")

# ------------------------------------------------------------------- trend ---
st.divider()
st.subheader("Quarter trend (consolidated)")
trows = []
for p in TREND_PERIODS:
    f = _facts(p, "consolidated")
    trows.append({
        "Period": p,
        "Revenue (₹ cr)": _get(f, "revenue_from_operations"),
        "PAT total (₹ cr)": _get(f, "pat_total"),
        "EPS basic (₹)": _get(f, "eps_basic"),
    })
tdf = pd.DataFrame(trows).set_index("Period")
st.line_chart(tdf[["Revenue (₹ cr)", "PAT total (₹ cr)"]])
st.line_chart(tdf[["EPS basic (₹)"]])
st.table(tdf.map(lambda v: inr_fmt(v) if pd.notna(v) and v and abs(v) > 100 else (f"{v:.2f}" if pd.notna(v) else "—")))

# ------------------------------------------------------------ basis compare ---
st.divider()
st.subheader(f"Consolidated vs standalone — {period}")
brows = []
for metric, label, _unit in KPI_METRICS:
    session_vals = {}
    for b in ("consolidated", "standalone"):
        bf = _facts(period, b)
        vv = _get(bf, metric)
        if metric == "pat_total":
            o = _get(bf, "pat_owners")
            vv = o if o is not None else vv
        session_vals[b] = vv
    c, s_ = session_vals["consolidated"], session_vals["standalone"]
    brows.append({
        "Metric": label,
        "Consolidated": fmt_value(metric, c) if c is not None else "—",
        "Standalone": fmt_value(metric, s_) if s_ is not None else "—",
    })
st.table(pd.DataFrame(brows).set_index("Metric"))
st.caption("Bases are shown side-by-side, never summed or blended — per the repo's house rule.")

# ------------------------------------------------------------ fact explorer ---
st.divider()
st.subheader("Fact explorer")
session = get_session()
try:
    from sqlalchemy import select as _select
    from app.models.database import FinancialFactRow
    metrics = sorted(r[0] for r in session.execute(
        _select(FinancialFactRow.metric).distinct()).all())
finally:
    session.close()
mchoice = st.selectbox("Metric", metrics, index=metrics.index("revenue_from_operations"))
erows = []
for p in ["Q1 FY26", "Q4 FY26", "Q1 FY27", "FY26"]:
    row = {"Period": p}
    for b in ("consolidated", "standalone"):
        bf = _facts(p, b)
        vals = [(seg, _get(bf, mchoice, seg)) for seg in
                (None, "o2c", "retail", "digital_services", "oil_and_gas", "others", "total")
                if _get(bf, mchoice, seg) is not None]
        row[b.capitalize()] = "; ".join(
            f"{seg or 'company'}: {inr_fmt(v)}" for seg, v in vals) or "—"
    erows.append(row)
st.table(pd.DataFrame(erows).set_index("Period"))

# ------------------------------------------------------- eval & corpus health ---
st.divider()
st.subheader("System health")
e1, e2 = st.columns(2)
with e1:
    st.markdown("**Evaluation (26-question set)**")
    if eval_report:
        st.metric("Pass rate", f"{eval_report['passed']}/{eval_report['total']}",
                  f"{eval_report['passed'] / eval_report['total'] * 100:.0f}%")
        by_cat = eval_report.get("by_category", {})
        cat_df = pd.DataFrame(
            {"pass %": [by_cat[c]["passed"] / by_cat[c]["total"] * 100 for c in by_cat]},
            index=list(by_cat),
        )
        st.bar_chart(cat_df)
        m = eval_report.get("metrics", {})
        st.table(pd.DataFrame({"score": m}).T)
    else:
        st.info("No eval_report.json — run scripts/run_eval.py.")
with e2:
    st.markdown("**Indexed documents**")
    if manifest:
        st.table(pd.DataFrame([{
            "File": d["filename"], "Type": d["document_type"],
            "Period": d["period"], "Basis": d["reporting_basis"],
        } for d in manifest]).set_index("File"))
    else:
        st.warning("No manifest — ingest documents first.")

st.divider()
st.caption("Q&A with citations lives in `streamlit_app.py`. This dashboard reads the same "
           "facts DB and needs no API key. Standalone vs consolidated is never mixed.")
