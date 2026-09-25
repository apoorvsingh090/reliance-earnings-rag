"""Milestone 2 tests: XBRL/PDF parsing, fact mapping, DB isolation + idempotency.

Uses the real Q1 FY27 seed files with an in-memory SQLite database, so tests
are deterministic without needing PostgreSQL.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, select

from app.config import PROJECT_ROOT, get_settings
from app.extraction.pdf import _collect_values, _to_float, parse_pdf
from app.extraction.pipeline import run_extraction
from app.extraction.xbrl import normalize_label, parse_number, parse_xbrl
from app.financial.query import compare_bases, get_fact
from app.models.database import FinancialFactRow, count_facts, get_session, init_db

RAW = PROJECT_ROOT / "data" / "raw" / "reliance"
CONS_XBRL = RAW / "NSE_Q1FY27_consolidated_xbrl.html"
STAND_XBRL = RAW / "NSE_Q1FY27_standalone_xbrl.html"
MEDIA_PDF = RAW / "RIL_Q1FY27_media_release.pdf"

needs_seed = pytest.mark.skipif(
    not (CONS_XBRL.exists() and STAND_XBRL.exists() and MEDIA_PDF.exists()),
    reason="seed documents not downloaded (run scripts/ingest_q1fy27.py)",
)


def _mem_session():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return get_session(engine)


# --- number parsing ---------------------------------------------------------

def test_parse_number_indian_grouping_negatives_missing():
    assert parse_number("3,11,85,000.00") == 31185000.0
    assert parse_number("(1,32,600.00)") == -132600.0
    assert parse_number("15.48") == 15.48
    assert parse_number("-") is None
    assert parse_number("") is None
    assert parse_number(None) is None
    assert parse_number("0.00") == 0.0


def test_pdf_value_tokens():
    assert _to_float("(1,326)") == -1326.0
    assert _to_float("340,257") == 340257.0
    assert _to_float("-") is None
    lines = ["Revenue from Operations ", "311,850 ", "298,621 ", "248,660 ", "1,075,675 "]
    assert _collect_values(lines, 0) == [311850.0, 298621.0, 248660.0, 1075675.0]


# --- XBRL document self-description ------------------------------------------

@needs_seed
def test_xbrl_states_own_basis_and_units():
    cons = parse_xbrl(CONS_XBRL)
    stand = parse_xbrl(STAND_XBRL)
    assert cons.stated_report_basis == "Consolidated"
    assert stand.stated_report_basis == "Standalone"
    assert cons.amount_unit.lower() == "lakhs"
    assert cons.currency == "INR"
    assert len(cons.tables) == 9


# --- PDF page provenance ------------------------------------------------------

@needs_seed
def test_pdf_page_count_labels_and_no_silent_gaps():
    parsed = parse_pdf(MEDIA_PDF)
    assert len(parsed.pages) == 34
    assert parsed.flagged_pages == []  # rich digital text; nothing needs OCR
    assert all(p.pdf_page == i + 1 for i, p in enumerate(parsed.pages))
    assert any("of 20" in p.printed_label for p in parsed.pages)
    assert any("of 14" in p.printed_label for p in parsed.pages)


# --- end-to-end fact values ----------------------------------------------------

@needs_seed
def test_pipeline_spot_values_and_isolation():
    session = _mem_session()
    run_extraction(engine=session.bind)

    def v(metric, period, basis, segment=None):
        row = get_fact(session, metric, period, basis, segment)
        assert row is not None, f"missing {metric} {period} {basis} seg={segment}"
        return row.value

    # Standalone vs consolidated are never confused.
    assert v("revenue_from_operations", "Q1 FY27", "consolidated") == 311850.0
    assert v("revenue_from_operations", "Q1 FY27", "standalone") == 166013.0
    assert v("pat_total", "Q1 FY27", "consolidated") == 23196.0
    assert v("pat_total", "Q1 FY27", "standalone") == 13272.0
    assert v("eps_basic", "Q1 FY27", "consolidated") == 15.48
    assert v("eps_basic", "Q1 FY27", "standalone") == 9.81
    # Parenthesised XBRL negatives survive the lakh->crore conversion.
    assert v("inventory_change", "Q1 FY27", "consolidated") == -1326.0
    # Annexure-only metrics (margins, net worth, comparatives).
    assert v("operating_margin", "Q1 FY27", "consolidated") == 9.5
    assert v("net_profit_margin", "Q1 FY27", "consolidated") == 6.8
    assert v("net_worth", "Q1 FY27", "consolidated") == 880365.0
    assert v("total_assets", "Q1 FY27", "consolidated") == 2208747.0
    assert v("revenue_from_operations", "Q1 FY26", "consolidated") == 248660.0
    assert v("revenue_from_operations", "Q4 FY26", "consolidated") == 298621.0
    assert v("revenue_from_operations", "FY26", "consolidated") == 1075675.0
    # Segments (PDF annexure, consolidated).
    assert v("segment_revenue", "Q1 FY27", "consolidated", "o2c") == 201803.0
    assert v("segment_ebitda", "Q1 FY27", "consolidated", "digital_services") == 21255.0
    assert v("segment_ebit", "Q1 FY27", "consolidated", "retail") == 4529.0
    # other_expenses excludes excise (XBRL sub-row == PDF row).
    assert v("other_expenses", "Q1 FY27", "consolidated") == 42870.0

    cmp_ = compare_bases(session, "revenue_from_operations", "Q1 FY27")
    assert cmp_["consolidated"] == 311850.0 and cmp_["standalone"] == 166013.0
    session.close()


@needs_seed
def test_absent_metrics_are_none_not_fabricated():
    session = _mem_session()
    run_extraction(engine=session.bind)
    assert get_fact(session, "total_debt", "Q1 FY27", "consolidated") is None
    assert get_fact(session, "total_equity", "Q1 FY27", "consolidated") is None
    assert get_fact(session, "revenue_from_operations", "Q9 FY99", "consolidated") is None
    session.close()


@needs_seed
def test_xbrl_pdf_cross_source_consistency():
    session = _mem_session()
    run_extraction(engine=session.bind)
    rows = session.execute(
        select(FinancialFactRow).where(
            FinancialFactRow.period == "Q1 FY27",
            FinancialFactRow.report_type == "consolidated",
            FinancialFactRow.segment.is_(None),
        )
    ).scalars().all()
    by_metric: dict[str, list[float]] = {}
    for r in rows:
        by_metric.setdefault(r.metric, []).append(r.value)
    multi = {m: vals for m, vals in by_metric.items() if len(vals) > 1}
    assert len(multi) >= 10  # real overlap between XBRL and annexure
    for metric, vals in multi.items():
        assert max(vals) - min(vals) <= 0.51, f"{metric} disagrees: {vals}"
    session.close()


@needs_seed
def test_pipeline_idempotent():
    session = _mem_session()
    first = run_extraction(engine=session.bind)
    n = count_facts(session)
    assert n > 200
    second = run_extraction(engine=session.bind)
    assert count_facts(session) == n
    assert second["facts_created"] == 0
    session.close()


@needs_seed
def test_provenance_points_at_real_sources():
    session = _mem_session()
    run_extraction(engine=session.bind)
    settings = get_settings()
    import json

    registry = json.loads(settings.registry_path.read_text(encoding="utf-8"))
    valid_hashes = set(registry)
    rows = session.execute(select(FinancialFactRow)).scalars().all()
    assert rows
    for r in rows:
        assert r.source_document_id in valid_hashes
        assert r.source_location  # non-empty human-readable spot
        assert r.extraction_method in {"xbrl_table", "pdf_annexure_table"}
    session.close()


def test_normalize_label():
    assert normalize_label("  Revenue\n   from  operations ") == "revenue from operations"
