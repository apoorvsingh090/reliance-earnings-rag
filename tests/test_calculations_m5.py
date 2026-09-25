"""Milestone 5 tests: pure math + DB-backed engine with basis guard."""

from __future__ import annotations

import pytest

from app.financial import calculations as C
from app.financial.engine import (
    MixedBasisError,
    MissingFactError,
    calc_absolute,
    calc_cagr,
    calc_margin,
    calc_pp_change,
    calc_qoq,
    calc_share,
    calc_yoy,
)
from app.models.database import get_session


# --- pure functions -------------------------------------------------------------

def test_yoy_qoq_margin_math():
    assert C.yoy_growth(311850.0, 248660.0).value == pytest.approx(25.41, abs=0.01)
    assert C.qoq_growth(311850.0, 298621.0).value == pytest.approx(4.43, abs=0.01)
    assert C.margin(14170.0, 201803.0).value == pytest.approx(7.02, abs=0.01)
    r = C.yoy_growth(100.0, 80.0)
    assert r.inputs == {"current": 100.0, "prior_year": 80.0} and r.unit == "percent"


def test_pp_cagr_share_absolute_guidance():
    assert C.pp_change(9.5, 10.6).value == pytest.approx(-1.1)
    assert C.pp_change(9.5, 10.6).unit == "pp"
    assert C.cagr(100.0, 133.1, 3).value == pytest.approx(10.0, abs=0.01)
    assert C.share(201803.0, 311850.0).value == pytest.approx(64.71, abs=0.01)
    assert C.absolute_change(311850.0, 298621.0).value == pytest.approx(13229.0)
    g = C.guidance_vs_actual(105.0, 100.0)
    assert g.value == pytest.approx(5.0) and g.inputs["variance_abs"] == pytest.approx(5.0)


def test_undefined_is_none_not_zero():
    assert C.yoy_growth(100.0, 0.0) is None
    assert C.yoy_growth(None, 80.0) is None
    assert C.margin(10.0, 0.0) is None
    assert C.share(10.0, 0.0) is None
    assert C.cagr(-5.0, 10.0, 2) is None
    assert C.cagr(100.0, 110.0, 0) is None
    assert C.guidance_vs_actual(10.0, 0.0) is None


# --- DB-backed engine -------------------------------------------------------------

def test_engine_yoy_qoq_revenue_both_bases():
    session = get_session()
    try:
        r = calc_yoy(session, "revenue_from_operations", "Q1 FY27", "Q1 FY26", "consolidated")
        assert r.value == pytest.approx(25.41, abs=0.01)
        assert r.basis == "consolidated" and len(r.sources) == 2
        assert all(r.sources)  # every input cites a real source location

        r = calc_yoy(session, "revenue_from_operations", "Q1 FY27", "Q1 FY26", "standalone")
        assert r.value == pytest.approx(36.79, abs=0.02)
        assert r.basis == "standalone"

        q = calc_qoq(session, "revenue_from_operations", "Q1 FY27", "Q4 FY26", "consolidated")
        assert q.value == pytest.approx(4.43, abs=0.01)
    finally:
        session.close()


def test_engine_margins_and_share():
    session = get_session()
    try:
        m = calc_margin(session, "segment_ebit", "segment_revenue", "Q1 FY27",
                        "consolidated", segment="o2c")
        assert m.value == pytest.approx(7.02, abs=0.01)
        assert m.basis == "consolidated"

        retail = calc_margin(session, "segment_ebitda", "segment_revenue", "Q1 FY27",
                             "consolidated", segment="retail")
        assert retail.value == pytest.approx(6.98, abs=0.01)

        s = calc_share(session, "segment_revenue", "revenue_from_operations", "Q1 FY27",
                       "consolidated", part_segment="o2c")
        assert s.value == pytest.approx(64.71, abs=0.01)

        pp = calc_pp_change(session, "operating_margin", "Q1 FY27", "Q1 FY26", "consolidated")
        assert pp.value == pytest.approx(-1.1, abs=0.01) and pp.unit == "pp"
    finally:
        session.close()


def test_engine_missing_fact_raises():
    session = get_session()
    try:
        with pytest.raises(MissingFactError):
            calc_yoy(session, "total_debt", "Q1 FY27", "Q1 FY26", "consolidated")
        with pytest.raises(MissingFactError):
            calc_yoy(session, "revenue_from_operations", "Q1 FY27", "Q9 FY99", "consolidated")
    finally:
        session.close()


def test_engine_basis_guard_and_explicit_comparison():
    # Engine resolves one basis per call, so mixing is constructed at the pure
    # layer here; the guard itself is exercised via _check_basis semantics:
    # same-basis pairs pass, and results are always basis-labelled.
    session = get_session()
    try:
        r = calc_absolute(session, "pat_total", "Q1 FY27", "Q1 FY26", "consolidated")
        assert r.basis == "consolidated"
        assert r.value == pytest.approx(23196.0 - 30783.0, abs=0.51)
    finally:
        session.close()


def test_check_basis_mixed():
    from app.financial.engine import _check_basis

    class R:
        def __init__(self, b):
            self.report_type = b

    assert _check_basis([R("consolidated"), R("consolidated")], False) == "consolidated"
    with pytest.raises(MixedBasisError):
        _check_basis([R("consolidated"), R("standalone")], False)
    assert _check_basis([R("consolidated"), R("standalone")], True) == \
        "mixed:consolidated+standalone"


def test_engine_cagr_needs_real_span():
    session = get_session()
    try:
        # FY26 vs Q1 FY26 revenue as a 4-quarter span is indicative, not canonical;
        # the engine computes whatever explicit span it is given.
        r = calc_cagr(session, "revenue_from_operations", "Q1 FY26", "Q1 FY27", 4, "consolidated")
        assert r.value == pytest.approx(5.82, abs=0.1)
    finally:
        session.close()
