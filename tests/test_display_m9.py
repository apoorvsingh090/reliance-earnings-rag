"""Milestone 9 tests: dashboard delta labels share the engine's convention."""

from __future__ import annotations

from app.financial import calculations as C
from app.financial.display import yoy_delta_label


def test_delta_matches_engine_convention():
    # (311850-248660)/248660 = +25.41% -> one-decimal label.
    assert yoy_delta_label(311850.0, 248660.0) == "+25.4%"


def test_delta_negative_prior_improvement_reads_positive():
    # Narrowing loss: engine convention measures against magnitude.
    assert yoy_delta_label(-50.0, -100.0) == "+50.0%"
    assert yoy_delta_label(-150.0, -100.0) == "-50.0%"


def test_delta_undefined_inputs_render_none():
    assert yoy_delta_label(100.0, 0) is None
    assert yoy_delta_label(None, 100.0) is None
    assert yoy_delta_label(100.0, None) is None


def test_yoy_matches_engine_delta():
    # Dashboard label must equal the engine value to 1 decimal, always.
    import random
    rng = random.Random(20261116)
    pairs = [(311850.0, 248660.0), (100.0, 90.0), (80.0, 100.0),
             (-50.0, -100.0), (-150.0, -100.0), (0.0, 5.0)]
    pairs += [(rng.uniform(-5000, 5000), rng.uniform(-5000, 5000)) for _ in range(300)]
    for current, prior in pairs:
        engine = C.yoy_growth(current, prior)
        if prior == 0:
            assert engine is None
            assert yoy_delta_label(current, prior) is None
        else:
            assert engine is not None
            assert yoy_delta_label(current, prior) == f"{engine.value:+.1f}%"
    assert yoy_delta_label(100.0, 0) is None
