"""Display formatting for computed deltas (dashboard/table rendering).

Uses the same growth-vs-magnitude convention as the calculation engine
(percent change relative to abs(prior)) so the dashboard never disagrees
with /calc: an improvement from a loss reads positive in both places.
Undefined inputs (missing or zero prior) render as None, never zero.
"""


def yoy_delta_label(current: float | None, prior: float | None) -> str | None:
    """One-decimal signed delta label ("+12.3%"), or None when undefined.

    Same value the engine reports (it rounds to 2 decimals first), so the
    dashboard agrees with /calc to the displayed decimal
    (test_yoy_matches_engine pins this, including randomized cases).
    """
    if current is None or prior is None or prior == 0:
        return None
    value = round((current - prior) / abs(prior) * 100, 2)
    return f"{value:+.1f}%"
