"""Read access to structured facts — standalone/consolidated isolation enforced.

Every lookup carries an explicit reporting basis (default: consolidated, the
house view for general Reliance questions). There is no code path that sums
or compares across bases; cross-basis comparison must be requested with two
explicit calls so the caller — and the eventual answer — states the basis.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from app.models.database import FinancialFactRow

log = logging.getLogger(__name__)


def get_fact(
    session,
    metric: str,
    period: str,
    report_type: str = "consolidated",
    segment: str | None = None,
) -> FinancialFactRow | None:
    """Return one fact. report_type is mandatory-in-effect (no silent mixing)."""
    stmt = select(FinancialFactRow).where(
        FinancialFactRow.metric == metric,
        FinancialFactRow.period == period,
        FinancialFactRow.report_type == report_type,
        FinancialFactRow.segment.is_(None) if segment is None
        else FinancialFactRow.segment == segment,
    )
    row = session.execute(stmt).scalars().first()
    if row is None:
        log.debug("fact missing: %s %s %s seg=%s", metric, period, report_type, segment)
    return row


def list_facts(session, *, period: str, report_type: str = "consolidated") -> list[FinancialFactRow]:
    stmt = (
        select(FinancialFactRow)
        .where(FinancialFactRow.period == period, FinancialFactRow.report_type == report_type)
        .order_by(FinancialFactRow.metric, FinancialFactRow.segment)
    )
    return list(session.execute(stmt).scalars().all())


def compare_bases(session, metric: str, period: str) -> dict:
    """Explicit standalone-vs-consolidated comparison (both bases stated)."""
    return {
        "metric": metric,
        "period": period,
        "consolidated": _value(get_fact(session, metric, period, "consolidated")),
        "standalone": _value(get_fact(session, metric, period, "standalone")),
    }


def _value(row: FinancialFactRow | None) -> float | None:
    return row.value if row else None
