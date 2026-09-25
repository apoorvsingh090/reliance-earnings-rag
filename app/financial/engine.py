"""DB-backed calculations with the standalone/consolidated guard.

All inputs to one calculation must share a reporting basis. Mixing bases
raises MixedBasisError unless the caller passes allow_mixed=True — the path
reserved for EXPLICIT user comparisons, which labels the result
basis="mixed:..." so answers can never silently blend standalone with
consolidated.
"""

from __future__ import annotations

from app.financial import calculations as C
from app.financial.calculations import CalcResult
from app.financial.query import get_fact


class MixedBasisError(ValueError):
    pass


class MissingFactError(ValueError):
    pass


def _resolve(session, metric: str, period: str, report_type: str, segment=None):
    row = get_fact(session, metric, period, report_type, segment)
    if row is None:
        raise MissingFactError(f"no fact: {metric} {period} {report_type} seg={segment}")
    return row


def _check_basis(rows: list, allow_mixed: bool) -> str:
    bases = {r.report_type for r in rows}
    if len(bases) > 1 and not allow_mixed:
        raise MixedBasisError(
            f"refusing to mix reporting bases in one calculation: {sorted(bases)}. "
            "Pass allow_mixed=True only for an explicit user-requested comparison."
        )
    if len(bases) > 1:
        return "mixed:" + "+".join(sorted(bases))
    return next(iter(bases))


def _attach(result: CalcResult | None, rows: list, basis: str, metric: str) -> CalcResult | None:
    if result is None:
        return None
    result.metric = metric
    result.basis = basis
    result.sources = [r.source_location for r in rows]
    return result


def calc_yoy(session, metric: str, current_period: str, prior_period: str,
             report_type: str = "consolidated", segment=None,
             allow_mixed: bool = False) -> CalcResult | None:
    cur = _resolve(session, metric, current_period, report_type, segment)
    pri = _resolve(session, metric, prior_period, report_type, segment)
    basis = _check_basis([cur, pri], allow_mixed)
    seg = f"_{segment}" if segment else ""
    return _attach(C.yoy_growth(cur.value, pri.value), [cur, pri], basis,
                   f"{metric}{seg}_yoy_growth")


def calc_qoq(session, metric: str, current_period: str, prior_period: str,
             report_type: str = "consolidated", segment=None,
             allow_mixed: bool = False) -> CalcResult | None:
    cur = _resolve(session, metric, current_period, report_type, segment)
    pri = _resolve(session, metric, prior_period, report_type, segment)
    basis = _check_basis([cur, pri], allow_mixed)
    seg = f"_{segment}" if segment else ""
    return _attach(C.qoq_growth(cur.value, pri.value), [cur, pri], basis,
                   f"{metric}{seg}_qoq_growth")


def calc_margin(session, num_metric: str, den_metric: str, period: str,
                report_type: str = "consolidated", segment=None,
                allow_mixed: bool = False) -> CalcResult | None:
    num = _resolve(session, num_metric, period, report_type, segment)
    den = _resolve(session, den_metric, period, report_type, segment)
    basis = _check_basis([num, den], allow_mixed)
    seg = f"_{segment}" if segment else ""
    return _attach(C.margin(num.value, den.value), [num, den], basis,
                   f"{num_metric}{seg}_margin_on_{den_metric}")


def calc_share(session, part_metric: str, total_metric: str, period: str,
               report_type: str = "consolidated", part_segment=None, total_segment=None,
               allow_mixed: bool = False) -> CalcResult | None:
    part = _resolve(session, part_metric, period, report_type, part_segment)
    total = _resolve(session, total_metric, period, report_type, total_segment)
    basis = _check_basis([part, total], allow_mixed)
    seg = f"_{part_segment}" if part_segment else ""
    return _attach(C.share(part.value, total.value), [part, total], basis,
                   f"{part_metric}{seg}_share_of_{total_metric}")


def calc_pp_change(session, metric: str, current_period: str, prior_period: str,
                   report_type: str = "consolidated", segment=None,
                   allow_mixed: bool = False) -> CalcResult | None:
    cur = _resolve(session, metric, current_period, report_type, segment)
    pri = _resolve(session, metric, prior_period, report_type, segment)
    basis = _check_basis([cur, pri], allow_mixed)
    seg = f"_{segment}" if segment else ""
    return _attach(C.pp_change(cur.value, pri.value), [cur, pri], basis,
                   f"{metric}{seg}_pp_change")


def calc_absolute(session, metric: str, current_period: str, prior_period: str,
                  report_type: str = "consolidated", segment=None,
                  allow_mixed: bool = False) -> CalcResult | None:
    cur = _resolve(session, metric, current_period, report_type, segment)
    pri = _resolve(session, metric, prior_period, report_type, segment)
    basis = _check_basis([cur, pri], allow_mixed)
    seg = f"_{segment}" if segment else ""
    # Absolute change inherits the fact's unit family (a margin delta is pp, not crore).
    unit = {"percent": "pp", "pp": "pp"}.get(cur.unit, cur.unit)
    return _attach(C.absolute_change(cur.value, pri.value, unit), [cur, pri], basis,
                   f"{metric}{seg}_absolute_change")


def calc_cagr(session, metric: str, start_period: str, end_period: str, steps: int,
              report_type: str = "consolidated", segment=None,
              allow_mixed: bool = False) -> CalcResult | None:
    start = _resolve(session, metric, start_period, report_type, segment)
    end = _resolve(session, metric, end_period, report_type, segment)
    basis = _check_basis([start, end], allow_mixed)
    seg = f"_{segment}" if segment else ""
    return _attach(C.cagr(start.value, end.value, steps), [start, end], basis,
                   f"{metric}{seg}_cagr")
