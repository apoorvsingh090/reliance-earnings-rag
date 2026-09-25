"""Canonical financial-fact contract.

Every structured number in the system is a FinancialFact: a (metric, period,
reporting basis, [segment]) tuple with a float value and full provenance.
Anything without a reliable value is simply absent (nullable by omission) —
we never fabricate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.document import ReportType

PeriodType = Literal["quarter", "year"]

# Canonical metric names. Money metrics are stored in INR crore unless the
# unit field says otherwise. Ratios/percent/EPS carry their own units.
MONEY_METRICS = frozenset(
    {
        "revenue_from_operations",
        "value_of_sales_services",
        "gst_recovered",
        "other_income",
        "total_income",
        "cost_of_materials",
        "purchases_stock_in_trade",
        "inventory_change",
        "employee_benefit",
        "finance_costs",
        "depreciation",
        "excise_duty",
        "other_expenses",
        "total_expenses",
        "pbt_before_exceptional",
        "exceptional_items",
        "pbt",
        "current_tax",
        "deferred_tax",
        "total_tax",
        "pat_continuing",
        "discontinued_net",
        "share_associates_jv",
        "pat_total",
        "pat_owners",
        "pat_nci",
        "oci_net",
        "total_comprehensive_income",
        "paid_up_capital",
        "other_equity",
        "net_worth",
        "total_assets",
        "segment_revenue",
        "segment_ebitda",
        "segment_ebit",
        "segment_result_pbt",
        "segment_assets",
        "segment_liabilities",
        "capital_redemption_reserve",
    }
)

RATIO_METRICS = frozenset(
    {
        "debt_equity_ratio",
        "dscr",
        "iscr",
        "current_ratio",
        "long_term_debt_to_working_capital",
        "current_liability_ratio",
        "total_debts_to_assets",
        "debtors_turnover",
        "inventory_turnover",
    }
)

PERCENT_METRICS = frozenset({"operating_margin", "net_profit_margin"})

SHARE_METRICS = frozenset({"eps_basic", "eps_diluted", "face_value"})

ALL_METRICS = MONEY_METRICS | RATIO_METRICS | PERCENT_METRICS | SHARE_METRICS

# Metrics intentionally NOT populated in M2 (no explicit source in Q1 FY27
# filings): total_debt (only ratios/NCD fragment disclosed), total_equity
# (net worth stored instead), EBITDA headline (segment EBITDA stored per
# segment; company-level EBITDA arrives with the calc engine in M5),
# operating_profit.
INTENTIONALLY_ABSENT = frozenset({"total_debt", "total_equity", "ebitda"})


class FinancialFact(BaseModel):
    company: str = "Reliance Industries Limited"
    ticker: str = "RELIANCE"
    financial_year: str  # e.g. "FY27"
    quarter: str | None = None  # e.g. "Q1" (None for full-year rows)
    period: str  # e.g. "Q1 FY27", "Q4 FY26", "FY26"
    period_type: PeriodType = "quarter"
    metric: str
    value: float
    unit: str = "crore"
    currency: str = "INR"
    segment: str | None = None  # e.g. "o2c"; None = company level
    report_type: ReportType
    source_document_id: str = Field(description="SHA256 of the source document")
    source_location: str = Field(description="Human-readable spot, e.g. XBRL table + row")
    extraction_method: str = Field(description="e.g. xbrl_table, pdf_annexure_table")

    def fact_key(self) -> tuple:
        return (
            self.source_document_id,
            self.metric,
            self.period,
            self.report_type.value,
            self.segment or "",
        )
