"""The three Q1 FY27 source documents.

Reporting basis is explicit per document — the standalone and consolidated
XBRL filings have near-identical NSE URLs, so guessing from the filename
or URL alone would be fragile. These specs are the single source of truth.
"""

from __future__ import annotations

from app.models.document import DocumentSpec

Q1FY27_DOCUMENTS: list[DocumentSpec] = [
    DocumentSpec(
        financial_year="FY27",
        quarter="Q1",
        period="Q1 FY27",
        document_type="xbrl",
        report_type="standalone",
        source_url=(
            "https://nsearchives.nseindia.com/corporate/ixbrl/"
            "INTEGRATED_FILING_INDAS_175606_17072026194906_iXBRL_WEB.html"
        ),
        filename="NSE_Q1FY27_standalone_xbrl.html",
        publication_date="2026-07-17",
    ),
    DocumentSpec(
        financial_year="FY27",
        quarter="Q1",
        period="Q1 FY27",
        document_type="xbrl",
        report_type="consolidated",
        source_url=(
            "https://nsearchives.nseindia.com/corporate/ixbrl/"
            "INTEGRATED_FILING_INDAS_175608_17072026195004_iXBRL_WEB.html"
        ),
        filename="NSE_Q1FY27_consolidated_xbrl.html",
        publication_date="2026-07-17",
    ),
    DocumentSpec(
        financial_year="FY27",
        quarter="Q1",
        period="Q1 FY27",
        document_type="media_release",
        report_type="consolidated",
        source_url=(
            "https://www.ril.com/sites/default/files/2026-07/"
            "Media_Release_RIL_Q1_FY2026-27_Financial_and_Operational_Performance.pdf"
        ),
        filename="RIL_Q1FY27_media_release.pdf",
        publication_date="2026-07-17",
    ),
]

# Minimum plausible sizes — catches block pages / truncated downloads.
MIN_BYTES: dict[str, int] = {
    "NSE_Q1FY27_standalone_xbrl.html": 10_000,
    "NSE_Q1FY27_consolidated_xbrl.html": 10_000,
    "RIL_Q1FY27_media_release.pdf": 100_000,
}

EXPECTED_KIND: dict[str, str] = {
    "NSE_Q1FY27_standalone_xbrl.html": "html",
    "NSE_Q1FY27_consolidated_xbrl.html": "html",
    "RIL_Q1FY27_media_release.pdf": "pdf",
}
