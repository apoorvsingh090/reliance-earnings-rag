"""Milestone 2 pipeline: parse registered documents -> map facts -> upsert DB.

Idempotent: documents and facts are upserted by identity (hash / fact key),
so re-running never duplicates. Per-document JSON artifacts go to
data/extracted/ for transparency and debugging.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import PROJECT_ROOT, get_settings
from app.extraction.pdf import extract_annexure_facts, parse_pdf
from app.extraction.xbrl import extract_pnl_facts, extract_segment_facts, parse_xbrl
from app.ingestion.registry import DocumentRegistry
from app.models.database import count_facts, get_session, init_db, upsert_document, upsert_fact
from app.models.document import ReportType
from app.models.financial import FinancialFact

log = logging.getLogger(__name__)


def _resolve_raw_path(local_path: str) -> Path:
    p = Path(local_path)
    if p.exists():
        return p
    return PROJECT_ROOT / local_path


def run_extraction(engine=None) -> dict:
    settings = get_settings()
    settings.ensure_dirs()
    registry = DocumentRegistry(settings.registry_path)
    records = registry.all()
    if not records:
        raise RuntimeError("registry is empty — run scripts/ingest_q1fy27.py first")

    engine = init_db(engine)
    session = get_session(engine)
    summary = {"documents": 0, "facts_created": 0, "facts_reused": 0, "per_document": {}}

    for record in records:
        raw_path = _resolve_raw_path(record.local_path)
        report_type = ReportType(record.report_type.value)
        facts: list[FinancialFact] = []
        artifact: dict = {
            "filename": record.filename,
            "document_type": record.document_type.value,
            "report_type": record.report_type.value,
            "period": record.period,
        }
        if record.document_type.value == "xbrl":
            parsed = parse_xbrl(raw_path)
            # Guardrail: the document itself must agree with our spec basis.
            stated = parsed.stated_report_basis.lower()
            if stated and stated != report_type.value:
                raise ValueError(
                    f"{record.filename}: spec says {report_type.value} but document states {stated}"
                )
            artifact["units"] = {"currency": parsed.currency, "amount_unit": parsed.amount_unit}
            artifact["stated_basis"] = parsed.stated_report_basis
            artifact["audit_status"] = parsed.audit_status
            artifact["tables"] = len(parsed.tables)
            facts += extract_pnl_facts(
                parsed, company=record.company, ticker=record.ticker,
                financial_year=record.financial_year, quarter=record.quarter,
                period=record.period, report_type=report_type,
                source_document_id=record.document_hash,
            )
            facts += extract_segment_facts(
                parsed, company=record.company, ticker=record.ticker,
                financial_year=record.financial_year, quarter=record.quarter,
                period=record.period, report_type=report_type,
                source_document_id=record.document_hash,
            )
        elif record.document_type.value == "media_release":
            parsed = parse_pdf(raw_path)
            if parsed.flagged_pages:
                log.warning("OCR fallback needed for pages %s (flagged, not parsed as tables)",
                            parsed.flagged_pages)
            artifact["pages"] = len(parsed.pages)
            artifact["flagged_pages"] = parsed.flagged_pages
            artifact["printed_labels_sample"] = [p.printed_label for p in parsed.pages[:5]]
            facts += extract_annexure_facts(
                parsed, company=record.company, ticker=record.ticker,
                source_document_id=record.document_hash,
            )
        else:
            log.info("no M2 extractor for %s — skipped", record.document_type.value)
            continue

        upsert_document(session, record)
        created = reused = 0
        for fact in facts:
            _, is_new = upsert_fact(session, fact)
            if is_new:
                created += 1
            else:
                reused += 1
        summary["documents"] += 1
        summary["facts_created"] += created
        summary["facts_reused"] += reused
        summary["per_document"][record.filename] = {"created": created, "reused": reused}

        out_path = settings.extracted_dir / f"{Path(record.filename).stem}.json"
        artifact["facts"] = [f.model_dump(mode="json") for f in facts]
        out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    summary["total_facts"] = count_facts(session)
    combined = settings.extracted_dir / "facts_all.json"
    combined.write_text(
        json.dumps(
            {"total_facts": summary["total_facts"], "per_document": summary["per_document"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    session.close()
    return summary
