"""Structured store: documents + financial facts (SQLAlchemy).

PostgreSQL (+pgvector later) is the production target; when it is unreachable
(local dev without docker) we fall back to a SQLite file so Milestones 2-5
stay runnable everywhere. The ORM schema is identical on both backends.
Upserts make the extraction pipeline idempotent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Float, ForeignKey, LargeBinary, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    document_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    company: Mapped[str] = mapped_column(String(200))
    ticker: Mapped[str] = mapped_column(String(32))
    financial_year: Mapped[str] = mapped_column(String(16))
    quarter: Mapped[str] = mapped_column(String(8))
    period: Mapped[str] = mapped_column(String(32))
    document_type: Mapped[str] = mapped_column(String(64))
    report_type: Mapped[str] = mapped_column(String(32), index=True)
    source_url: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(String(256))
    publication_date: Mapped[str] = mapped_column(String(16))
    file_size: Mapped[int] = mapped_column()
    ingestion_timestamp: Mapped[str] = mapped_column(String(64))


class FinancialFactRow(Base):
    __tablename__ = "financial_facts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    company: Mapped[str] = mapped_column(String(200))
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    financial_year: Mapped[str] = mapped_column(String(16))
    quarter: Mapped[str | None] = mapped_column(String(8), nullable=True)
    period: Mapped[str] = mapped_column(String(32), index=True)
    period_type: Mapped[str] = mapped_column(String(16))
    metric: Mapped[str] = mapped_column(String(128), index=True)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32))
    currency: Mapped[str] = mapped_column(String(8))
    segment: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    report_type: Mapped[str] = mapped_column(String(32), index=True)
    source_document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.document_hash")
    )
    source_location: Mapped[str] = mapped_column(Text)
    extraction_method: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[str] = mapped_column(
        String(64), default=lambda: datetime.now(timezone.utc).isoformat()
    )

    __table_args__ = (
        UniqueConstraint(
            "source_document_id", "metric", "period", "report_type", "segment",
            name="uq_fact_identity",
        ),
    )


def sqlite_default_path() -> Path:
    from app.config import get_settings

    return get_settings().processed_dir / "reliance_earnings.db"


def get_engine(url: str | None = None):
    """Engine for url (or settings DATABASE_URL), with SQLite fallback."""
    from app.config import get_settings

    candidate = url or get_settings().database_url
    if candidate.startswith("postgresql"):
        try:
            engine = create_engine(candidate, pool_pre_timeout=3, connect_args={"connect_timeout": 3})
            with engine.connect():
                pass
            log.info("connected to PostgreSQL")
            return engine
        except Exception as exc:  # noqa: BLE001 — dev fallback
            log.warning("PostgreSQL unreachable (%s); using local SQLite fallback", exc)
    fallback = sqlite_default_path()
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{fallback}")


def init_db(engine=None):
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None) -> Session:
    engine = engine or get_engine()
    return sessionmaker(bind=engine)()


def upsert_document(session: Session, record) -> DocumentRow:
    """record: app.models.document.DocumentRecord."""
    row = session.get(DocumentRow, record.document_hash)
    if row is not None:
        return row
    row = DocumentRow(
        document_hash=record.document_hash,
        company=record.company,
        ticker=record.ticker,
        financial_year=record.financial_year,
        quarter=record.quarter,
        period=record.period,
        document_type=record.document_type.value,
        report_type=record.report_type.value,
        source_url=record.source_url,
        filename=record.filename,
        publication_date=record.publication_date,
        file_size=record.file_size,
        ingestion_timestamp=record.ingestion_timestamp,
    )
    session.add(row)
    session.commit()
    return row


def upsert_fact(session: Session, fact) -> tuple[FinancialFactRow, bool]:
    """fact: app.models.financial.FinancialFact. Returns (row, created)."""
    stmt = select(FinancialFactRow).where(
        FinancialFactRow.source_document_id == fact.source_document_id,
        FinancialFactRow.metric == fact.metric,
        FinancialFactRow.period == fact.period,
        FinancialFactRow.report_type == fact.report_type.value,
        FinancialFactRow.segment.is_(None) if fact.segment is None
        else FinancialFactRow.segment == fact.segment,
    )
    existing = session.execute(stmt).scalars().first()
    if existing is not None:
        if abs(existing.value - fact.value) > 1e-9 or existing.unit != fact.unit:
            # Same identity, different value = parser conflict. Keep the first
            # fact and warn loudly instead of silently overwriting numbers.
            log.warning(
                "fact conflict kept-first: %s %s %s seg=%s existing=%s new=%s (%s)",
                fact.metric, fact.period, fact.report_type.value, fact.segment,
                existing.value, fact.value, fact.source_location,
            )
        return existing, False
    row = FinancialFactRow(
        company=fact.company,
        ticker=fact.ticker,
        financial_year=fact.financial_year,
        quarter=fact.quarter,
        period=fact.period,
        period_type=fact.period_type,
        metric=fact.metric,
        value=fact.value,
        unit=fact.unit,
        currency=fact.currency,
        segment=fact.segment,
        report_type=fact.report_type.value,
        source_document_id=fact.source_document_id,
        source_location=fact.source_location,
        extraction_method=fact.extraction_method,
    )
    session.add(row)
    session.commit()
    return row, True


def count_facts(session: Session) -> int:
    from sqlalchemy import func

    return session.execute(select(func.count()).select_from(FinancialFactRow)).scalar() or 0


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), ForeignKey("documents.document_hash"), index=True)
    page_number: Mapped[int | None] = mapped_column(nullable=True)
    printed_label: Mapped[str] = mapped_column(String(32), default="")
    section: Mapped[str] = mapped_column(String(200), default="")
    chunk_type: Mapped[str] = mapped_column(String(16), default="text")
    text: Mapped[str] = mapped_column(Text)
    company: Mapped[str] = mapped_column(String(200), default="")
    ticker: Mapped[str] = mapped_column(String(32), index=True, default="")
    financial_year: Mapped[str] = mapped_column(String(16), default="")
    quarter: Mapped[str] = mapped_column(String(8), index=True, default="")
    period: Mapped[str] = mapped_column(String(32), index=True, default="")
    document_type: Mapped[str] = mapped_column(String(64), index=True, default="")
    report_type: Mapped[str] = mapped_column(String(32), index=True, default="")
    source_location: Mapped[str] = mapped_column(Text, default="")


class ChunkEmbeddingRow(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[str] = mapped_column(String(32), ForeignKey("chunks.chunk_id"), primary_key=True)
    model: Mapped[str] = mapped_column(String(128))
    dim: Mapped[int] = mapped_column()
    embedding: Mapped[bytes] = mapped_column(LargeBinary)


def upsert_chunk(session: Session, chunk) -> tuple[ChunkRow, bool]:
    """chunk: app.models.chunk.Chunk. Returns (row, created)."""
    row = session.get(ChunkRow, chunk.chunk_id)
    if row is not None:
        return row, False
    row = ChunkRow(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        page_number=chunk.page_number,
        printed_label=chunk.printed_label,
        section=chunk.section,
        chunk_type=chunk.chunk_type,
        text=chunk.text,
        company=chunk.company,
        ticker=chunk.ticker,
        financial_year=chunk.financial_year,
        quarter=chunk.quarter,
        period=chunk.period,
        document_type=chunk.document_type,
        report_type=chunk.report_type.value,
        source_location=chunk.source_location,
    )
    session.add(row)
    session.commit()
    return row, True


def upsert_embedding(
    session: Session, chunk_id: str, vector, model: str
) -> tuple[ChunkEmbeddingRow, bool]:
    import numpy as np

    arr = np.asarray(vector, dtype=np.float32)
    row = session.get(ChunkEmbeddingRow, chunk_id)
    if row is not None and row.model == model and row.dim == arr.shape[0]:
        return row, False
    row = ChunkEmbeddingRow(
        chunk_id=chunk_id, model=model, dim=int(arr.shape[0]), embedding=arr.tobytes()
    )
    session.merge(row)
    session.commit()
    return row, True


def count_chunks(session: Session) -> int:
    from sqlalchemy import func

    return session.execute(select(func.count()).select_from(ChunkRow)).scalar() or 0


def count_embeddings(session: Session) -> int:
    from sqlalchemy import func

    return session.execute(select(func.count()).select_from(ChunkEmbeddingRow)).scalar() or 0
