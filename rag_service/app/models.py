"""SQLAlchemy persistence models for the rebuildable knowledge index."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

try:  # pgvector is optional while running lightweight unit tests.
    from pgvector.sqlalchemy import Vector as PgVector
except ImportError:  # pragma: no cover - exercised only in minimal installs
    PgVector = None  # type: ignore[assignment,misc]


# The hashing provider uses 384 by default.  The database column itself is
# intentionally unbounded (``VECTOR`` rather than ``VECTOR(384)``) so a user
# can switch to an embedding model with another configured dimension without
# rewriting the table definition.  Ingestion validates every vector against
# ``Settings.embedding_dimensions`` before inserting it.
DEFAULT_EMBEDDING_DIMENSIONS = 384


class Base(DeclarativeBase):
    pass


class EmbeddingType(TypeDecorator[list[float]]):
    """Use pgvector on PostgreSQL and JSON elsewhere.

    SQLite is a convenient test database, but pgvector's custom type cannot be
    compiled by SQLite.  This adapter keeps the model/table contract identical
    while allowing tests to exercise transactions and cascade deletes locally.
    """

    impl = JSON
    cache_ok = True

    class Comparator(TypeDecorator.Comparator[list[float]]):
        """Expose pgvector operators while retaining SQLite JSON fallback."""

        def cosine_distance(self, other: object):
            return self.op("<=>", return_type=Float)(other)

        def l2_distance(self, other: object):
            return self.op("<->", return_type=Float)(other)

    comparator_factory = Comparator

    def __init__(self, dimensions: int | None = None, **kwargs: Any) -> None:
        self.dimensions = dimensions
        super().__init__(**kwargs)

    def load_dialect_impl(self, dialect):  # type: ignore[no-untyped-def]
        if dialect.name == "postgresql" and PgVector is not None:
            return dialect.type_descriptor(PgVector(self.dimensions))
        return dialect.type_descriptor(JSON())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Document(TimestampMixin, Base):
    """One source file and its current indexing metadata."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_path: Mapped[str] = mapped_column(String(2048), unique=True, index=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_signature: Mapped[str] = mapped_column(String(128), nullable=False)
    chunking_signature: Mapped[str] = mapped_column(String(256), nullable=False)
    embedding_signature: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="ready", index=True, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_mtime: Mapped[float | None] = mapped_column(Float, nullable=True)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
    )


class Chunk(Base):
    """A retrievable text segment with source-location metadata and embedding."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_order", "document_id", "chunk_index", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    heading_path: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    # Derived from the owning document metadata and chunk content.  The
    # ingestion path populates it; init_db backfills rows created by older
    # versions before PostgreSQL full-text indexes are used.
    search_text: Mapped[str] = mapped_column(
        Text, default="", server_default=text("''"), nullable=False
    )
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        EmbeddingType(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class IngestionJob(TimestampMixin, Base):
    """Persistent status and progress for one ingestion run."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="running", index=True, nullable=False)
    force: Mapped[bool] = mapped_column(default=False, nullable=False)
    active_key: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    current_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    current_file: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    scanned_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    deleted_documents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    failures: Mapped[list["IngestionFailure"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="IngestionFailure.id"
    )


class IngestionFailure(TimestampMixin, Base):
    """A redacted failure associated with one ingestion job and source file."""

    __tablename__ = "ingestion_failures"
    __table_args__ = (Index("ix_ingestion_failures_job_id", "job_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False)
    source_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    error_type: Mapped[str] = mapped_column(String(128), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)

    job: Mapped[IngestionJob] = relationship(back_populates="failures")


class EvalRun(TimestampMixin, Base):
    """Persistent metadata and aggregate metrics for one evaluation run."""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True, nullable=False)
    active_key: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)
    total_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    tuning: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="EvalResult.case_index"
    )


class EvalResult(TimestampMixin, Base):
    """One redacted case result belonging to an evaluation run."""

    __tablename__ = "eval_results"
    __table_args__ = (Index("ix_eval_results_run_case", "run_id", "case_id", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False)
    case_index: Mapped[int] = mapped_column(Integer, nullable=False)
    case_id: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="completed", nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[EvalRun] = relationship(back_populates="results")


class QALog(TimestampMixin, Base):
    """Optional audit record for a generated answer."""

    __tablename__ = "qa_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    retrieval_debug: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class Conversation(TimestampMixin, Base):
    """A persistent chat conversation and its message timeline."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(512), default="New conversation", nullable=False)
    archived: Mapped[bool] = mapped_column(default=False, index=True, nullable=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="(Message.created_at, Message.id)",
    )


class Message(Base):
    """One user or assistant message persisted in a conversation."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="complete", nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    retrieval_debug: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    confidence: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mode: Mapped[str] = mapped_column(String(32), default="kb_only", nullable=False)
    web_sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


def configure_embedding_dimensions(dimensions: int) -> None:
    """Set the pgvector column dimension before creating the schema.

    PostgreSQL vector columns are dimensioned at DDL time.  The configured
    provider is therefore applied when ``init_db`` runs; changing it later
    requires rebuilding the index/database rather than silently mixing models.
    SQLite tests continue to use JSON and do not depend on this value.
    """

    if dimensions < 8:
        raise ValueError("embedding dimensions must be at least 8")
    column = Chunk.__table__.c.embedding
    if isinstance(column.type, EmbeddingType):
        column.type.dimensions = dimensions


__all__ = [
    "Base",
    "Chunk",
    "Document",
    "EmbeddingType",
    "configure_embedding_dimensions",
    "IngestionJob",
    "IngestionFailure",
    "EvalRun",
    "EvalResult",
    "Conversation",
    "Message",
    "QALog",
]
