"""Database setup helpers.

The application uses PostgreSQL in production, but the engine/session helpers
are deliberately lazy.  Importing the application (or running unit tests)
therefore does not require a running database.  Tests can pass an explicit
SQLite URL to ``get_engine`` and use the same SQLAlchemy models.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine, inspect, select, text, update
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .models import (
    Base,
    Chunk,
    Document,
    EvalResult,
    EvalRun,
    IngestionFailure,
    IngestionJob,
    configure_embedding_dimensions,
)
from .search_text import build_search_text


# pgvector's standard ``vector`` type supports HNSW indexes up to 2,000
# dimensions. Higher-dimensional providers can still use stored vectors and
# lexical retrieval; they simply skip this optional index.
_PGVECTOR_HNSW_MAX_DIMENSIONS = 2000


def get_engine(settings: Settings | None = None, *, database_url: str | None = None) -> Engine:
    """Create an SQLAlchemy engine without opening a database connection.

    ``database_url`` is useful for tests and takes precedence over the
    configured URL.  The function intentionally does not cache explicit test
    engines; callers own those engines and may dispose them when finished.
    """

    if database_url is None:
        settings = settings or get_settings()
        database_url = settings.database_url

    kwargs: dict[str, object] = {"future": True, "pool_pre_ping": True}
    if database_url.startswith("postgresql"):
        kwargs["connect_args"] = {"connect_timeout": 3}
    # SQLite's in-memory database is commonly used by unit tests.  Its default
    # pool would create a new database for every connection, so share one
    # connection across the thread in that special case.
    if database_url.startswith("sqlite") and ":memory:" in database_url:
        from sqlalchemy.pool import StaticPool

        kwargs.update(connect_args={"check_same_thread": False}, poolclass=StaticPool)
    return create_engine(database_url, **kwargs)


@lru_cache(maxsize=8)
def _cached_session_factory(database_url: str) -> sessionmaker[Session]:
    """Return a process-local factory for the normal configured URL."""

    return sessionmaker(bind=get_engine(database_url=database_url), autoflush=False, expire_on_commit=False)


def get_session_factory(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    database_url: str | None = None,
) -> sessionmaker[Session]:
    """Return a ``sessionmaker`` suitable for dependency injection."""

    if engine is not None:
        return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    settings = settings or get_settings()
    return _cached_session_factory(database_url or settings.database_url)


@contextmanager
def session_scope(
    settings: Settings | None = None,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> Iterator[Session]:
    """Open, commit, and close one session.

    This helper is intentionally small so services can use a transaction
    boundary per source file and isolate ingestion failures.
    """

    factory = session_factory or get_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None, settings: Settings | None = None) -> Engine:
    """Create extensions, tables, derived search data and retrieval indexes.

    PostgreSQL extensions are created only for a PostgreSQL engine.  This makes
    the same function useful for SQLite unit tests while keeping the production
    setup explicit and repeatable.  ``create_all`` creates new tables; the
    explicit column/backfill statements below handle existing databases.
    """

    settings = settings or get_settings()
    engine = engine or get_engine(settings)
    configure_embedding_dimensions(settings.embedding_dimensions)
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    Base.metadata.create_all(engine)
    _ensure_ingestion_job_columns(engine)
    _ensure_eval_columns(engine)
    _ensure_search_text_column(engine)
    _backfill_search_text(engine)
    if engine.dialect.name == "postgresql":
        # Keep the two expensive retrieval channels index-backed once the
        # knowledge base grows beyond the local smoke-test size.  These are
        # idempotent and can also be created by rerunning init_db.py after an
        # existing database has been populated.
        with engine.begin() as connection:
            if settings.embedding_dimensions <= _PGVECTOR_HNSW_MAX_DIMENSIONS:
                connection.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                        "ON chunks USING hnsw (embedding vector_cosine_ops)"
                    )
                )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chunks_content_trgm "
                    "ON chunks USING gin (content gin_trgm_ops)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chunks_search_text_fts "
                    "ON chunks USING gin (to_tsvector('simple', search_text))"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chunks_search_text_trgm "
                    "ON chunks USING gin (search_text gin_trgm_ops)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_documents_title_trgm "
                    "ON documents USING gin (title gin_trgm_ops)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_documents_source_path_trgm "
                    "ON documents USING gin (source_path gin_trgm_ops)"
                )
            )
    return engine


def _ensure_ingestion_job_columns(engine: Engine) -> None:
    """Add async job fields to databases created by earlier versions."""

    if not isinstance(engine, Engine):
        return
    if engine.dialect.name not in {"sqlite", "postgresql"}:
        return
    inspector = inspect(engine)
    if not inspector.has_table("ingestion_jobs"):
        return
    columns = {column["name"] for column in inspector.get_columns("ingestion_jobs")}
    definitions = {
        "force": "BOOLEAN NOT NULL DEFAULT FALSE",
        "active_key": "INTEGER NULL",
        "started_at": "TIMESTAMP WITH TIME ZONE NULL",
        "finished_at": "TIMESTAMP WITH TIME ZONE NULL",
        "current_stage": "VARCHAR(32) NULL",
        "current_file": "VARCHAR(2048) NULL",
    }
    with engine.begin() as connection:
        for name, definition in definitions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE ingestion_jobs ADD COLUMN {name} {definition}"))
        # Older SQLite databases may contain the long synchronous terminal
        # marker. Normalize it before the bounded job status contract is used.
        connection.execute(
            text("UPDATE ingestion_jobs SET status = 'failed' WHERE status = 'completed_with_errors'")
        )
        if engine.dialect.name == "sqlite":
            connection.execute(
                text("CREATE UNIQUE INDEX IF NOT EXISTS ix_ingestion_jobs_active_key ON ingestion_jobs(active_key)")
            )
        else:
            connection.execute(
                text("CREATE UNIQUE INDEX IF NOT EXISTS ix_ingestion_jobs_active_key ON ingestion_jobs(active_key)")
            )


def _ensure_eval_columns(engine: Engine) -> None:
    """Backfill columns for evaluation tables created by an earlier build."""

    if not isinstance(engine, Engine):
        return
    if engine.dialect.name not in {"sqlite", "postgresql"}:
        return
    inspector = inspect(engine)
    if inspector.has_table("eval_runs"):
        columns = {column["name"] for column in inspector.get_columns("eval_runs")}
        definitions = {
            "active_key": "INTEGER NULL",
            "total_cases": "INTEGER NOT NULL DEFAULT 0",
            "completed_cases": "INTEGER NOT NULL DEFAULT 0",
            "failed_cases": "INTEGER NOT NULL DEFAULT 0",
            "progress": "FLOAT NOT NULL DEFAULT 0",
            "started_at": "TIMESTAMP WITH TIME ZONE NULL",
            "finished_at": "TIMESTAMP WITH TIME ZONE NULL",
            "error_message": "TEXT NULL",
            "metrics": "JSON NOT NULL DEFAULT '{}'",
            "config": "JSON NOT NULL DEFAULT '{}'",
            "profile": "VARCHAR(64) NULL",
            "dataset": "JSON NOT NULL DEFAULT '{}'",
            "tuning": "JSON NOT NULL DEFAULT '{}'",
        }
        with engine.begin() as connection:
            for name, definition in definitions.items():
                if name not in columns:
                    connection.execute(text(f"ALTER TABLE eval_runs ADD COLUMN {name} {definition}"))
            connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_eval_runs_active_key ON eval_runs(active_key)"))
    if inspector.has_table("eval_results"):
        columns = {column["name"] for column in inspector.get_columns("eval_results")}
        definitions = {
            "case_index": "INTEGER NOT NULL DEFAULT 0",
            "case_id": "VARCHAR(256) NOT NULL DEFAULT ''",
            "status": "VARCHAR(16) NOT NULL DEFAULT 'completed'",
            "result": "JSON NOT NULL DEFAULT '{}'",
            "error_message": "TEXT NULL",
        }
        with engine.begin() as connection:
            for name, definition in definitions.items():
                if name not in columns:
                    connection.execute(text(f"ALTER TABLE eval_results ADD COLUMN {name} {definition}"))
            connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_eval_results_run_case ON eval_results(run_id, case_id)"))


def _ensure_search_text_column(engine: Engine) -> None:
    """Add the derived column to databases created before this schema version."""

    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_text TEXT NOT NULL DEFAULT ''")
            )
        return
    if engine.dialect.name != "sqlite":
        return
    inspector = inspect(engine)
    if not inspector.has_table("chunks"):
        return
    columns = {column["name"] for column in inspector.get_columns("chunks")}
    if "search_text" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE chunks ADD COLUMN search_text TEXT NOT NULL DEFAULT ''"))


def _backfill_search_text(engine: Engine) -> None:
    """Recompute search text using bound values for every existing chunk."""

    with engine.begin() as connection:
        result = connection.execute(
            select(
                Chunk.id,
                Chunk.content,
                Chunk.heading_path,
                Document.title,
                Document.source_path,
            ).join(Document, Chunk.document_id == Document.id)
        )
        if result is None:
            rows = []
        else:
            rows = result.all() if hasattr(result, "all") else list(result)
        for row in rows:
            chunk_id, content, heading_path, title, source_path = row[:5]
            derived = build_search_text(
                title=str(title or ""),
                heading_path=heading_path or [],
                source_path=str(source_path or ""),
                content=str(content or ""),
            )
            connection.execute(
                update(Chunk.__table__)
                .where(Chunk.__table__.c.id == chunk_id)
                .values(search_text=derived)
            )


def dispose_cached_engines() -> None:
    """Dispose cached engines (primarily useful for test teardown)."""
    # ``cache_info`` does not expose cached values.  Clearing the cache still
    # releases references to session factories and their engines.
    _cached_session_factory.cache_clear()


def drop_derived_tables(engine: Engine) -> list[str]:
    """Drop only rebuildable index tables while preserving source files."""

    derived_tables = (
        Chunk.__table__,
        IngestionFailure.__table__,
        Document.__table__,
        IngestionJob.__table__,
    )
    removed: list[str] = []
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            for table in derived_tables:
                connection.execute(text(f"DROP TABLE IF EXISTS {table.name} CASCADE"))
                removed.append(table.name)
        return removed
    with engine.begin() as connection:
        for table in derived_tables:
            if inspect(connection).has_table(table.name):
                table.drop(connection, checkfirst=True)
                removed.append(table.name)
    return removed
