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

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .models import Base, configure_embedding_dimensions


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
    """Create extensions and all application tables.

    PostgreSQL extensions are created only for a PostgreSQL engine.  This makes
    the same function useful for SQLite unit tests while keeping the production
    setup explicit and repeatable.
    """

    settings = settings or get_settings()
    engine = engine or get_engine(settings)
    configure_embedding_dimensions(settings.embedding_dimensions)
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    Base.metadata.create_all(engine)
    return engine


def dispose_cached_engines() -> None:
    """Dispose cached engines (primarily useful for test teardown)."""
    # ``cache_info`` does not expose cached values.  Clearing the cache still
    # releases references to session factories and their engines.
    _cached_session_factory.cache_clear()
