from sqlalchemy import inspect

from rag_service.app.config import Settings
from rag_service.app.db import get_engine, init_db
from rag_service.app.models import Chunk, Document


def test_sqlite_schema_contains_search_text_and_backfills_derived_text():
    settings = Settings(database_url="sqlite+pysqlite:///:memory:")
    engine = init_db(get_engine(settings), settings=settings)
    assert "search_text" in {column["name"] for column in inspect(engine).get_columns("chunks")}

    from rag_service.app.db import get_session_factory

    factory = get_session_factory(engine=engine)
    with factory() as session:
        document = Document(
            source_path="docs/pgvector.md",
            source_type="markdown",
            title="pgvector 指南",
            file_hash="a" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add(document)
        session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                content="PostgreSQL 向量检索",
                heading_path=["向量"],
                token_count=3,
                embedding=[0.0] * settings.embedding_dimensions,
            )
        )
        session.commit()

    # Re-running init_db is the explicit backfill step for rows created by an
    # older ingestion process that did not populate the derived column.
    init_db(engine, settings=settings)

    with factory() as session:
        chunk = session.query(Chunk).one()
        assert "pgvector" in chunk.search_text
        assert "向量" in chunk.search_text


def test_postgres_index_ddl_is_idempotent_and_uses_static_identifiers(monkeypatch):
    from rag_service.app import db

    class FakeConnection:
        def __init__(self, statements):
            self.statements = statements

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, statement, *args, **kwargs):
            self.statements.append(str(statement))
            return []

    class FakeEngine:
        class Dialect:
            name = "postgresql"

        dialect = Dialect()

        def __init__(self):
            self.statements = []

        def begin(self):
            return FakeConnection(self.statements)

    engine = FakeEngine()
    monkeypatch.setattr(db.Base.metadata, "create_all", lambda _engine: None)
    db.init_db(
        engine,
        settings=Settings(database_url="postgresql+psycopg://rag:rag@localhost/ragdb"),
    )
    db.init_db(
        engine,
        settings=Settings(database_url="postgresql+psycopg://rag:rag@localhost/ragdb"),
    )
    ddl = "\n".join(engine.statements)
    assert "ADD COLUMN IF NOT EXISTS search_text" in ddl
    assert "to_tsvector('simple', search_text)" in ddl
    assert "CREATE INDEX IF NOT EXISTS" in ddl
