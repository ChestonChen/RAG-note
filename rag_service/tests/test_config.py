from pathlib import Path


def test_settings_use_project_defaults(monkeypatch):
    from rag_service.app.config import Settings

    monkeypatch.delenv("KNOWLEDGE_ROOT", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RETRIEVAL_PER_DOCUMENT_LIMIT", raising=False)
    monkeypatch.delenv("EVAL_CASES_ROOT", raising=False)

    settings = Settings(_env_file=None)

    assert settings.knowledge_root == Path("knowledge")
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.retrieval_per_document_limit == 5
    assert settings.eval_cases_root == Path("eval")
    assert settings.resolved_eval_cases_root() == settings.project_root / "eval"


def test_settings_read_environment_overrides(monkeypatch, tmp_path):
    from rag_service.app.config import Settings

    monkeypatch.setenv("KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "320")
    monkeypatch.setenv("EVAL_CASES_ROOT", str(tmp_path / "approved-eval"))

    settings = Settings()

    assert settings.knowledge_root == tmp_path
    assert settings.chunk_max_tokens == 320
    assert settings.eval_cases_root == tmp_path / "approved-eval"


def test_section_aware_chunking_signature_reindexes_legacy_chunks():
    from rag_service.app.config import Settings
    from rag_service.app.ingestion import should_reindex

    settings = Settings(
        chunking_version="chunk-v1",
        chunk_max_tokens=520,
        chunk_overlap_tokens=80,
    )
    legacy = {
        "status": "ready",
        "file_hash": "a" * 64,
        "parser_signature": "parser-v1",
        "chunking_signature": "parser-v1:chunk-v1:max=520:overlap=80",
        "embedding_signature": settings.embedding_signature(),
    }

    assert "section-aware-v2" in settings.chunking_signature()
    assert should_reindex(
        legacy,
        file_hash=legacy["file_hash"],
        parser_signature=settings.parser_version,
        chunking_signature=settings.chunking_signature(),
        embedding_signature=settings.embedding_signature(),
    )


def test_settings_expose_optional_postgres_trigram_channel():
    from rag_service.app.config import Settings

    settings = Settings()

    assert settings.retrieval_trigram_candidates == 30
    assert settings.retrieval_trigram_rrf_weight == 0.5
    assert settings.retrieval_vector_min_score == 0.12
    assert settings.retrieval_trigram_min_similarity == 0.1


def test_embedding_signature_changes_when_provider_model_or_dimension_changes():
    from rag_service.app.config import Settings

    baseline = Settings()
    assert baseline.embedding_signature() != Settings(
        embedding_provider="openai_compatible",
        embedding_api_key="secret",
        embedding_model=baseline.embedding_model,
        embedding_dimensions=baseline.embedding_dimensions,
    ).embedding_signature()
    assert baseline.embedding_signature() != Settings(
        embedding_model="another-model",
        embedding_dimensions=baseline.embedding_dimensions,
    ).embedding_signature()
    assert baseline.embedding_signature() != Settings(
        embedding_dimensions=baseline.embedding_dimensions + 1,
    ).embedding_signature()


def test_init_db_skips_hnsw_for_dimensions_above_pgvector_limit(monkeypatch):
    from rag_service.app import db
    from rag_service.app.config import Settings
    from rag_service.app.models import Chunk

    class FakeConnection:
        def __init__(self, statements):
            self.statements = statements

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, statement):
            self.statements.append(str(statement))

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

    original_dimensions = Chunk.__table__.c.embedding.type.dimensions
    try:
        db.init_db(
            engine,
            settings=Settings(
                database_url="postgresql+psycopg://rag:rag@localhost/ragdb",
                embedding_dimensions=2048,
            ),
        )
    finally:
        Chunk.__table__.c.embedding.type.dimensions = original_dimensions

    assert not any("embedding_hnsw" in statement for statement in engine.statements)
    assert any("chunks_content_trgm" in statement for statement in engine.statements)
