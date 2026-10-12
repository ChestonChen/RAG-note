from pathlib import Path


def test_settings_use_project_defaults(monkeypatch):
    from rag_service.app.config import Settings

    monkeypatch.delenv("KNOWLEDGE_ROOT", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = Settings()

    assert settings.knowledge_root == Path("knowledge")
    assert settings.database_url.startswith("postgresql+psycopg://")


def test_settings_read_environment_overrides(monkeypatch, tmp_path):
    from rag_service.app.config import Settings

    monkeypatch.setenv("KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "320")

    settings = Settings()

    assert settings.knowledge_root == tmp_path
    assert settings.chunk_max_tokens == 320


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
