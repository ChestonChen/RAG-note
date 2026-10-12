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
