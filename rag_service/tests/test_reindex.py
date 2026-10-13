from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeEmbeddingProvider:
    def __init__(self, vectors: list[list[float]], *, name: str = "fake") -> None:
        self.vectors = vectors
        self.provider_name = name
        self.model = "fake-model"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.vectors[: len(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def test_validate_embedding_provider_reports_dimension_without_credentials():
    from rag_service.app.reindex import validate_embedding_provider

    report = validate_embedding_provider(
        FakeEmbeddingProvider([[0.0, 1.0, 0.0, 0.0]]),
        expected_dimensions=4,
        sample_text="dimension probe",
    )

    assert report.provider_name == "fake"
    assert report.model_name == "fake-model"
    assert report.configured_dimension == 4
    assert report.observed_dimension == 4
    assert report.ok is True
    assert "api_key" not in report.to_dict()


def test_validate_embedding_provider_records_mismatch_as_not_ok():
    from rag_service.app.reindex import validate_embedding_provider

    report = validate_embedding_provider(
        FakeEmbeddingProvider([[0.0, 1.0]]),
        expected_dimensions=4,
    )

    assert report.ok is False
    assert report.observed_dimension == 2


def test_rebuild_derived_index_requires_confirmation_before_postgres_changes(monkeypatch):
    from rag_service.app.config import Settings
    from rag_service.app import reindex

    settings = Settings(database_url="postgresql+psycopg://rag:rag@localhost/ragdb")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("destructive operation should not run")

    monkeypatch.setattr(reindex, "_drop_derived_objects", forbidden)

    with pytest.raises(PermissionError, match="confirm"):
        reindex.rebuild_derived_index(settings, confirm=False)
    assert called is False


def test_rebuild_summary_is_serializable_and_preserves_source_root(tmp_path, monkeypatch):
    from rag_service.app.config import Settings
    from rag_service.app import reindex

    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "note.md").write_text("source", encoding="utf-8")
    settings = Settings(knowledge_root=root, database_url="sqlite+pysqlite:///:memory:")
    fake_summary = SimpleNamespace(
        scanned_files=1,
        processed_files=1,
        skipped_files=0,
        failed_files=0,
        deleted_documents=0,
        chunk_count=1,
        job_id=7,
    )
    monkeypatch.setattr(reindex, "_drop_derived_objects", lambda *args, **kwargs: None)
    monkeypatch.setattr(reindex.IngestionService, "scan", lambda self, force=False: fake_summary)
    result = reindex.rebuild_derived_index(settings, confirm=True)

    assert result["status"] == "completed"
    assert result["source_root"] == str(root.resolve())
    assert result["ingestion"]["processed_files"] == 1
    assert (root / "note.md").read_text(encoding="utf-8") == "source"
