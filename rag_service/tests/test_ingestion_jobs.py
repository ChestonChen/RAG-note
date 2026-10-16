from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_service.app.api import create_app
from rag_service.app.config import Settings
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.ingestion import IngestionService
from rag_service.app.job_runner import ConflictError, IngestionJobRunner
from rag_service.app.models import Document, IngestionFailure, IngestionJob


class FakeEmbeddingProvider:
    dimensions = 384
    provider_name = "hashing"
    model = "feature-hashing"

    def embed_texts(self, texts):
        return [[float(index == 0) for index in range(self.dimensions)] for _ in texts]


def _runner(tmp_path, parser=None):
    root = tmp_path / "knowledge"
    root.mkdir()
    settings = Settings(
        knowledge_root=root,
        database_url="sqlite+pysqlite:///:memory:",
        embedding_provider="hashing",
        embedding_dimensions=384,
        admin_api_key="test-admin",
    )
    engine = init_db(get_engine(settings))
    factory = get_session_factory(engine=engine)
    service = IngestionService(
        settings,
        session_factory=factory,
        embedding_provider=FakeEmbeddingProvider(),
        parser=parser or (lambda path: SimpleNamespace(title=path.stem)),
        chunker=lambda document, settings: [
            SimpleNamespace(
                content=f"content for {document.title}",
                heading_path=[],
                token_count=1,
                page_from=None,
                page_to=None,
            )
        ],
    )
    return root, settings, factory, service, IngestionJobRunner(factory, service)


def _wait_terminal(runner, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job.status in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("ingestion job did not reach a terminal state")


def test_async_ingestion_persists_progress_and_completion(tmp_path):
    root, _, factory, _, runner = _runner(tmp_path)
    (root / "note.md").write_text("source", encoding="utf-8")

    queued = runner.create(force=False, run_async=True)
    completed = _wait_terminal(runner, queued.id)

    assert completed.status == "completed"
    assert completed.scanned_files == 1
    assert completed.processed_files == 1
    assert completed.current_stage == "completed"
    assert completed.started_at is not None
    assert completed.finished_at is not None
    with factory() as session:
        assert session.query(IngestionJob).count() == 1
    runner.shutdown()


def test_file_failure_is_redacted_and_does_not_stop_later_files(tmp_path):
    def parser(path):
        if path.stem == "bad":
            raise ValueError("api_key=do-not-store https://provider.example/secret")
        return SimpleNamespace(title=path.stem)

    root, _, factory, _, runner = _runner(tmp_path, parser=parser)
    (root / "bad.md").write_text("bad", encoding="utf-8")
    (root / "good.md").write_text("good", encoding="utf-8")

    job = runner.create(force=False, run_async=False)
    failures = runner.get_failures(job.id)

    assert job.status == "failed"
    assert job.failed_files == 1
    assert job.processed_files == 1
    assert [(failure.source_path, failure.error_type) for failure in failures] == [
        ("bad.md", "ValueError")
    ]
    assert "do-not-store" not in failures[0].error_message
    assert "provider.example" not in failures[0].error_message
    with factory() as session:
        assert session.query(IngestionFailure).count() == 1
        document_error = (
            session.query(Document)
            .filter_by(source_path="bad.md")
            .one()
            .error_message
            or ""
        )
    assert "do-not-store" not in document_error
    assert "provider.example" not in document_error
    assert "ValueError" in document_error
    runner.shutdown()


def test_active_ingestion_job_conflicts(tmp_path):
    _, _, factory, service, runner = _runner(tmp_path)
    entered = False
    scan = service.scan

    def blocking_scan(*args, **kwargs):
        nonlocal entered
        entered = True
        time.sleep(0.2)
        return scan(*args, **kwargs)

    runner.ingestion_service.scan = blocking_scan
    first = runner.create(force=False, run_async=True)
    deadline = time.monotonic() + 1
    while not entered and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(ConflictError):
        runner.create(force=False, run_async=True)
    _wait_terminal(runner, first.id)
    runner.shutdown()


def test_admin_ingestion_job_routes_are_protected_and_queryable(tmp_path):
    root, settings, factory, service, runner = _runner(tmp_path)
    (root / "note.md").write_text("source", encoding="utf-8")
    client = TestClient(
        create_app(
            settings=settings,
            ingestion_service=service,
            session_factory=factory,
            ingestion_job_runner=runner,
        )
    )

    assert client.get("/v1/admin/ingest/jobs").status_code == 401
    response = client.post(
        "/v1/admin/ingest/jobs",
        headers={"X-Admin-API-Key": "test-admin"},
        json={"force": False, "run_async": False},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    assert response.json()["status"] == "completed"
    assert client.get(
        "/v1/admin/ingest/jobs",
        headers={"X-Admin-API-Key": "test-admin"},
    ).json()["items"][0]["id"] == job_id
    assert client.get(
        f"/v1/admin/ingest/jobs/{job_id}",
        headers={"X-Admin-API-Key": "test-admin"},
    ).json()["processed_files"] == 1
    assert client.get(
        f"/v1/admin/ingest/jobs/{job_id}/failures",
        headers={"X-Admin-API-Key": "test-admin"},
    ).json() == []


def test_admin_ingestion_job_routes_are_available_without_local_admin_key(tmp_path):
    root, settings, factory, service, runner = _runner(tmp_path)
    (root / "note.md").write_text("source", encoding="utf-8")
    settings.admin_api_key = None
    client = TestClient(
        create_app(
            settings=settings,
            ingestion_service=service,
            session_factory=factory,
            ingestion_job_runner=runner,
        )
    )

    response = client.post(
        "/v1/admin/ingest/jobs",
        json={"force": False, "run_async": False},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    runner.shutdown()


def test_admin_ingestion_job_routes_reject_wrong_key_when_configured(tmp_path):
    root, settings, factory, service, runner = _runner(tmp_path)
    (root / "note.md").write_text("source", encoding="utf-8")
    client = TestClient(
        create_app(
            settings=settings,
            ingestion_service=service,
            session_factory=factory,
            ingestion_job_runner=runner,
        )
    )

    response = client.post(
        "/v1/admin/ingest/jobs",
        headers={"X-Admin-API-Key": "wrong"},
        json={"force": False, "run_async": False},
    )

    assert response.status_code == 401
    runner.shutdown()


def test_submission_after_shutdown_does_not_leave_active_job(tmp_path):
    _, _, factory, _, runner = _runner(tmp_path)
    runner.shutdown()

    with pytest.raises(RuntimeError):
        runner.create(force=False, run_async=True)

    with factory() as session:
        job = session.query(IngestionJob).one()
        assert job.status == "failed"
        assert job.active_key is None


def test_drop_derived_tables_includes_ingestion_failures(tmp_path):
    from rag_service.app.db import drop_derived_tables, get_engine, init_db

    settings = Settings(
        knowledge_root=tmp_path / "knowledge",
        database_url="sqlite+pysqlite:///:memory:",
        embedding_provider="hashing",
        embedding_dimensions=384,
    )
    engine = init_db(get_engine(settings))

    assert "ingestion_failures" in drop_derived_tables(engine)


def test_init_db_normalizes_legacy_overlong_ingestion_status(tmp_path):
    settings = Settings(
        knowledge_root=tmp_path / "knowledge",
        database_url="sqlite+pysqlite:///:memory:",
        embedding_provider="hashing",
        embedding_dimensions=384,
    )
    engine = init_db(get_engine(settings))
    factory = get_session_factory(engine=engine)
    with factory() as session:
        session.add(IngestionJob(status="completed_with_errors"))
        session.commit()

    init_db(engine, settings=settings)

    with factory() as session:
        assert session.query(IngestionJob).one().status == "failed"
