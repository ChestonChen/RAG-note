from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rag_service.app.api import create_app
from rag_service.app.config import Settings
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.embeddings import HashingEmbeddingProvider
from rag_service.app.models import Chunk, Document
from rag_service.app.retrieval import RetrievalService


def _settings(tmp_path, **kwargs):
    return Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'knowledge-api.sqlite'}",
        knowledge_root=tmp_path / "knowledge",
        **kwargs,
    )


def _app(tmp_path, **settings_kwargs):
    settings = _settings(tmp_path, **settings_kwargs)
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    ingestion = SimpleNamespace(scan=lambda force=False: SimpleNamespace(
        scanned_files=1,
        processed_files=1,
        skipped_files=0,
        failed_files=0,
        deleted_documents=0,
        chunk_count=1,
    ))
    return settings, factory, create_app(settings, session_factory=factory, ingestion_service=ingestion)


def test_document_list_includes_chunk_counts_and_status(tmp_path):
    settings, factory, app = _app(tmp_path)
    with factory() as session:
        document = Document(
            source_path="guide.md",
            source_type="markdown",
            title="Guide",
            file_hash="a" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature="hashing:model:dim=384",
            status="ready",
        )
        session.add(document)
        session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                content="one",
                heading_path=[],
                token_count=1,
                embedding=[0.0] * 384,
            )
        )
        session.commit()

    response = TestClient(app).get("/v1/knowledge/documents")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 1,
            "source_path": "guide.md",
            "title": "Guide",
            "type": "markdown",
            "status": "ready",
            "chunk_count": 1,
            "updated_at": response.json()[0]["updated_at"],
            "error_message": None,
        }
    ]


def test_upload_markdown_writes_inside_root_and_handles_duplicate_name(tmp_path):
    settings, _, app = _app(tmp_path)
    client = TestClient(app)

    first = client.post(
        "/v1/knowledge/upload",
        files={"file": ("guide.md", b"# Guide\n\nhello", "text/markdown")},
    )
    second = client.post(
        "/v1/knowledge/upload",
        files={"file": ("guide.md", b"# Another\n\nworld", "text/markdown")},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    first_path = settings.resolved_knowledge_root() / first.json()["source_path"]
    second_path = settings.resolved_knowledge_root() / second.json()["source_path"]
    assert first_path.read_bytes() == b"# Guide\n\nhello"
    assert second_path.read_bytes() == b"# Another\n\nworld"
    assert first.json()["source_path"] != second.json()["source_path"]
    assert first_path.parent == settings.resolved_knowledge_root()
    assert second_path.parent == settings.resolved_knowledge_root()


def test_upload_rejects_unsupported_and_path_like_filenames(tmp_path):
    _, _, app = _app(tmp_path)
    client = TestClient(app)

    assert client.post(
        "/v1/knowledge/upload",
        files={"file": ("notes.txt", b"not allowed", "text/plain")},
    ).status_code == 400
    assert client.post(
        "/v1/knowledge/upload",
        files={"file": ("../escape.md", b"no", "text/markdown")},
    ).status_code == 400
    assert client.post(
        "/v1/knowledge/upload",
        files={"file": ("sub\\escape.md", b"no", "text/markdown")},
    ).status_code == 400
    for filename in ("foo:bar.md", "CON.md", "name.md ", "name.md."):
        assert client.post(
            "/v1/knowledge/upload",
            files={"file": (filename, b"no", "text/markdown")},
        ).status_code == 400


def test_concurrent_same_name_uploads_reserve_distinct_destinations(tmp_path):
    settings, _, app = _app(tmp_path)

    def upload(content):
        return TestClient(app).post(
            "/v1/knowledge/upload",
            files={"file": ("race.md", content, "text/markdown")},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(upload, (b"one", b"two")))

    assert [response.status_code for response in responses] == [201, 201]
    paths = [settings.resolved_knowledge_root() / response.json()["source_path"] for response in responses]
    assert len({path.name for path in paths}) == 2
    assert {path.read_bytes() for path in paths} == {b"one", b"two"}


def test_upload_temp_file_is_not_an_ingestion_candidate(tmp_path):
    settings, _, app = _app(tmp_path)
    captured_sources = []
    captured_reservations = []
    import rag_service.app.api as api_module

    original_replace = api_module.os.replace

    def capture_replace(source, destination):
        captured_sources.append(source)
        root = settings.resolved_knowledge_root()
        captured_reservations.extend(path for path in root.iterdir() if path.suffix == ".uploadlock")
        assert not [
            path
            for path in root.iterdir()
            if path.suffix.lower() in {".md", ".markdown", ".pdf"}
        ]
        return original_replace(source, destination)

    with patch.object(api_module.os, "replace", side_effect=capture_replace):
        response = TestClient(app).post(
            "/v1/knowledge/upload",
            files={"file": ("scan-safe.md", b"safe", "text/markdown")},
        )

    assert response.status_code == 201
    assert captured_sources
    assert captured_sources[0].suffix == ".uploadtmp"
    assert captured_sources[0].suffix not in {".md", ".markdown", ".pdf"}
    assert captured_reservations
    assert all(path.suffix not in {".md", ".markdown", ".pdf"} for path in captured_reservations)
    assert all(not path.exists() for path in captured_reservations)


def test_upload_rejects_files_over_configured_limit(tmp_path):
    _, _, app = _app(tmp_path, knowledge_upload_max_bytes=4)

    response = TestClient(app).post(
        "/v1/knowledge/upload",
        files={"file": ("large.md", b"12345", "text/markdown")},
    )

    assert response.status_code == 413


def test_ingest_delegates_force_and_requires_admin_when_configured(tmp_path):
    settings = _settings(tmp_path, admin_api_key="secret")
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    calls = []

    class Ingestion:
        def scan(self, force=False):
            calls.append(force)
            return SimpleNamespace(
                scanned_files=2,
                processed_files=1,
                skipped_files=1,
                failed_files=0,
                deleted_documents=0,
                chunk_count=3,
            )

    app = create_app(settings, session_factory=factory, ingestion_service=Ingestion())
    client = TestClient(app)

    assert client.post("/v1/knowledge/ingest", json={"force": True}).status_code == 401
    response = client.post(
        "/v1/knowledge/ingest",
        json={"force": True},
        headers={"X-Admin-API-Key": "secret"},
    )

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 3
    assert calls == [True]


def test_file_serving_stays_inside_knowledge_root(tmp_path):
    settings, factory, app = _app(tmp_path)
    root = settings.resolved_knowledge_root()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "guide.md"
    source.write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    with factory() as session:
        session.add_all(
            [
                Document(
                    source_path="guide.md",
                    source_type="markdown",
                    title="Guide",
                    file_hash="a" * 64,
                    parser_signature="parser-v1",
                    chunking_signature="chunk-v1",
                    embedding_signature="hashing:model:dim=384",
                    status="ready",
                ),
                Document(
                    source_path="../outside.md",
                    source_type="markdown",
                    title="Outside",
                    file_hash="b" * 64,
                    parser_signature="parser-v1",
                    chunking_signature="chunk-v1",
                    embedding_signature="hashing:model:dim=384",
                    status="ready",
                ),
            ]
        )
        session.commit()

    client = TestClient(app)
    safe = client.get("/v1/knowledge/documents/1/file")
    escaped = client.get("/v1/knowledge/documents/2/file")

    assert safe.status_code == 200
    assert safe.text == "safe"
    assert safe.headers["content-disposition"].startswith("inline;")
    assert escaped.status_code == 404


def test_knowledge_routes_use_configured_admin_guard(tmp_path):
    settings, _, app = _app(tmp_path, admin_api_key="secret")
    client = TestClient(app)

    assert client.get("/v1/knowledge/documents").status_code == 401
    assert client.post(
        "/v1/knowledge/upload",
        files={"file": ("guide.md", b"safe", "text/markdown")},
    ).status_code == 401
    assert client.get("/v1/knowledge/documents/1/file").status_code == 401


def test_delete_knowledge_document_removes_source_document_and_chunks(tmp_path):
    settings, factory, app = _app(tmp_path)
    root = settings.resolved_knowledge_root()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "guide.md"
    source.write_text("safe", encoding="utf-8")
    with factory() as session:
        document = Document(
            source_path="guide.md",
            source_type="markdown",
            title="Guide",
            file_hash="a" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature="hashing:model:dim=384",
            status="ready",
        )
        session.add(document)
        session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                content="one",
                heading_path=[],
                token_count=1,
                embedding=[0.0] * 384,
            )
        )
        document_id = document.id
        session.commit()

    client = TestClient(app)
    response = client.delete(f"/v1/knowledge/documents/{document_id}")

    assert response.status_code == 204
    assert not source.exists()
    with factory() as session:
        assert session.get(Document, document_id) is None
        assert session.query(Chunk).filter_by(document_id=document_id).count() == 0
    assert client.get("/v1/knowledge/documents").json() == []


def test_delete_knowledge_document_removes_it_from_retrieval(tmp_path):
    settings, factory, app = _app(tmp_path)
    root = settings.resolved_knowledge_root()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "retrieval-delete.md"
    source.write_text("retirement marker", encoding="utf-8")
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path=source.name,
            source_type="markdown",
            title="Retrieval deletion marker",
            file_hash="0" * 64,
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
                content="This note contains the retrieval deletion marker.",
                heading_path=["Deletion"],
                token_count=8,
                embedding=provider.embed_query("retrieval deletion marker"),
            )
        )
        document_id = document.id
        session.commit()

    retrieval = RetrievalService(factory, settings=settings, embedding_provider=provider)
    assert retrieval.search("retrieval deletion marker")

    response = TestClient(app).delete(f"/v1/knowledge/documents/{document_id}")

    assert response.status_code == 204
    assert retrieval.search("retrieval deletion marker") == []


def test_delete_knowledge_document_cleans_stale_record_when_source_is_missing(tmp_path):
    _, factory, app = _app(tmp_path)
    with factory() as session:
        document = Document(
            source_path="missing.md",
            source_type="markdown",
            title="Missing",
            file_hash="b" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature="hashing:model:dim=384",
            status="ready",
        )
        session.add(document)
        session.commit()
        document_id = document.id

    response = TestClient(app).delete(f"/v1/knowledge/documents/{document_id}")

    assert response.status_code == 204
    with factory() as session:
        assert session.get(Document, document_id) is None


def test_delete_unknown_document_returns_404(tmp_path):
    _, _, app = _app(tmp_path)

    assert TestClient(app).delete("/v1/knowledge/documents/999").status_code == 404


def test_delete_rejects_source_path_outside_knowledge_root(tmp_path):
    settings, factory, app = _app(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("must remain", encoding="utf-8")
    with factory() as session:
        document = Document(
            source_path="../outside.md",
            source_type="markdown",
            title="Outside",
            file_hash="c" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature="hashing:model:dim=384",
            status="ready",
        )
        session.add(document)
        session.commit()
        document_id = document.id

    response = TestClient(app).delete(f"/v1/knowledge/documents/{document_id}")

    assert response.status_code == 404
    assert outside.read_text(encoding="utf-8") == "must remain"
    with factory() as session:
        assert session.get(Document, document_id) is not None


def test_delete_rejects_symlink_source_path_outside_knowledge_root(tmp_path):
    settings, factory, app = _app(tmp_path)
    root = settings.resolved_knowledge_root()
    root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.md"
    outside.write_text("must remain", encoding="utf-8")
    linked = root / "linked.md"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with factory() as session:
        document = Document(
            source_path="linked.md",
            source_type="markdown",
            title="Linked",
            file_hash="d" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature="hashing:model:dim=384",
            status="ready",
        )
        session.add(document)
        session.commit()
        document_id = document.id

    response = TestClient(app).delete(f"/v1/knowledge/documents/{document_id}")

    assert response.status_code == 404
    assert outside.read_text(encoding="utf-8") == "must remain"
    assert linked.is_symlink()
    with factory() as session:
        assert session.get(Document, document_id) is not None


def test_delete_file_unlink_error_returns_generic_500_and_preserves_rows(tmp_path):
    settings, factory, app = _app(tmp_path)
    root = settings.resolved_knowledge_root()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "guide.md"
    source.write_text("safe", encoding="utf-8")
    with factory() as session:
        document = Document(
            source_path="guide.md",
            source_type="markdown",
            title="Guide",
            file_hash="e" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature="hashing:model:dim=384",
            status="ready",
        )
        session.add(document)
        session.flush()
        session.add(Chunk(document_id=document.id, chunk_index=0, content="one", heading_path=[], token_count=1, embedding=[0.0] * 384))
        document_id = document.id
        session.commit()

    with patch("rag_service.app.api.Path.unlink", side_effect=OSError("permission denied")):
        response = TestClient(app).delete(f"/v1/knowledge/documents/{document_id}")

    assert response.status_code == 500
    assert "permission denied" not in response.text
    with factory() as session:
        assert session.get(Document, document_id) is not None
        assert session.query(Chunk).filter_by(document_id=document_id).count() == 1


def test_delete_database_failure_returns_generic_500_and_preserves_rows(tmp_path):
    settings, factory, _ = _app(tmp_path)
    root = settings.resolved_knowledge_root()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "guide.md"
    source.write_text("safe", encoding="utf-8")
    with factory() as session:
        document = Document(
            source_path="guide.md",
            source_type="markdown",
            title="Guide",
            file_hash="f" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature="hashing:model:dim=384",
            status="ready",
        )
        session.add(document)
        session.flush()
        session.add(Chunk(document_id=document.id, chunk_index=0, content="one", heading_path=[], token_count=1, embedding=[0.0] * 384))
        document_id = document.id
        session.commit()

    class ExecuteFailingSession:
        def __init__(self, session):
            self._session = session

        def __enter__(self):
            self._session.__enter__()
            return self

        def __exit__(self, *args):
            return self._session.__exit__(*args)

        def get(self, *args, **kwargs):
            return self._session.get(*args, **kwargs)

        def execute(self, *args, **kwargs):
            raise OSError("database unavailable")

        def delete(self, *args, **kwargs):
            return self._session.delete(*args, **kwargs)

        def rollback(self):
            return self._session.rollback()

        def commit(self):
            return self._session.commit()

    def failing_factory():
        return ExecuteFailingSession(factory())

    app = create_app(settings, session_factory=failing_factory, ingestion_service=SimpleNamespace(scan=lambda force=False: SimpleNamespace()))
    response = TestClient(app).delete(f"/v1/knowledge/documents/{document_id}")

    assert response.status_code == 500
    assert "database unavailable" not in response.text
    assert source.read_text(encoding="utf-8") == "safe"
    with factory() as session:
        assert session.get(Document, document_id) is not None
        assert session.query(Chunk).filter_by(document_id=document_id).count() == 1


def test_delete_document_lookup_failure_returns_generic_500(tmp_path):
    settings, factory, _ = _app(tmp_path)

    class LookupFailingSession:
        def __init__(self, session):
            self._session = session

        def __enter__(self):
            self._session.__enter__()
            return self

        def __exit__(self, *args):
            return self._session.__exit__(*args)

        def get(self, *args, **kwargs):
            raise OSError("database unavailable")

        def rollback(self):
            return self._session.rollback()

    def failing_factory():
        return LookupFailingSession(factory())

    app = create_app(
        settings,
        session_factory=failing_factory,
        ingestion_service=SimpleNamespace(scan=lambda force=False: SimpleNamespace()),
    )
    response = TestClient(app).delete("/v1/knowledge/documents/1")

    assert response.status_code == 500
    assert "database unavailable" not in response.text


def test_delete_with_configured_admin_key_accepts_valid_header(tmp_path):
    settings, factory, app = _app(tmp_path, admin_api_key="secret")
    root = settings.resolved_knowledge_root()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "guide.md"
    source.write_text("safe", encoding="utf-8")
    with factory() as session:
        document = Document(
            source_path="guide.md",
            source_type="markdown",
            title="Guide",
            file_hash="1" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature="hashing:model:dim=384",
            status="ready",
        )
        session.add(document)
        session.commit()
        document_id = document.id

    response = TestClient(app).delete(
        f"/v1/knowledge/documents/{document_id}",
        headers={"X-Admin-API-Key": "secret"},
    )

    assert response.status_code == 204


def test_delete_knowledge_document_uses_configured_admin_guard(tmp_path):
    settings = _settings(tmp_path, admin_api_key="secret")
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    app = create_app(settings, session_factory=factory)

    assert TestClient(app).delete("/v1/knowledge/documents/1").status_code == 401
