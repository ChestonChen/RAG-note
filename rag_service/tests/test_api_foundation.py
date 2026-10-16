from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag_service.app.api import create_app
from rag_service.app.config import Settings
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.evaluation import JudgeResult
from rag_service.app.models import QALog
from rag_service.app.schemas import SearchHit, WebSource
from rag_service.app.web_search import WebSearchResult


class _FakeRetrieval:
    def __init__(self, hits):
        self.hits = hits
        self.latest_debug = {"requested_limit": 5, "primary_count": len(hits), "channels": {}}

    def search(self, question, limit=None):
        return self.hits


class _FakeIngestion:
    def scan(self, force=False):
        return SimpleNamespace(
            scanned_files=1,
            processed_files=1,
            skipped_files=0,
            failed_files=0,
            deleted_documents=0,
            chunk_count=1,
        )


class _FakeGeneration:
    def __init__(self):
        self.calls = []

    def generate(self, question, context):
        self.calls.append((question, context))
        return "基于证据的回答 [DOC-1]"


class _FakeWebSearch:
    def search(self, question):
        return WebSearchResult(
            sources=[WebSource(title="External", url="https://example.test/source")],
            status="ok",
        )


class _FakeJudge:
    def evaluate(self, question, answer, context):
        return JudgeResult(status="ok", passed=True, score=1.0, reason="supported")


def _hit(score=0.8):
    return SearchHit(
        chunk_id=1,
        content="pgvector stores vectors in PostgreSQL.",
        source="knowledge/pgvector.md",
        title="pgvector",
        heading=["Storage"],
        score=score,
        matched_channels=["vector"],
    )


def _sqlite_settings(tmp_path, **kwargs):
    return Settings(database_url=f"sqlite+pysqlite:///{tmp_path / 'api.sqlite'}", eval_cases_root=tmp_path, **kwargs)


def test_low_confidence_can_use_web_fallback_and_expose_sources(tmp_path):
    settings = _sqlite_settings(tmp_path, web_search_enabled=True)
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    generation = _FakeGeneration()
    app = create_app(
        settings,
        retrieval_service=_FakeRetrieval([_hit(score=0.1)]),
        ingestion_service=_FakeIngestion(),
        generation_service=generation,
        web_search_service=_FakeWebSearch(),
        session_factory=factory,
    )

    response = TestClient(app).post(
        "/v1/qa/ask",
        json={"question": "pgvector 是什么", "allow_web_fallback": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "kb_plus_web"
    assert body["used_web_search"] is True
    assert body["web_sources"][0]["url"] == "https://example.test/source"
    assert generation.calls[0][1].web_sources


def test_retrieval_debug_persists_evidence_report(tmp_path):
    settings = _sqlite_settings(tmp_path)
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    app = create_app(
        settings,
        retrieval_service=_FakeRetrieval([_hit()]),
        ingestion_service=_FakeIngestion(),
        generation_service=_FakeGeneration(),
        session_factory=factory,
    )

    response = TestClient(app).post(
        "/v1/retrieval/debug",
        json={"question": "pgvector 是什么"},
    )

    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert evidence["direct_hit_count"] == 1
    assert evidence["eligible_citation_count"] == 1
    assert 0 <= evidence["score"] <= 1


def test_weak_only_evidence_abstains_without_calling_answer_model(tmp_path):
    class MustNotGenerate:
        def generate(self, question, context):
            raise AssertionError("weak-only evidence must not reach the answer model")

    settings = _sqlite_settings(tmp_path)
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    weak_hit = _hit(score=0.95).model_copy(update={"topic_match": "body_only"})
    app = create_app(
        settings,
        retrieval_service=_FakeRetrieval([weak_hit]),
        ingestion_service=_FakeIngestion(),
        generation_service=MustNotGenerate(),
        session_factory=factory,
    )

    response = TestClient(app).post(
        "/v1/qa/ask",
        json={"question": "pgvector 是什么", "allow_web_fallback": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == "low"
    assert body["citations"] == []
    assert body["retrieval_debug"]["evidence"]["caps"]


def test_qa_answer_is_persisted_to_log(tmp_path):
    settings = _sqlite_settings(tmp_path)
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    app = create_app(
        settings,
        retrieval_service=_FakeRetrieval([_hit()]),
        ingestion_service=_FakeIngestion(),
        generation_service=_FakeGeneration(),
        session_factory=factory,
    )

    response = TestClient(app).post("/v1/qa/ask", json={"question": "pgvector 是什么"})

    assert response.status_code == 200
    with factory() as session:
        log = session.query(QALog).one()
        assert log.question == "pgvector 是什么"
        assert log.answer == "基于证据的回答 [DOC-1]"
        assert log.citations
        assert log.retrieval_debug["evidence"]["direct_hit_count"] == 1


def test_admin_ingest_requires_configured_api_key(tmp_path):
    settings = _sqlite_settings(tmp_path, admin_api_key="secret")
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    app = create_app(
        settings,
        retrieval_service=_FakeRetrieval([]),
        ingestion_service=_FakeIngestion(),
        session_factory=factory,
    )
    client = TestClient(app)

    assert client.post("/v1/admin/ingest", json={"force": False}).status_code == 401
    assert client.post("/v1/ingest", json={"force": False}).status_code == 401
    assert (
        client.post(
            "/v1/ingest",
            json={"force": False},
            headers={"X-Admin-API-Key": "secret"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/admin/ingest",
            json={"force": False},
            headers={"X-Admin-API-Key": "secret"},
        ).status_code
        == 200
    )


def test_admin_qa_logs_returns_persisted_answer(tmp_path):
    settings = _sqlite_settings(tmp_path, admin_api_key="secret")
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    app = create_app(
        settings,
        retrieval_service=_FakeRetrieval([_hit()]),
        ingestion_service=_FakeIngestion(),
        generation_service=_FakeGeneration(),
        session_factory=factory,
    )
    client = TestClient(app)
    headers = {"X-Admin-API-Key": "secret"}

    assert client.post("/v1/qa/ask", json={"question": "pgvector 是什么"}).status_code == 200
    response = client.get("/v1/admin/qa-logs", headers=headers)

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["items"][0]["answer"] == "基于证据的回答 [DOC-1]"


def test_admin_eval_judge_is_protected_and_returns_optional_status(tmp_path):
    # Keep this contract test offline and independent from a developer's
    # project .env. The production resolver intentionally falls back to the
    # normal LLM key, so an explicit empty judge configuration is required.
    settings = _sqlite_settings(
        tmp_path,
        admin_api_key="secret",
        eval_judge_api_key=None,
        eval_judge_model=None,
        llm_api_key=None,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    app = create_app(settings, session_factory=factory)
    client = TestClient(app)

    assert client.post(
        "/v1/admin/eval/judge",
        json={"question": "q", "answer": "a", "context": "c"},
    ).status_code == 401
    response = client.post(
        "/v1/admin/eval/judge",
        json={"question": "q", "answer": "a", "context": "c"},
        headers={"X-Admin-API-Key": "secret"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "not_configured"


def test_admin_eval_run_generates_answer_when_ai_judge_is_requested(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"id":"one","question":"q","relevant_sources":["knowledge/pgvector.md"],"answerable":true,"evidence_groups":[["pgvector"]]}\n',
        encoding="utf-8",
    )
    settings = _sqlite_settings(tmp_path, admin_api_key="secret")
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    app = create_app(
        settings,
        retrieval_service=_FakeRetrieval([_hit()]),
        ingestion_service=_FakeIngestion(),
        generation_service=_FakeGeneration(),
        evaluation_judge=_FakeJudge(),
        session_factory=factory,
    )
    response = TestClient(app).post(
        "/v1/admin/eval/run",
        headers={"X-Admin-API-Key": "secret"},
        json={"cases_path": cases.name, "judge_with_ai": True},
    )

    assert response.status_code == 200
    assert response.json()["answers"][0]["answer"] == "基于证据的回答 [DOC-1]"
    assert response.json()["answers"][0]["judge"]["passed"] is True
