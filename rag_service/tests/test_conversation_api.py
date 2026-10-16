import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import select

from rag_service.app.api import create_app
from rag_service.app.config import Settings
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.models import Conversation, Message
from rag_service.app.schemas import SearchHit


class FakeRetrieval:
    def __init__(self, hits):
        self.hits = hits
        self.latest_debug = {"requested_limit": 8, "primary_count": len(hits), "channels": {}}

    def search(self, question, limit=None):
        return self.hits


class FakeWebSearch:
    def __init__(self, sources=None):
        self.sources = sources or []

    def search(self, question):
        return SimpleNamespace(sources=self.sources, status="ok", reason=None, error=None)


class FakeGeneration:
    def __init__(self, fragments=None, error=None):
        self.fragments = fragments or ["模型", "回答"]
        self.error = error

    def stream(self, question, context):
        if self.error:
            raise self.error
        yield from self.fragments


def _client(hits=None, generation=None, web=None):
    engine = init_db(get_engine(database_url="sqlite+pysqlite:///:memory:"))
    session_factory = get_session_factory(engine=engine)
    settings = Settings(database_url="sqlite+pysqlite:///:memory:", llm_api_key="test-key")
    app = create_app(
        settings,
        retrieval_service=FakeRetrieval(hits or []),
        generation_service=generation or FakeGeneration(),
        web_search_service=web or FakeWebSearch(),
        session_factory=session_factory,
    )
    return TestClient(app), session_factory


def _events(response):
    events = []
    current = None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            current = line[7:]
        elif line.startswith("data: "):
            events.append((current, json.loads(line[6:])))
    return events


def test_conversation_crud_and_stream_persists_messages_and_metadata():
    hit = SearchHit(
        chunk_id=1,
        content="AirCode 是 OpenAI 兼容服务。",
        source="knowledge/aircode.md",
        title="AirCode",
        heading=["接口"],
        score=0.8,
        matched_channels=["vector"],
    )
    client, sessions = _client([hit])

    created = client.post("/v1/conversations", json={"title": "测试会话"})
    assert created.status_code == 200
    conversation_id = created.json()["id"]
    assert client.get("/v1/conversations").json()[0]["title"] == "测试会话"

    response = client.post(
        f"/v1/conversations/{conversation_id}/stream",
        json={"question": "AirCode 是什么", "allow_web_fallback": False},
    )
    assert response.status_code == 200
    events = _events(response)
    assert [name for name, _ in events] == ["message_start", "token", "token", "metadata", "complete"]
    assert events[0][1]["user_message_id"]
    assert events[1][1]["text"] == "模型"
    assert events[-2][1]["citations"][0]["citation_id"] == "DOC-1"
    assert events[-2][1]["retrieval_debug"]["evidence"]["direct_hit_count"] == 1
    assert events[-1][1]["answer"] == "模型回答"

    detail = client.get(f"/v1/conversations/{conversation_id}").json()
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["content"] == "模型回答"

    renamed = client.patch(f"/v1/conversations/{conversation_id}", json={"title": "已重命名"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "已重命名"
    assert client.delete(f"/v1/conversations/{conversation_id}").status_code == 204
    with sessions() as session:
        assert session.scalars(select(Conversation)).all() == []
        assert session.scalars(select(Message)).all() == []


def test_low_evidence_stream_returns_recoverable_answer_without_generation():
    class FailingGeneration:
        def stream(self, question, context):
            raise AssertionError("must not generate")

    client, _ = _client([], FailingGeneration())
    conversation_id = client.post("/v1/conversations", json={}).json()["id"]
    response = client.post(
        f"/v1/conversations/{conversation_id}/stream",
        json={"question": "未知问题", "allow_web_fallback": False},
    )
    events = _events(response)
    assert [name for name, _ in events] == ["message_start", "token", "metadata", "complete"]
    assert events[-2][1]["confidence"] == "low"


def test_weak_only_evidence_stream_abstains_without_generation():
    class FailingGeneration:
        def stream(self, question, context):
            raise AssertionError("weak-only evidence must not reach the answer model")

    weak = SearchHit(
        chunk_id=7,
        content="A generic body mention.",
        source="weak.md",
        title="Weak",
        score=0.95,
        matched_channels=["vector"],
        topic_match="body_only",
    )
    client, _ = _client([weak], FailingGeneration())
    conversation_id = client.post("/v1/conversations", json={}).json()["id"]

    response = client.post(
        f"/v1/conversations/{conversation_id}/stream",
        json={"question": "specific question", "allow_web_fallback": False},
    )

    events = _events(response)
    metadata = next(payload for name, payload in events if name == "metadata")
    assert metadata["confidence"] == "low"
    assert metadata["citations"] == []
    assert metadata["retrieval_debug"]["evidence"]["caps"]


def test_metadata_only_stream_abstains_without_generation():
    class FailingGeneration:
        def stream(self, question, context):
            raise AssertionError("metadata-only hit must not reach the answer model")

    hit = SearchHit(
        chunk_id=9,
        content="Redis persistence uses RDB snapshots and AOF logs.",
        source="knowledge/Redis Cluster.pdf",
        title="Redis Cluster failover primary election",
        score=0.99,
        matched_channels=["vector", "keyword"],
    )
    client, _ = _client([hit], FailingGeneration())
    conversation_id = client.post("/v1/conversations", json={}).json()["id"]

    response = client.post(
        f"/v1/conversations/{conversation_id}/stream",
        json={
            "question": "How does Redis Cluster elect a new primary during failover?",
            "allow_web_fallback": False,
        },
    )

    events = _events(response)
    metadata = next(payload for name, payload in events if name == "metadata")
    assert metadata["confidence"] == "low"
    assert metadata["citations"] == []


def test_generation_error_marks_user_message_and_emits_safe_error(caplog):
    client, sessions = _client(
        [
            SearchHit(
                chunk_id=1,
        content="证据",
                source="source.md",
                title="source",
                score=0.8,
                matched_channels=["vector"],
            )
        ],
        FakeGeneration(error=RuntimeError("provider exploded")),
    )
    conversation_id = client.post("/v1/conversations", json={}).json()["id"]
    response = client.post(
        f"/v1/conversations/{conversation_id}/stream",
        json={"question": "证据", "allow_web_fallback": False},
    )
    events = _events(response)
    assert response.status_code == 200
    assert events[-1][0] == "error"
    assert "provider exploded" not in json.dumps(events[-1][1])
    assert "provider exploded" not in caplog.text
    with sessions() as session:
        messages = session.scalars(select(Message)).all()
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].status == "error"
        assert messages[0].error_message


def test_web_search_provider_error_is_not_exposed_in_stream_metadata_or_persistence():
    provider_error = "provider-secret-error-with-sensitive-details"

    class ErrorWebSearch:
        def search(self, question):
            return SimpleNamespace(
                sources=[],
                status="error",
                reason="web search request failed",
                error=provider_error,
            )

    client, sessions = _client([], web=ErrorWebSearch())
    conversation_id = client.post("/v1/conversations", json={}).json()["id"]
    response = client.post(
        f"/v1/conversations/{conversation_id}/stream",
        json={"question": "联网查询", "allow_web_fallback": True},
    )

    events = _events(response)
    metadata = next(payload for name, payload in events if name == "metadata")
    assert provider_error not in json.dumps(metadata, ensure_ascii=False)
    with sessions() as session:
        assistant = session.scalar(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.role == "assistant",
            )
        )
        assert assistant is not None
        assert provider_error not in json.dumps(assistant.retrieval_debug, ensure_ascii=False)
