from types import SimpleNamespace
import json

from fastapi.testclient import TestClient

from rag_service.app.api import create_app
from rag_service.app.config import Settings
from rag_service.app.schemas import SearchHit


class FakeRetrieval:
    def __init__(self, hits):
        self.hits = hits
        self.latest_debug = {
            "requested_limit": 5,
            "primary_count": len(hits),
            "channels": {"vector": [{"chunk_id": hit.chunk_id, "score": hit.vector_score} for hit in hits]},
            "fusion_order": [hit.chunk_id for hit in hits],
            "query_plan": {"intent": "definition"},
            "reranker": {"status": "disabled", "attempted": False},
            "timings_ms": {"embedding": 1.2},
        }

    def search(self, question, limit=None):
        return self.hits


class FakeIngestion:
    def scan(self, force=False):
        return SimpleNamespace(
            scanned_files=1,
            processed_files=1,
            skipped_files=0,
            failed_files=0,
            deleted_documents=0,
            chunk_count=2,
        )


class FakeGeneration:
    def generate(self, question, context):
        return "答案 [DOC-1]"


def test_empty_question_is_rejected():
    client = TestClient(create_app(retrieval_service=FakeRetrieval([]), ingestion_service=FakeIngestion()))
    response = client.post("/v1/retrieval/search", json={"question": ""})
    assert response.status_code == 422


def test_search_returns_source_metadata():
    hit = SearchHit(
        chunk_id=1,
        content="Use pgvector.",
        source="knowledge/example.md",
        title="Example",
        heading=["Storage"],
        score=0.9,
        matched_channels=["vector"],
    )
    client = TestClient(create_app(retrieval_service=FakeRetrieval([hit]), ingestion_service=FakeIngestion()))
    response = client.post("/v1/retrieval/search", json={"question": "pgvector"})
    assert response.status_code == 200
    body = response.json()
    assert body["hits"][0]["source"] == "knowledge/example.md"
    assert body["citations"][0]["citation_id"] == "DOC-1"


def test_retrieval_debug_includes_primary_count_requested_limit_and_channels():
    hit = SearchHit(
        chunk_id=1,
        content="Use pgvector.",
        source="knowledge/example.md",
        title="Example",
        heading=["Storage"],
        score=0.9,
        matched_channels=["vector"],
    )
    client = TestClient(create_app(retrieval_service=FakeRetrieval([hit]), ingestion_service=FakeIngestion()))
    response = client.post("/v1/retrieval/debug", json={"question": "vector", "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["requested_limit"] == 5
    assert body["primary_count"] == 1
    assert "channels" in body


def test_retrieval_debug_exposes_query_plan_reranker_and_timings():
    hit = SearchHit(
        chunk_id=1,
        content="Use pgvector.",
        source="knowledge/example.md",
        title="Example",
        heading=["Storage"],
        score=0.9,
        matched_channels=["vector"],
    )
    client = TestClient(create_app(retrieval_service=FakeRetrieval([hit]), ingestion_service=FakeIngestion()))
    response = client.post("/v1/retrieval/debug", json={"question": "vector", "limit": 5})
    body = response.json()
    assert body["query_plan"]["intent"] == "definition"
    assert body["reranker"]["status"] == "disabled"
    assert body["timings_ms"]["embedding"] == 1.2


def test_retrieval_debug_redacts_sensitive_query_plan_terms():
    hit = SearchHit(chunk_id=1, content="safe", source="safe.md", title="Safe", heading=[], score=0.9, matched_channels=["vector"])
    retrieval = FakeRetrieval([hit])
    retrieval.latest_debug["query_plan"] = {
        "intent": "definition",
        "rewritten_query": "private-token",
        "expanded_terms": ["private-token"],
        "entities": [{"matched_alias": "private-token"}],
    }
    client = TestClient(create_app(retrieval_service=retrieval, ingestion_service=FakeIngestion()))
    body = client.post("/v1/retrieval/debug", json={"question": "safe"}).json()
    assert "private-token" not in str(body["query_plan"])


def test_low_evidence_does_not_call_generation():
    class FailingGeneration:
        def generate(self, question, context):
            raise AssertionError("must not generate")

    client = TestClient(
        create_app(
            retrieval_service=FakeRetrieval([]),
            ingestion_service=FakeIngestion(),
            generation_service=FailingGeneration(),
        )
    )
    response = client.post("/v1/qa/ask", json={"question": "unknown"})
    assert response.status_code == 200
    assert response.json()["confidence"] == "low"


def test_metadata_only_retrieval_does_not_call_generation():
    class FailingGeneration:
        def generate(self, question, context):
            raise AssertionError("metadata-only hit must not generate")

    hit = SearchHit(
        chunk_id=1,
        content="Redis persistence uses RDB snapshots and AOF logs.",
        source="knowledge/Redis Cluster.pdf",
        title="Redis Cluster failover primary election",
        score=0.99,
        matched_channels=["vector", "keyword"],
    )
    client = TestClient(
        create_app(
            retrieval_service=FakeRetrieval([hit]),
            ingestion_service=FakeIngestion(),
            generation_service=FailingGeneration(),
        )
    )

    response = client.post(
        "/v1/qa/ask",
        json={
            "question": "How does Redis Cluster elect a new primary during failover?",
            "allow_web_fallback": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["confidence"] == "low"
    assert response.json()["citations"] == []


def test_qa_web_search_debug_does_not_expose_provider_error_details():
    provider_error = "provider-secret-key-and-response-body"

    class ErrorWebSearch:
        def search(self, question):
            return SimpleNamespace(
                sources=[],
                status="error",
                reason="web search request failed",
                error=provider_error,
            )

    client = TestClient(
        create_app(
            retrieval_service=FakeRetrieval([]),
            ingestion_service=FakeIngestion(),
            web_search_service=ErrorWebSearch(),
        )
    )
    response = client.post(
        "/v1/qa/ask",
        json={"question": "unknown", "allow_web_fallback": True},
    )

    assert response.status_code == 200
    assert provider_error not in response.text
    assert "error" not in response.json()["retrieval_debug"]["web_search"]


def test_qa_llm_error_does_not_expose_provider_error_details():
    provider_error = "llm-provider-secret-response-body"

    class FailingGeneration:
        def generate(self, question, context):
            raise RuntimeError(provider_error)

    hit = SearchHit(
        chunk_id=1,
        content="Use pgvector.",
        source="knowledge/example.md",
        title="Example",
        heading=["Storage"],
        score=0.9,
        matched_channels=["vector"],
    )
    client = TestClient(
        create_app(
            retrieval_service=FakeRetrieval([hit]),
            ingestion_service=FakeIngestion(),
            generation_service=FailingGeneration(),
        )
    )

    response = client.post("/v1/qa/ask", json={"question": "pgvector"})

    assert response.status_code == 502
    assert provider_error not in response.text
    assert response.json()["detail"] == "LLM generation failed (RuntimeError)"


def test_knowledge_ingest_error_does_not_expose_provider_error_details():
    provider_error = "embedding-provider-secret-response-body"

    class FailingIngestion:
        def scan(self, force=False):
            raise RuntimeError(provider_error)

    client = TestClient(
        create_app(
            retrieval_service=FakeRetrieval([]),
            ingestion_service=FailingIngestion(),
        )
    )

    response = client.post("/v1/knowledge/ingest", json={"force": True})

    assert response.status_code == 500
    assert provider_error not in response.text
    assert response.json()["detail"] == "ingestion failed (RuntimeError)"


def test_admin_eval_generation_error_does_not_expose_provider_error_details(tmp_path):
    provider_error = "eval-llm-secret-response-body"
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {"id": "unknown", "question": "unknown", "relevant_sources": [], "answerable": False}
        )
        + "\n",
        encoding="utf-8",
    )

    class FailingGeneration:
        def generate(self, question, context):
            raise RuntimeError(provider_error)

    client = TestClient(
        create_app(
            settings=Settings(admin_api_key="test-admin", eval_cases_root=tmp_path),
            retrieval_service=FakeRetrieval([]),
            ingestion_service=FakeIngestion(),
            generation_service=FailingGeneration(),
        )
    )

    response = client.post(
        "/v1/admin/eval/run",
        headers={"X-Admin-API-Key": "test-admin"},
        json={"cases_path": cases_path.name, "generate_answers": True},
    )

    assert response.status_code == 502
    assert provider_error not in response.text
    assert response.json()["detail"] == "LLM generation failed (RuntimeError)"
