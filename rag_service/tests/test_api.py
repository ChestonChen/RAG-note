from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag_service.app.api import create_app
from rag_service.app.schemas import SearchHit


class FakeRetrieval:
    def __init__(self, hits):
        self.hits = hits
        self.latest_debug = {
            "requested_limit": 5,
            "primary_count": len(hits),
            "channels": {"vector": [{"chunk_id": hit.chunk_id, "score": hit.vector_score} for hit in hits]},
            "fusion_order": [hit.chunk_id for hit in hits],
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
    response = client.post("/v1/retrieval/search", json={"question": "vector"})
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
