from rag_service.app.config import Settings
from rag_service.app.context import ContextBuilder
from rag_service.app.generation import ChatGenerationService
from rag_service.app.schemas import SearchHit


def _context():
    hit = SearchHit(
        chunk_id=1,
        content="pgvector 将 embedding 向量存储在 PostgreSQL 中。",
        source="knowledge/pgvector.md",
        title="pgvector 笔记",
        heading=["向量数据库"],
        score=0.8,
        matched_channels=["vector", "keyword"],
    )
    return ContextBuilder().build("pgvector 是什么", [hit])


def test_generation_without_key_returns_retrieval_summary():
    service = ChatGenerationService(Settings(llm_api_key=None))

    answer = service.generate("pgvector 是什么", _context())

    assert "未配置" in answer
    assert "pgvector" in answer


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "模型回答 [DOC-1]"}}]}


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def test_generation_uses_openai_compatible_chat_client():
    client = _FakeClient()
    service = ChatGenerationService(
        Settings(llm_api_key="test-key", llm_model="deepseek-chat"),
        client=client,
    )

    answer = service.generate("pgvector 是什么", _context())

    assert answer == "模型回答 [DOC-1]"
    request = client.chat.completions.calls[0]
    assert request["model"] == "deepseek-chat"
    assert request["messages"][1]["content"].find("pgvector") >= 0
