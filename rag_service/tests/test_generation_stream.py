from rag_service.app.config import Settings
from rag_service.app.context import ContextBuilder
from rag_service.app.generation import ChatGenerationService
from rag_service.app.schemas import SearchHit


def _context():
    hit = SearchHit(
        chunk_id=1,
        content="AirCode 提供 OpenAI 兼容的流式接口。",
        source="knowledge/aircode.md",
        title="AirCode",
        heading=["接口"],
        score=0.8,
        matched_channels=["vector"],
    )
    return ContextBuilder().build("AirCode 是什么", [hit])


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(
            [
                {"choices": [{"delta": {"content": "第一段"}}]},
                {"choices": [{"delta": {"content": "第二段"}}]},
                {"choices": [{"delta": {}}]},
            ]
        )


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def test_generation_stream_uses_openai_stream_flag_and_yields_content_fragments():
    client = _FakeClient()
    service = ChatGenerationService(Settings(llm_api_key="test-key"), client=client)

    fragments = list(service.stream("AirCode 是什么", _context()))

    assert fragments == ["第一段", "第二段"]
    request = client.chat.completions.calls[0]
    assert request["stream"] is True
    assert request["model"] == service.settings.llm_model


def test_generation_stream_without_key_yields_fallback_answer():
    service = ChatGenerationService(Settings(llm_api_key=None))

    fragments = list(service.stream("AirCode 是什么", _context()))

    assert fragments
    assert "未配置" in "".join(fragments)
