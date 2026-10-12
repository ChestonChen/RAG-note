from rag_service.app.config import Settings
from rag_service.app.evaluation import EvaluationJudge


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"passed": true, "score": 0.91, "reason": "evidence supports answer"}'
                    }
                }
            ]
        }


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def test_ai_judge_parses_groundedness_result():
    client = _FakeClient()
    settings = Settings(
        eval_judge_api_key="judge-key",
        eval_judge_model="judge-model",
    )
    judge = EvaluationJudge(settings, client=client)

    result = judge.evaluate(
        question="pgvector 是什么",
        answer="它是 PostgreSQL 的向量扩展。",
        context="[DOC-1] pgvector 是 PostgreSQL 的向量扩展。",
    )

    assert result.status == "ok"
    assert result.passed is True
    assert result.score == 0.91
    assert client.chat.completions.calls[0]["model"] == "judge-model"


def test_ai_judge_is_optional_when_no_key_is_configured():
    result = EvaluationJudge(Settings(eval_judge_api_key=None, llm_api_key=None)).evaluate(
        question="unknown",
        answer="fallback",
        context="no evidence",
    )

    assert result.status == "not_configured"
    assert result.passed is None


class _MalformedCompletions:
    def create(self, **kwargs):
        return {"choices": [{"message": {"content": "not json"}}]}


class _MalformedClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _MalformedCompletions()})()


def test_ai_judge_returns_error_for_malformed_json_without_raising():
    result = EvaluationJudge(
        Settings(eval_judge_api_key="judge-key", eval_judge_model="judge-model"),
        client=_MalformedClient(),
    ).evaluate("question", "answer", "context")

    assert result.status == "error"
    assert result.passed is None
    assert "JSON" in (result.reason or "") or "json" in (result.reason or "")


def test_ai_judge_can_fall_back_to_llm_configuration():
    client = _FakeClient()
    result = EvaluationJudge(
        Settings(llm_api_key="llm-key", llm_model="llm-judge"),
        client=client,
    ).evaluate("question", "answer", "context")

    assert result.status == "ok"
    assert client.chat.completions.calls[0]["model"] == "llm-judge"
