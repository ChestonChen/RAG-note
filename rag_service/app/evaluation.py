"""Optional LLM-as-a-judge evaluation for generated RAG answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import Settings, get_settings


JUDGE_SYSTEM_PROMPT = """你是一个严格的 RAG 答案评测器。
请判断 ANSWER 是否能够被 CONTEXT 中的证据支持，并且是否回答了 QUESTION。
只能输出一个 JSON 对象，不要输出 Markdown、代码围栏或额外解释：
{"passed": true 或 false, "score": 0 到 1 之间的数字, "reason": "简短原因"}
score 表示答案被上下文支持且回答问题的程度。上下文没有足够证据时应判定为 false。"""


@dataclass(frozen=True)
class JudgeResult:
    """A normalized result from the optional AI judge."""

    status: Literal["ok", "not_configured", "error"]
    passed: bool | None = None
    score: float | None = None
    reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a dictionary response."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _resolved_setting(settings: Settings, name: str, fallback_name: str | None = None) -> Any:
    """Use Settings' explicit resolver when available, with local fallback compatibility."""

    resolver = getattr(settings, f"resolved_{name}", None)
    if callable(resolver):
        value = resolver()
        if value is not None:
            return value
    value = getattr(settings, name, None)
    if value is not None:
        return value
    return getattr(settings, fallback_name, None) if fallback_name else None


def build_judge_prompt(question: str, answer: str, context: str) -> str:
    """Build the user message sent to the judge model."""

    return f"""QUESTION:\n{question.strip()}\n\nANSWER:\n{answer.strip()}\n\nCONTEXT:\n{context.strip() or '(没有检索到证据)'}"""


def _extract_content(response: Any) -> str:
    choices = _field(response, "choices", []) or []
    if not choices:
        raise ValueError("judge response did not contain choices")
    message = _field(choices[0], "message")
    if message is None:
        raise ValueError("judge response did not contain a message")
    content = _field(message, "content")
    if isinstance(content, list):
        # Some compatible gateways return content blocks instead of a string.
        content = "".join(str(_field(block, "text", "")) for block in content)
    if not content:
        raise ValueError("judge response contained an empty message")
    return str(content).strip()


def _parse_json(content: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else content.strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("judge response JSON must be an object")
    passed = parsed.get("passed")
    score = parsed.get("score")
    if not isinstance(passed, bool):
        raise ValueError("judge response field 'passed' must be boolean")
    if isinstance(score, bool):
        raise ValueError("judge response field 'score' must be a number")
    try:
        score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("judge response field 'score' must be a number") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError("judge response field 'score' must be between 0 and 1")
    reason = parsed.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)
    return {"passed": passed, "score": score, "reason": reason}


class EvaluationJudge:
    """Evaluate answer groundedness with an optional OpenAI-compatible model."""

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            key = _resolved_setting(self.settings, "eval_judge_api_key", "llm_api_key")
            if not key:
                raise ValueError("EVAL_JUDGE_API_KEY is not configured")
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("install the openai package to use AI evaluation") from exc
            base_url = _resolved_setting(self.settings, "eval_judge_base_url", "llm_base_url")
            timeout = getattr(self.settings, "eval_judge_timeout_seconds", None)
            kwargs: dict[str, Any] = {"api_key": key}
            if base_url:
                kwargs["base_url"] = str(base_url).rstrip("/")
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._client = OpenAI(**kwargs)
        return self._client

    def evaluate(self, question: str, answer: str, context: str) -> JudgeResult:
        key = _resolved_setting(self.settings, "eval_judge_api_key", "llm_api_key")
        model = _resolved_setting(self.settings, "eval_judge_model", "llm_model")
        if not key or not model:
            return JudgeResult(status="not_configured", reason="AI judge is not configured")

        try:
            response = self.client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": build_judge_prompt(question, answer, context)},
                ],
            )
            content = _extract_content(response)
            parsed = _parse_json(content)
            return JudgeResult(
                status="ok",
                passed=parsed["passed"],
                score=parsed["score"],
                reason=parsed["reason"],
                raw=parsed,
            )
        except json.JSONDecodeError as exc:
            return JudgeResult(
                status="error",
                reason=f"invalid judge JSON: {exc}",
                raw={"error": str(exc)},
            )
        except Exception as exc:  # Keep an optional evaluator from breaking QA.
            return JudgeResult(status="error", reason=str(exc), raw={"error": str(exc)})


__all__ = ["EvaluationJudge", "JudgeResult", "JUDGE_SYSTEM_PROMPT", "build_judge_prompt"]
