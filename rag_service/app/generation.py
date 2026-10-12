"""OpenAI-compatible answer generation with evidence-only prompting."""

from __future__ import annotations

from typing import Any

from .config import Settings, get_settings
from .context import ContextPackage


SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。
只能依据 CONTEXT 中的内容回答，不要补充上下文之外的事实。
每个关键结论都要在句末使用 [DOC-n] 引用；如果上下文不足，明确说无法从知识库确认。
回答使用中文，先给出直接结论，再给出必要的解释。不要伪造引用编号。"""


def build_prompt(question: str, context: ContextPackage) -> str:
    web_context = "\n".join(
        f"[WEB-{index}] {source.title} | {source.url}\n{source.snippet or ''}".strip()
        for index, source in enumerate(context.web_sources, start=1)
    ) or "(没有联网补充资料)"
    return f"""QUESTION:
{question.strip()}

CONTEXT:
{context.text or '(没有检索到证据)'}

WEB SOURCES:
{web_context}

请严格遵守系统规则回答。"""


class ChatGenerationService:
    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.settings.llm_api_key:
                raise ValueError("LLM_API_KEY is not configured")
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("install the openai package to use remote generation") from exc
            self._client = OpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url.rstrip("/"),
                timeout=self.settings.llm_timeout_seconds,
            )
        return self._client

    def generate(self, question: str, context: ContextPackage) -> str:
        if not self.settings.llm_api_key:
            return self._fallback_answer(question, context)
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, context)},
            ],
        )
        def field(value: Any, name: str, default: Any = None) -> Any:
            if isinstance(value, dict):
                return value.get(name, default)
            return getattr(value, name, default)

        choices = field(response, "choices", []) or []
        if not choices or not field(choices[0], "message"):
            raise RuntimeError("LLM response did not contain a message")
        content = field(field(choices[0], "message"), "content")
        if not content:
            raise RuntimeError("LLM response contained an empty message")
        return str(content).strip()

    @staticmethod
    def _fallback_answer(question: str, context: ContextPackage) -> str:
        if not context.hits and not context.web_sources:
            return "当前未配置生成模型，且知识库与联网搜索都没有检索到相关资料。"
        lines = [
            "当前未配置生成模型，下面返回检索到的知识库摘要。",
            f"问题：{question}",
            "",
            "检索到的资料：",
        ]
        for index, hit in enumerate(context.hits[:3], start=1):
            location = hit.source
            if hit.heading:
                location += " > " + " > ".join(hit.heading)
            lines.append(f"{index}. [{location}] {hit.content[:500].replace(chr(10), ' ')}")
        if context.web_sources:
            lines.append("\n联网补充资料：")
            for index, source in enumerate(context.web_sources[:3], start=1):
                lines.append(f"{index}. {source.title} - {source.url}")
        return "\n".join(lines)


__all__ = ["ChatGenerationService", "SYSTEM_PROMPT", "build_prompt"]
