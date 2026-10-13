"""Deterministic query normalization and lightweight expansion.

The planner intentionally does not call an LLM.  It turns common technical
aliases and question intents into extra retrieval terms while keeping the
original question available for phrase matching and display.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .term_dictionary import TermDictionary


_INTENT_MARKERS: dict[str, tuple[str, ...]] = {
    "definition": ("什么是", "是什么", "定义", "概念", "介绍", "作用"),
    "implementation": ("原理", "底层", "实现", "机制", "架构", "怎么做", "如何做"),
    "comparison": ("区别", "对比", "比较", "不同", "优缺点"),
    "usage": ("怎么用", "如何使用", "用法", "命令", "示例", "实践"),
    "troubleshooting": ("报错", "失败", "异常", "阻塞", "排查", "怎么办"),
}

# Kept intentionally compact.  This is a built-in baseline; a later version
# can load a user-maintained JSON dictionary without changing the planner API.
_TERM_ENTRIES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("pgvector", ("pgvector", "向量数据库", "向量检索", "vector database"), ("embedding", "余弦相似度", "PostgreSQL")),
    ("rag", ("rag", "RAG", "检索增强生成", "检索增强"), ("retrieval", "embedding", "上下文", "引用")),
    ("embedding", ("embedding", "嵌入", "向量化", "向量表示"), ("向量", "语义相似度")),
    ("stl", ("STL", "标准模板库", "模板库"), ("C++", "容器", "迭代器", "算法")),
    ("vector", ("vector", "std::vector", "动态数组", "顺序表"), ("C++", "扩容", "容量", "迭代器")),
    ("tcp", ("TCP", "传输控制协议", "三次握手", "四次挥手"), ("可靠传输", "连接")),
    ("redis", ("Redis", "缓存", "键值数据库"), ("内存数据库", "持久化", "高可用")),
    ("mysql", ("MySQL", "SQL", "关系型数据库"), ("SELECT", "JOIN", "索引")),
)


def normalize_query(question: str) -> str:
    """Normalize Unicode punctuation, case and whitespace deterministically."""

    text = unicodedata.normalize("NFKC", str(question or ""))
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def _alias_occurs(alias: str, text: str) -> bool:
    normalized_alias = normalize_query(alias)
    if not normalized_alias:
        return False
    if re.search(r"[a-z0-9]", normalized_alias):
        return re.search(
            rf"(?<![a-z0-9_]){re.escape(normalized_alias)}(?![a-z0-9_])",
            text,
        ) is not None
    return normalized_alias in text


@dataclass(frozen=True)
class QueryEntity:
    canonical: str
    matched_alias: str
    expansions: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    normalized_query: str
    rewritten_query: str
    intent: str
    expanded_terms: list[str] = field(default_factory=list)
    entities: list[QueryEntity] = field(default_factory=list)
    dictionary_source: str = "builtin"
    dictionary_version: str = "builtin-v1"
    selected_expansions: list[str] = field(default_factory=list)

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "dictionary": {
                "source": self.dictionary_source if self.dictionary_source in {"builtin", "external"} else "builtin",
                "version": self.dictionary_version if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", self.dictionary_version or "") else "unknown",
            },
            "entity_count": len(self.entities),
            "expanded_term_count": len(self.expanded_terms),
            "selected_expansion_count": len(self.selected_expansions),
        }


class QueryPlanner:
    """Build a small, explainable query plan without network/model calls."""

    def __init__(self, dictionary: TermDictionary | None = None) -> None:
        self.dictionary = dictionary or TermDictionary.builtin()

    def plan(self, question: str) -> QueryPlan:
        normalized = normalize_query(question)
        entities: list[QueryEntity] = []
        terms: list[str] = []
        seen_entities: set[str] = set()
        for entry in self.dictionary.entities():
            canonical, aliases, expansions = entry.canonical, entry.aliases, entry.expansions
            matched = next((alias for alias in aliases if _alias_occurs(alias, normalized)), None)
            if matched is None or canonical in seen_entities:
                continue
            seen_entities.add(canonical)
            entities.append(QueryEntity(canonical, matched, expansions, aliases))
            terms.extend((canonical, *aliases, *expansions))

        intent = self._intent(normalized)
        intent_expansions = self.dictionary.intent_expansions().get(intent, _INTENT_MARKERS.get(intent, ()))
        terms.extend(intent_expansions)
        required = list(self.dictionary.required_terms())
        for value in (*self.dictionary.generic_terms(), *self.dictionary.broad_terms(), *self.dictionary.technical_hints()):
            if _alias_occurs(value, normalized):
                terms.append(value)
        selected_expansions = self._dedupe(
            [entity.canonical for entity in entities]
            + [value for entity in entities for value in entity.expansions]
            + list(intent_expansions)
        )
        # Preserve meaningful raw tokens even if they are not in the compact
        # dictionary; this improves exact identifier matching.
        terms.extend(re.findall(r"[a-z0-9_+#:.\-/]{2,}|[\u4e00-\u9fff]{2,}", normalized))
        expanded = self._dedupe([*required, *terms])[:40]
        budget = 24
        selected = self._dedupe([normalized, *required, *expanded])
        rewritten = " ".join(selected[: budget + 1 + len(required)])
        return QueryPlan(
            original_query=str(question or ""),
            normalized_query=normalized,
            rewritten_query=rewritten,
            intent=intent,
            expanded_terms=expanded,
            entities=entities,
            dictionary_source=self.dictionary.source,
            dictionary_version=self.dictionary.version,
            selected_expansions=selected_expansions,
        )

    @staticmethod
    def _intent(normalized: str) -> str:
        best = "general"
        best_count = 0
        for intent, markers in _INTENT_MARKERS.items():
            count = sum(1 for marker in markers if normalize_query(marker) in normalized)
            if count > best_count:
                best, best_count = intent, count
        return best

    @staticmethod
    def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = " ".join(str(value).split()).strip()
            key = normalize_query(cleaned)
            if not cleaned or key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result


__all__ = ["QueryEntity", "QueryPlan", "QueryPlanner", "normalize_query"]
