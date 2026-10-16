"""Editable, validated query term dictionaries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class TermEntry:
    canonical: str
    aliases: tuple[str, ...]
    expansions: tuple[str, ...]


_BUILTIN_ENTITIES = (
    ("pgvector", ("pgvector", "向量数据库", "向量检索", "vector database"), ("embedding", "余弦相似度", "PostgreSQL")),
    ("rag", ("rag", "RAG", "检索增强生成", "检索增强"), ("retrieval", "embedding", "上下文", "引用")),
    ("embedding", ("embedding", "嵌入", "向量化", "向量表示"), ("向量", "语义相似度")),
    ("stl", ("STL", "标准模板库", "模板库"), ("C++", "容器", "迭代器", "算法")),
    ("vector", ("vector", "std::vector", "动态数组", "顺序表"), ("C++", "扩容", "容量", "迭代器")),
    ("tcp", ("TCP", "传输控制协议", "三次握手", "四次挥手"), ("可靠传输", "连接")),
    ("udp", ("UDP", "用户数据报协议"), ("无连接", "数据报")),
    ("redis", ("Redis", "缓存", "键值数据库"), ("内存数据库", "持久化", "高可用")),
    ("mysql", ("MySQL", "SQL", "关系型数据库"), ("SELECT", "JOIN", "索引")),
)
_BUILTIN_INTENTS = {
    "definition": ("什么是", "是什么", "定义", "概念", "介绍", "作用"),
    "implementation": ("原理", "底层", "实现", "机制", "架构", "怎么做", "如何做"),
    "comparison": ("区别", "对比", "比较", "不同", "优缺点"),
    "usage": ("怎么用", "如何使用", "用法", "命令", "示例", "实践"),
    "troubleshooting": ("报错", "失败", "异常", "阻塞", "排查", "怎么办"),
}
_BUILTIN_GENERIC = ("什么是", "如何", "怎么", "原理", "实现")
_BUILTIN_BROAD = ("数据库", "系统", "网络", "编程")
_BUILTIN_TECHNICAL = ("RAG", "pgvector", "MySQL", "Redis", "TCP", "vector")


def _strings(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


class TermDictionary:
    def __init__(
        self,
        entries: tuple[TermEntry, ...],
        *,
        generic: tuple[str, ...] = (),
        broad: tuple[str, ...] = (),
        technical: tuple[str, ...] = (),
        required: tuple[str, ...] = (),
        intents: Mapping[str, tuple[str, ...]] | None = None,
        source: str = "builtin",
        version: str = "builtin-v1",
    ) -> None:
        self._entries = entries
        self._generic = generic
        self._broad = broad
        self._technical = technical
        self._required = required
        self._intents = MappingProxyType(dict(intents or {}))
        self.source = source
        self.version = version

    @classmethod
    def builtin(cls) -> "TermDictionary":
        return cls(
            tuple(TermEntry(c, tuple(a), tuple(e)) for c, a, e in _BUILTIN_ENTITIES),
            generic=_BUILTIN_GENERIC,
            broad=_BUILTIN_BROAD,
            technical=_BUILTIN_TECHNICAL,
            required=(),
            intents=_BUILTIN_INTENTS,
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "TermDictionary":
        baseline = cls.builtin()
        if path is None:
            return baseline
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("dictionary root must be an object")
            entries = {entry.canonical: entry for entry in baseline._entries}
            raw_entities = payload.get("entities", {})
            if isinstance(raw_entities, list):
                if any(not isinstance(item, dict) or not isinstance(item.get("canonical"), str) for item in raw_entities):
                    raise ValueError("invalid entity list")
                raw_entities = {item["canonical"]: item for item in raw_entities}
            if not isinstance(raw_entities, dict):
                raise ValueError("entities must be an object or list")
            for canonical, raw in raw_entities.items():
                if not isinstance(canonical, str) or not canonical.strip() or not isinstance(raw, dict):
                    raise ValueError("invalid entity")
                key = canonical.strip()
                old = entries.get(key, TermEntry(key, (), ()))
                if "aliases" in raw and "synonyms" in raw:
                    raise ValueError("entity aliases and synonyms are ambiguous")
                alias_key = "aliases" if "aliases" in raw else "synonyms"
                aliases = tuple(dict.fromkeys((*old.aliases, *(_strings(raw[alias_key], field="aliases") if alias_key in raw else ()))))
                expansions = tuple(dict.fromkeys((*old.expansions, *(_strings(raw["expansions"], field="expansions") if "expansions" in raw else ()))))
                entries[key] = TermEntry(key, aliases or (key,), expansions)
            raw_synonyms = payload.get("synonyms", {})
            if raw_synonyms is not None:
                if not isinstance(raw_synonyms, dict):
                    raise ValueError("synonyms must be an object")
                for canonical, values in raw_synonyms.items():
                    if not isinstance(canonical, str) or not canonical.strip():
                        raise ValueError("invalid synonym key")
                    canonical = canonical.strip()
                    old = entries.get(canonical, TermEntry(canonical, (canonical,), ()))
                    entries[canonical] = TermEntry(
                        canonical,
                        tuple(dict.fromkeys((*old.aliases, *_strings(values, field="synonyms")))),
                        old.expansions,
                    )
            intents = dict(baseline._intents)
            raw_intents = payload.get("intent_expansions", {})
            if raw_intents is not None:
                if not isinstance(raw_intents, dict):
                    raise ValueError("intent_expansions must be an object")
                for intent, values in raw_intents.items():
                    if not isinstance(intent, str) or not intent.strip():
                        raise ValueError("invalid intent")
                    intent = intent.strip()
                    intents[intent] = tuple(dict.fromkeys((*intents.get(intent, ()), *_strings(values, field="intent_expansions"))))
            generic = tuple(dict.fromkeys((*baseline._generic, *(_strings(payload["generic_terms"], field="generic_terms") if "generic_terms" in payload else ()))))
            broad = tuple(dict.fromkeys((*baseline._broad, *(_strings(payload["broad_terms"], field="broad_terms") if "broad_terms" in payload else ()))))
            technical = tuple(dict.fromkeys((*baseline._technical, *(_strings(payload["technical_hints"], field="technical_hints") if "technical_hints" in payload else ()))))
            required = _strings(payload["required_terms"], field="required_terms") if "required_terms" in payload else ()
            version = payload.get("version", "external-v1")
            if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", version):
                raise ValueError("version must be a safe identifier")
            return cls(tuple(entries.values()), generic=generic, broad=broad, technical=technical, required=required, intents=intents, source="external", version=version)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            return baseline

    def entities(self) -> tuple[TermEntry, ...]:
        return self._entries

    def generic_terms(self) -> tuple[str, ...]:
        return self._generic

    def broad_terms(self) -> tuple[str, ...]:
        return self._broad

    def technical_hints(self) -> tuple[str, ...]:
        return self._technical

    def required_terms(self) -> tuple[str, ...]:
        return self._required

    def synonyms(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType({entry.canonical: entry.aliases for entry in self._entries})

    def intent_expansions(self) -> Mapping[str, tuple[str, ...]]:
        return self._intents


__all__ = ["TermDictionary", "TermEntry"]
