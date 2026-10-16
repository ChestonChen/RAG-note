"""Bounded context assembly and citation metadata."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from .config import Settings, get_settings
from .query_planning import QueryPlanner
from .schemas import EvidenceReport, SearchHit, SourceCitation, WebSource
from .term_dictionary import TermDictionary


def _rough_tokens(text: str) -> int:
    # A predictable approximation works for both English and Chinese and keeps
    # context assembly independent from a particular tokenizer implementation.
    return max(1, len(text) // 4)


_QUESTION_STOP_WORDS = {
    "a", "an", "and", "are", "can", "does", "do", "during", "for", "from", "how",
    "in", "is", "it", "mean", "new", "of", "on", "or", "the", "to", "use", "what",
    "which", "why", "with", "about", "explain", "tell", "please", "work", "works",
    "什么", "怎么", "如何", "为什么", "请问", "可以", "是否", "以及", "哪些", "这个", "那个",
    "的", "了", "吗", "呢", "是", "有", "在", "和", "与", "或", "我", "你", "它", "能",
    "分别", "介绍", "说明", "解释", "怎样", "如何做", "怎么做", "什么是", "是什么",
}
_DISCOVERY_MARKERS = (
    "which notes", "which documents", "which files", "what notes", "what documents",
    "哪些笔记", "哪些文档", "哪些文件", "哪些资料", "哪些知识",
)
_DISCOVERY_QUERY_PATTERN = re.compile(r"\bwhich\b.{0,48}\b(?:notes|documents|files)\b")
_RELATION_MARKERS = (
    "cause", "control", "affect", "influence", "impact", "depend", "determine", "schedule",
    "trigger", "lead to", "关系", "影响", "控制", "导致", "决定", "依赖", "调度", "触发",
)
_CAUSAL_MARKERS = (
    "cause", "control", "affect", "influence", "impact", "depend", "determine", "schedule",
    "trigger", "lead to", "影响", "控制", "导致", "决定", "依赖", "调度", "触发",
)
_COMPARISON_MARKERS = (
    "differ", "different", "compare", "comparison", "versus", "vs.",
    "区别", "对比", "比较", "不同",
)
_EVIDENCE_TERM_ALIASES = {
    "store": ("stores", "stored", "storage", "存储", "储存", "保存", "存放", "放在"),
    "distance": ("距离",),
    "semantic": ("语义",),
    "retrieval": ("检索",),
    "represent": ("表示", "代表"),
    "write": ("写入", "追加", "保存"),
    "connection": ("连接",),
    "datagrams": ("数据报", "数据包"),
    "bytes": ("字节",),
    "similar": ("相似",),
    "reliable": ("可靠",),
    "ordered": ("有序",),
    "delivery": ("传输", "交付"),
    "相似": ("similar", "similarity"),
    "文本": ("text", "passage", "chunk"),
    "可靠": ("reliable", "reliability"),
    "有序": ("ordered", "in order"),
    "数据传输": ("transport", "transmission", "delivery"),
    "数据": ("data", "bytes"),
    "字节": ("bytes",),
}
_SUBTOPIC_REQUIREMENTS = (
    (re.compile(r"\b(?:elect|election)\b"), (("election", "elect", "vote", "voting", "quorum", "majority"),)),
    (re.compile(r"\bmvcc\b"), (("mvcc",),)),
    (re.compile(r"\bgap\s+locks?\b"), (("gap lock", "next-key lock"),)),
    (re.compile(r"\b(?:cluster|failover)\b.*\belect\b|\belect\b.*\b(?:cluster|failover)\b"), (("election", "elect", "vote", "voting", "quorum", "majority"),)),
)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def _contains_term(term: str, text: str) -> bool:
    term = _normalize(term)
    text = _normalize(text)
    if not term or not text:
        return False
    if re.search(r"[a-z0-9]", term):
        tokens = re.findall(r"[a-z0-9][a-z0-9_+#.-]*", text)
        stem = term[:6] if len(term) >= 7 else term
        return any(token == term or (len(stem) >= 5 and token.startswith(stem)) for token in tokens)
    return term in text


def _entity_aliases(entity: object) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            value
            for value in (
                str(getattr(entity, "canonical", "")),
                str(getattr(entity, "matched_alias", "")),
                *(str(alias) for alias in (getattr(entity, "aliases", ()) or ())),
            )
            if value
        )
    )


@dataclass
class ContextPackage:
    text: str
    citations: list[SourceCitation]
    hits: list[SearchHit]
    confidence_score: float
    evidence: EvidenceReport = field(
        default_factory=lambda: EvidenceReport(
            score=0,
            direct_hit_count=0,
            eligible_citation_count=0,
            channel_agreement=0,
            term_coverage=0,
            metadata_alignment=0,
            score_gap=0,
            document_diversity=0,
        )
    )
    web_sources: list[WebSource] = field(default_factory=list)


class ContextBuilder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.query_planner = QueryPlanner(
            TermDictionary.load(self.settings.resolved_term_dictionary_path())
        )

    @staticmethod
    def _question_terms(question: str, entities: list[object]) -> list[str]:
        entity_text = " ".join(
            alias for entity in entities for alias in _entity_aliases(entity)
        )
        terms: list[str] = []
        for term in re.findall(r"[a-z0-9][a-z0-9_+#:.-]*", _normalize(question)):
            if term in _QUESTION_STOP_WORDS or len(term) < 3:
                continue
            if any(_contains_term(alias, term) for alias in entity_text.split() if len(alias) >= 3):
                continue
            if not _contains_term(term, entity_text):
                terms.append(term)

        cleaned = _normalize(question)
        for stop in sorted(_QUESTION_STOP_WORDS, key=len, reverse=True):
            if not re.search(r"[a-z0-9]", stop):
                cleaned = cleaned.replace(stop, " ")
        for run in re.findall(r"[\u4e00-\u9fff]+", cleaned):
            if len(run) >= 2:
                terms.append(run)
            for width in (2, 3):
                terms.extend(run[index : index + width] for index in range(len(run) - width + 1))

        unique: list[str] = []
        for term in terms:
            normalized = _normalize(term)
            if (
                normalized
                and normalized not in _QUESTION_STOP_WORDS
                and not _contains_term(normalized, entity_text)
                and normalized not in unique
            ):
                unique.append(normalized)
        return unique

    @staticmethod
    def _direct(hit: SearchHit) -> bool:
        return not hit.is_neighbor and "neighbor" not in hit.matched_channels

    @staticmethod
    def _metadata_text(hit: SearchHit) -> str:
        return " ".join((hit.title, hit.source, *hit.heading))

    @staticmethod
    def _is_relationship_query(question: str, entity_count: int) -> bool:
        if entity_count < 2:
            return False
        normalized = _normalize(question)
        if any(marker in normalized for marker in _CAUSAL_MARKERS):
            return True
        return bool(
            re.search(r"(?:之间|彼此).{0,6}关系|关系.{0,6}(?:如何|是什么|是否|会)", normalized)
            or re.search(r"\brelationship\b|\brelates? to\b|\bbetween\b", normalized)
        )

    @staticmethod
    def _is_comparison_query(question: str, entity_count: int) -> bool:
        if entity_count < 2:
            return False
        normalized = _normalize(question)
        return any(marker in normalized for marker in _COMPARISON_MARKERS)

    @staticmethod
    def _term_in_body(term: str, body: str) -> bool:
        normalized_term = _normalize(term)
        aliases = [term, *_EVIDENCE_TERM_ALIASES.get(term, ())]
        aliases.extend(
            alias
            for key, values in _EVIDENCE_TERM_ALIASES.items()
            if key in normalized_term
            for alias in values
        )
        return any(
            _contains_term(alias, body)
            for alias in aliases
        )

    @staticmethod
    def _required_subtopic_groups(question: str) -> tuple[tuple[str, ...], ...]:
        normalized = _normalize(question)
        return tuple(
            alternatives
            for pattern, alternatives in _SUBTOPIC_REQUIREMENTS
            if pattern.search(normalized)
        )

    def _assess_hits(
        self, question: str, direct_hits: list[SearchHit]
    ) -> tuple[list[SearchHit], float, bool, str]:
        plan = self.query_planner.plan(question)
        entities = list(plan.entities)
        terms = self._question_terms(question, entities)
        question_text = _normalize(question)
        relation_query = self._is_relationship_query(question, len(entities))
        comparison_query = self._is_comparison_query(question, len(entities))
        required_subtopics = self._required_subtopic_groups(question)
        discovery_query = (
            len(entities) >= 2
            and (
                any(marker in question_text for marker in _DISCOVERY_MARKERS)
                or bool(_DISCOVERY_QUERY_PATTERN.search(question_text))
            )
        )
        score_qualified = [
            hit
            for hit in direct_hits
            if hit.score >= self.settings.evidence_min_citation_score
        ]

        if discovery_query:
            supported_entities: dict[str, list[SearchHit]] = {}
            for entity in entities:
                aliases = _entity_aliases(entity)
                supported_entities[str(getattr(entity, "canonical", ""))] = [
                    hit
                    for hit in score_qualified
                    if any(_contains_term(alias, hit.content) for alias in aliases)
                ]
            if supported_entities and all(supported_entities.values()):
                discovery_hits: list[SearchHit] = []
                discovery_ids: set[int] = set()
                for entity_hits in supported_entities.values():
                    for hit in entity_hits:
                        if hit.chunk_id not in discovery_ids:
                            discovery_hits.append(hit)
                            discovery_ids.add(hit.chunk_id)
                coverage = len(supported_entities) / len(entities)
                return discovery_hits, coverage, False, "multi_source_discovery"
            return [], 0.0, False, "multi_source_evidence_incomplete"

        if comparison_query:
            comparison_terms = [
                term
                for term in terms
                if term not in _COMPARISON_MARKERS
                and not any(marker in term for marker in _COMPARISON_MARKERS)
            ]
            supported_entities = {
                str(getattr(entity, "canonical", "")): [
                    hit
                    for hit in score_qualified
                    if any(_contains_term(alias, hit.content) for alias in _entity_aliases(entity))
                    and (
                        not comparison_terms
                        or any(self._term_in_body(term, hit.content) for term in comparison_terms)
                    )
                ]
                for entity in entities
            }
            if supported_entities and all(supported_entities.values()):
                comparison_hits: list[SearchHit] = []
                comparison_ids: set[int] = set()
                for entity_hits in supported_entities.values():
                    for hit in entity_hits:
                        if hit.chunk_id not in comparison_ids:
                            comparison_hits.append(hit)
                            comparison_ids.add(hit.chunk_id)
                return comparison_hits, 1.0, False, "multi_source_comparison"
            return [], 0.0, False, "comparison_entity_evidence_incomplete"

        relation_supported: list[SearchHit] = []
        if relation_query:
            for hit in score_qualified:
                if all(
                    any(_contains_term(alias, hit.content) for alias in _entity_aliases(entity))
                    for entity in entities
                ) and any(_contains_term(marker, hit.content) for marker in _RELATION_MARKERS if marker in question_text):
                    relation_supported.append(hit)
            if not relation_supported:
                return [], 0.0, False, "relationship_not_supported_in_one_chunk"

        required_supported = [
            hit
            for hit in score_qualified
            if all(
                any(_contains_term(anchor, hit.content) for anchor in alternatives)
                for alternatives in required_subtopics
            )
        ]
        if required_subtopics and not required_supported:
            return [], 0.0, False, "required_subtopic_not_supported"

        candidates = (
            relation_supported
            if relation_query
            else required_supported
            if required_subtopics
            else score_qualified
        )
        eligible: list[SearchHit] = []
        total_anchor_count = len(terms) + len(entities)
        matched_anchor_keys: set[str] = set()
        for hit in candidates:
            body_terms = {term for term in terms if self._term_in_body(term, hit.content)}
            body_entities = {
                str(getattr(entity, "canonical", ""))
                for entity in entities
                if any(_contains_term(alias, hit.content) for alias in _entity_aliases(entity))
            }
            matched_anchor_keys.update(body_terms)
            matched_anchor_keys.update(body_entities)
            if hit.topic_match in {"body_only", "anchor_only"} and len(body_terms) < 2:
                continue
            if relation_query:
                eligible.append(hit)
            elif terms:
                if body_terms:
                    eligible.append(hit)
            elif body_entities:
                if hit.topic_match not in {"body_only", "anchor_only"}:
                    eligible.append(hit)

        if relation_query:
            coverage = 1.0 if eligible else 0.0
            return eligible, coverage, bool(eligible), "direct_relationship_evidence" if eligible else "relationship_not_supported_in_one_chunk"
        coverage = (
            min(1.0, len(matched_anchor_keys) / total_anchor_count)
            if total_anchor_count
            else (1.0 if eligible else 0.0)
        )
        if eligible:
            reason = "direct_body_evidence"
        elif not direct_hits:
            reason = "no_direct_hits"
        elif not score_qualified:
            reason = "below_citation_score"
        else:
            reason = "no_question_specific_body_anchor"
        return eligible, coverage, False, reason

    def build(self, question: str, hits: Iterable[SearchHit]) -> ContextPackage:
        selected: list[SearchHit] = []
        citations: list[SourceCitation] = []
        citation_by_key: dict[tuple[str, int | None, int | None, tuple[str, ...]], str] = {}
        used_chunks: set[int] = set()
        used_tokens = 0
        for hit in hits:
            if hit.chunk_id in used_chunks:
                continue
            chunk_tokens = _rough_tokens(hit.content)
            if selected and used_tokens + chunk_tokens > self.settings.chunk_max_tokens * 3:
                break
            used_chunks.add(hit.chunk_id)
            selected.append(hit)
            used_tokens += chunk_tokens

        direct = [hit for hit in selected if self._direct(hit)]
        eligible, anchor_coverage, relationship_evidence, gate_reason = self._assess_hits(
            question, direct
        )
        eligible_ids = {hit.chunk_id for hit in eligible}
        lines: list[str] = []
        for hit in selected:
            is_eligible = hit.chunk_id in eligible_ids
            key = (hit.source, hit.page_from, hit.page_to, tuple(hit.heading))
            citation_id = citation_by_key.get(key) if is_eligible else None
            if citation_id is None:
                if is_eligible:
                    citation_id = f"DOC-{len(citations) + 1}"
                    citation_by_key[key] = citation_id
                    citations.append(
                        SourceCitation(
                            source=hit.source,
                            title=hit.title,
                            heading=hit.heading,
                            page_from=hit.page_from,
                            page_to=hit.page_to,
                            score=hit.score,
                            citation_id=citation_id,
                        )
                    )
            if not is_eligible:
                continue
            location = hit.source
            if hit.heading:
                location += " > " + " > ".join(hit.heading)
            if hit.page_from is not None:
                location += f" (p.{hit.page_from}"
                if hit.page_to and hit.page_to != hit.page_from:
                    location += f"-{hit.page_to}"
                location += ")"
            label = f"[{citation_id}]" if citation_id else "[CONTEXT ONLY]"
            lines.append(f"{label} {location}\n{hit.content.strip()}")

        # RRF scores are not probabilities.  This bounded value is only used
        # to decide whether an answer has enough evidence to call the LLM.
        top = max((hit.score for hit in eligible), default=0.0)
        second = sorted((hit.score for hit in eligible), reverse=True)[1:2]
        score_gap = max(0.0, top - second[0]) if second else top
        channel_agreement = (
            sum(min(1.0, max(0, len(set(hit.matched_channels) - {"neighbor"}) - 1) / 2) for hit in eligible)
            / len(eligible) if eligible else 0.0
        )
        corpus = " ".join(hit.content for hit in eligible).casefold()
        terms = self._question_terms(question, list(self.query_planner.plan(question).entities))
        covered = sum(1 for term in terms if _contains_term(term, corpus))
        term_coverage = covered / len(terms) if terms else (1.0 if eligible else 0.0)
        metadata_alignment = (
            sum(
                bool(
                    hit.phrase_score
                    or hit.exact_score
                    or any(term in (hit.title + " " + " ".join(hit.heading)).casefold() for term in terms)
                )
                for hit in eligible
            )
            / len(eligible)
            if eligible
            else 0.0
        )
        diversity = min(1.0, len({hit.source for hit in eligible}) / 3) if eligible else 0.0
        caps: list[str] = []
        reasons: list[str] = []
        if not direct:
            confidence = 0.0
            reasons.append("no_direct_hits")
        else:
            confidence = (
                0.45 * top
                + 0.10 * channel_agreement
                + 0.12 * max((max(hit.phrase_score, hit.exact_score) for hit in eligible), default=0.0)
                + 0.12 * term_coverage
                + 0.10 * metadata_alignment
                + 0.06 * score_gap
                + 0.05 * diversity
            )
            if not eligible:
                confidence = min(
                    self.settings.evidence_weak_score_cap,
                    max((hit.score * 0.45 for hit in direct), default=0.0),
                )
                caps.append("insufficient_direct_body_evidence")
                if gate_reason == "relationship_not_supported_in_one_chunk":
                    caps.append("unsupported_cross_chunk_relationship")
            if eligible:
                reasons.append("direct_evidence")
            if any(hit.phrase_score > 0 or hit.exact_score > 0 for hit in eligible):
                reasons.append("exact_or_phrase_match")
            if channel_agreement:
                reasons.append("independent_channel_agreement")
        if eligible and confidence < self.settings.low_confidence_threshold:
            caps.append("below_answer_confidence_threshold")
        confidence = min(1.0, max(0.0, confidence))
        evidence = EvidenceReport(
            score=confidence,
            direct_hit_count=len(direct),
            eligible_citation_count=len(citations),
            evidence_gate_passed=bool(eligible),
            evidence_gate_reason=gate_reason,
            evidence_anchor_coverage=anchor_coverage,
            relationship_evidence=relationship_evidence,
            channel_agreement=channel_agreement,
            term_coverage=term_coverage,
            metadata_alignment=metadata_alignment,
            score_gap=score_gap,
            document_diversity=diversity,
            caps=caps,
            reasons=reasons,
        )
        return ContextPackage(
            text="\n\n".join(lines),
            citations=citations,
            hits=selected,
            confidence_score=confidence,
            evidence=evidence,
        )


__all__ = ["ContextBuilder", "ContextPackage"]
