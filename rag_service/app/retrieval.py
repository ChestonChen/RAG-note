"""Hybrid retrieval over the rebuildable PostgreSQL/pgvector index."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .embeddings import EmbeddingProvider, create_embedding_provider
from .models import Chunk, Document
from .schemas import SearchHit


def reciprocal_rank_fusion(
    channels: dict[str, list[int]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> dict[int, float]:
    """Fuse ranked ids from independent channels using reciprocal rank fusion."""

    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[int, float] = defaultdict(float)
    weights = weights or {}
    for channel_name, ranked_ids in channels.items():
        weight = float(weights.get(channel_name, 1.0))
        if weight <= 0:
            continue
        seen: set[int] = set()
        for rank, item_id in enumerate(ranked_ids, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] += weight / (k + rank)
    return dict(scores)


_CHINESE_STOP_WORDS = {
    "什么", "怎么", "如何", "为什么", "请问", "可以", "是否", "以及", "哪些", "这个", "那个",
    "的", "了", "吗", "呢", "是", "有", "在", "和", "与", "或", "我", "你", "它", "能",
}
_ASCII_STOP_WORDS = {
    "a", "an", "the", "and", "or", "for", "how", "what", "why", "does", "do", "is", "are",
    "can", "please", "explain", "tell", "about", "of", "to", "in", "on",
}


def _ascii_identifiers(text: str) -> list[str]:
    return [
        match.casefold()
        for match in re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-+]+[A-Za-z0-9]*)*", text)
        if match.casefold() not in _ASCII_STOP_WORDS
    ]


def _chinese_ngrams(text: str) -> list[str]:
    terms: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        # Remove filler words before forming n-grams so their boundary
        # characters cannot create noisy terms such as ``么是``.
        useful_runs = [part for part in re.split("|".join(map(re.escape, _CHINESE_STOP_WORDS)), run) if part]
        for useful in useful_runs:
            for size in (2, 3):
                terms.extend(useful[index : index + size] for index in range(len(useful) - size + 1))
    return terms


def _terms(question: str) -> list[str]:
    terms = _ascii_identifiers(question) + _chinese_ngrams(question.casefold())
    return list(dict.fromkeys(term for term in terms if term))


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _core_phrase(question: str) -> str:
    ascii_terms = _ascii_identifiers(question)
    chinese = ""
    for run in re.findall(r"[\u4e00-\u9fff]+", question):
        cleaned = run
        for stop_word in sorted(_CHINESE_STOP_WORDS, key=len, reverse=True):
            cleaned = cleaned.replace(stop_word, " ")
        chinese += "".join(cleaned.split())
    return " ".join(ascii_terms + ([chinese] if chinese else []))


def _keyword_details(question: str, document: Document, chunk: Chunk) -> tuple[float, list[str]]:
    terms = _terms(question)
    if not terms:
        return 0.0, []
    fields = (
        (" ".join(chunk.heading_path or []), 2.8),
        (document.title, 2.4),
        (document.source_path, 1.8),
        (chunk.content, 1.0),
    )
    matched: list[str] = []
    weighted_hits = 0.0
    total_weight = sum(weight for _, weight in fields)
    for term in terms:
        term_weight = max((weight for value, weight in fields if term in _normalized_text(value)), default=0.0)
        if term_weight:
            weighted_hits += term_weight
            matched.append(term)
    return min(1.0, weighted_hits / max(total_weight, 1.0) * 2.0), matched


def _phrase_score(question: str, document: Document, chunk: Chunk) -> float:
    query = _normalized_text(question)
    if not query:
        return 0.0
    haystack = _normalized_text(
        " ".join([document.title, document.source_path, " ".join(chunk.heading_path or []), chunk.content])
    )
    if query in haystack:
        return 1.0
    core = _normalized_text(_core_phrase(question))
    if core and core in haystack:
        return 0.7
    return 0.0


def _keyword_score(question: str, content: str) -> float:
    # Kept as a small compatibility helper for callers that used the former
    # body-only scorer; retrieval itself uses the metadata-aware scorer.
    query = _normalized_text(question)
    haystack = _normalized_text(content)
    terms = _terms(question)
    if not query or not haystack or not terms:
        return 0.0
    score = sum(term in haystack for term in terms) / len(terms)
    return min(1.0, score + (0.25 if query in haystack else 0.0))


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    a = list(left)
    b = list(right)
    if len(a) != len(b) or not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


@dataclass
class _Candidate:
    chunk: Chunk
    document: Document
    vector_score: float = 0.0
    keyword_score: float = 0.0
    phrase_score: float = 0.0
    matched_terms: list[str] | None = None


class RetrievalService:
    """Run vector and lexical retrieval, then fuse their ranks.

    The SQLite/in-memory path intentionally computes scores in Python so unit
    tests can exercise the complete behavior without a running PostgreSQL.
    PostgreSQL still gets the pgvector distance expression for the vector
    channel; lexical scoring remains deliberately transparent for this MVP.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session] | Any,
        *,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings or get_settings()
        self.embedding_provider = embedding_provider or create_embedding_provider(self.settings)
        self.latest_debug: dict[str, Any] = {
            "requested_limit": None,
            "primary_count": 0,
            "channels": {"vector": [], "keyword": [], "phrase": []},
            "fusion_order": [],
        }
        self.last_debug = self.latest_debug

    def _rows(self, session: Session) -> list[tuple[Chunk, Document]]:
        statement = (
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.status == "ready")
        )
        return list(session.execute(statement).all())

    def _postgres_vector_scores(
        self, session: Session, vector: list[float]
    ) -> tuple[list[int], dict[int, float]] | None:
        """Ask pgvector for the vector channel when the active DB is PostgreSQL.

        The transparent Python path remains useful for SQLite tests and also
        acts as a fallback for an older schema that cannot compile the vector
        operator.  A failed optional optimization must never make keyword
        retrieval unavailable.
        """

        bind = session.get_bind()
        if bind is None or bind.dialect.name != "postgresql":
            return None
        try:
            distance = Chunk.embedding.cosine_distance(vector)
            statement = (
                select(Chunk.id, distance.label("distance"))
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == "ready")
                .order_by(distance)
                .limit(self.settings.retrieval_vector_candidates)
            )
            rows = list(session.execute(statement).all())
            ids = [int(chunk_id) for chunk_id, _ in rows]
            scores = {int(chunk_id): max(0.0, 1.0 - float(distance_value)) for chunk_id, distance_value in rows}
            return ids, scores
        except Exception:
            return None

    def search(self, question: str, limit: int | None = None) -> list[SearchHit]:
        normalized = " ".join(question.split())
        requested_limit = limit or self.settings.retrieval_context_limit
        self.latest_debug = {
            "requested_limit": requested_limit,
            "primary_count": 0,
            "channels": {"vector": [], "keyword": [], "phrase": []},
            "fusion_order": [],
        }
        self.last_debug = self.latest_debug
        if not normalized:
            return []
        limit = requested_limit
        vector = self.embedding_provider.embed_query(normalized)
        with self.session_factory() as session:
            rows = self._rows(session)
            if not rows:
                return []

            candidates: dict[int, _Candidate] = {}
            postgres_vector = self._postgres_vector_scores(session, vector)
            for chunk, document in rows:
                candidate = _Candidate(chunk=chunk, document=document)
                candidate.vector_score = max(0.0, _cosine(vector, chunk.embedding or []))
                candidate.keyword_score, candidate.matched_terms = _keyword_details(normalized, document, chunk)
                candidate.phrase_score = _phrase_score(normalized, document, chunk)
                candidates[chunk.id] = candidate

            if postgres_vector is not None:
                vector_ids, postgres_scores = postgres_vector
                for chunk_id, score in postgres_scores.items():
                    if chunk_id in candidates:
                        candidates[chunk_id].vector_score = score
            else:
                vector_ids = [
                    item.chunk.id
                    for item in sorted(
                        candidates.values(), key=lambda item: item.vector_score, reverse=True
                    )[: self.settings.retrieval_vector_candidates]
                ]
            keyword_ids = [
                item.chunk.id
                for item in sorted(
                    candidates.values(),
                    key=lambda item: item.keyword_score,
                    reverse=True,
                )
                if item.keyword_score > 0
            ][: self.settings.retrieval_keyword_candidates]
            phrase_ids = [
                item.chunk.id
                for item in sorted(
                    candidates.values(), key=lambda item: item.phrase_score, reverse=True
                )
                if item.phrase_score > 0
            ][: self.settings.retrieval_phrase_candidates]
            channels = {"vector": vector_ids, "keyword": keyword_ids, "phrase": phrase_ids}
            weights = {
                "vector": self.settings.retrieval_vector_rrf_weight,
                "keyword": self.settings.retrieval_keyword_rrf_weight,
                "phrase": self.settings.retrieval_phrase_rrf_weight,
            }
            fused = reciprocal_rank_fusion(channels, weights=weights)
            self.latest_debug["channels"] = {
                name: [{"chunk_id": item_id, "score": getattr(candidates[item_id], f"{name}_score", 0.0)} for item_id in ids]
                for name, ids in channels.items()
            }
            if not fused:
                return []

            channel_sets = {name: set(ids) for name, ids in channels.items()}
            ranked = sorted(
                fused,
                key=lambda chunk_id: (fused[chunk_id], candidates[chunk_id].vector_score),
                reverse=True,
            )
            self.latest_debug["fusion_order"] = list(ranked)
            result: list[SearchHit] = []
            per_document: dict[int, int] = defaultdict(int)
            for chunk_id in ranked:
                candidate = candidates[chunk_id]
                document_id = candidate.document.id
                if per_document[document_id] >= self.settings.retrieval_per_document_limit:
                    continue
                per_document[document_id] += 1
                matched = [name for name, ids in channel_sets.items() if chunk_id in ids]
                fusion = fused[chunk_id]
                # ``score`` is a bounded display score, not a probability.  It
                # is useful to compare hits while the raw channel scores remain
                # visible to the debug endpoint.
                bounded = min(
                    1.0,
                    0.55 * candidate.vector_score
                    + 0.35 * candidate.keyword_score
                    + 0.10 * min(1.0, fusion * 10),
                )
                result.append(
                    SearchHit(
                        chunk_id=chunk_id,
                        content=candidate.chunk.content,
                        source=candidate.document.source_path,
                        title=candidate.document.title,
                        heading=list(candidate.chunk.heading_path or []),
                        page_from=candidate.chunk.page_from,
                        page_to=candidate.chunk.page_to,
                        score=bounded,
                        vector_score=candidate.vector_score,
                        keyword_score=candidate.keyword_score,
                        phrase_score=candidate.phrase_score,
                        fusion_score=fusion,
                        matched_channels=matched,
                        matched_terms=list(candidate.matched_terms or []),
                    )
                )
                if len(result) >= limit:
                    break

            for rank, hit in enumerate(result, start=1):
                hit.rank = rank
            self.latest_debug["primary_count"] = len(result)

            # Adjacent chunks often contain the definition or example that a
            # top-ranked chunk refers to. Add a small bounded window after
            # primary ranking, keeping neighbor hits visibly separate from
            # evidence that matched the query directly.
            neighbor_window = self.settings.retrieval_neighbor_window
            if result and neighbor_window:
                by_position = {
                    (candidate.document.id, candidate.chunk.chunk_index): candidate
                    for candidate in candidates.values()
                }
                selected_ids = {hit.chunk_id for hit in result}
                neighbor_budget = neighbor_window * 2
                for primary in list(result):
                    primary_candidate = candidates.get(primary.chunk_id)
                    if primary_candidate is None:
                        continue
                    for offset in range(-neighbor_window, neighbor_window + 1):
                        if offset == 0 or neighbor_budget <= 0:
                            continue
                        adjacent = by_position.get(
                            (
                                primary_candidate.document.id,
                                primary_candidate.chunk.chunk_index + offset,
                            )
                        )
                        if adjacent is None or adjacent.chunk.id in selected_ids:
                            continue
                        selected_ids.add(adjacent.chunk.id)
                        result.append(
                            SearchHit(
                                chunk_id=adjacent.chunk.id,
                                content=adjacent.chunk.content,
                                source=adjacent.document.source_path,
                                title=adjacent.document.title,
                                heading=list(adjacent.chunk.heading_path or []),
                                page_from=adjacent.chunk.page_from,
                                page_to=adjacent.chunk.page_to,
                                score=max(0.0, primary.score * 0.75),
                                vector_score=adjacent.vector_score,
                                keyword_score=adjacent.keyword_score,
                                phrase_score=adjacent.phrase_score,
                                fusion_score=primary.fusion_score * 0.5,
                                matched_channels=["neighbor"],
                                is_neighbor=True,
                            )
                        )
                        neighbor_budget -= 1
                        if neighbor_budget <= 0:
                            break
                    if neighbor_budget <= 0:
                        break
            return result


__all__ = ["RetrievalService", "reciprocal_rank_fusion"]
