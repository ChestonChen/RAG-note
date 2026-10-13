"""Hybrid retrieval over the rebuildable PostgreSQL/pgvector index."""

from __future__ import annotations

import math
import re
import time
from contextvars import ContextVar
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from sqlalchemy import func, literal_column, select
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .embeddings import EmbeddingProvider, create_embedding_provider
from .models import Chunk, Document
from .query_planning import QueryPlanner
from .term_dictionary import TermDictionary
from .reranker import Reranker, create_reranker
from .search_text import build_full_text_query, build_search_text
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


def _term_occurs(term: str, text: str) -> bool:
    """Match ASCII identifiers on token boundaries and CJK terms by span."""

    normalized_term = _normalized_text(term)
    normalized_text = _normalized_text(text)
    if not normalized_term or not normalized_text:
        return False
    if re.search(r"[a-z0-9]", normalized_term):
        # Avoid substring false positives such as ``set`` in ``dataset`` while
        # still allowing symbols commonly used by technical identifiers.
        pattern = rf"(?<![a-z0-9_]){re.escape(normalized_term)}(?![a-z0-9_])"
        return re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None
    return normalized_term in normalized_text


def _topic_alignment(question: str, plan: Any, document: Document, chunk: Chunk) -> tuple[float, bool, bool, list[str]]:
    """Score whether a candidate stays on an explicitly named topic.

    This is deliberately metadata-first: a topic in the title, source path, or
    heading is strong evidence; a topic found only in the body is weak unless
    another specific query term is present.  The result is used only as a
    bounded candidate guard when a strong topic candidate exists, so generic
    queries and documents without planner entities retain the old behavior.
    """

    entities = list(getattr(plan, "entities", []) or [])
    if not entities:
        return 0.0, False, False, []
    metadata = " ".join(
        [document.title, document.source_path, " ".join(chunk.heading_path or [])]
    )
    body = chunk.content or ""
    metadata_match = False
    body_match = False
    for entity in entities:
        aliases = [
            str(getattr(entity, "canonical", "")),
            str(getattr(entity, "matched_alias", "")),
            *(str(value) for value in (getattr(entity, "aliases", ()) or ())),
        ]
        if any(_term_occurs(alias, metadata) for alias in aliases if alias):
            metadata_match = True
        if any(_term_occurs(alias, body) for alias in aliases if alias):
            body_match = True

    # Anchors are technical terms supplied by the user beyond the broad topic
    # entity. CJK bigrams are too generic for this purpose, so keep 3+ chars.
    entity_terms = {
        str(value).casefold()
        for entity in entities
        for value in (getattr(entity, "canonical", ""), getattr(entity, "matched_alias", ""))
        if value
    }
    anchor_terms: list[str] = []
    for term in _terms(question):
        key = term.casefold()
        if key in entity_terms:
            continue
        if re.search(r"[a-z0-9]", key) or len(key) >= 3:
            anchor_terms.append(term)
    anchors = list(dict.fromkeys(term for term in anchor_terms if _term_occurs(term, metadata + " " + body)))
    if metadata_match:
        score = 1.0
    elif body_match and anchors:
        score = 0.65
    elif body_match:
        score = 0.2
    elif len(anchors) >= 2:
        score = 0.45
    elif anchors:
        score = 0.25
    else:
        score = 0.0
    return score, metadata_match, body_match, anchors


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
        term_weight = max(
            (weight for value, weight in fields if _term_occurs(term, value)),
            default=0.0,
        )
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
    score = sum(_term_occurs(term, haystack) for term in terms) / len(terms)
    return min(1.0, score + (0.25 if query in haystack else 0.0))


def _search_text_for(document: Document, chunk: Chunk) -> str:
    """Use the stored derived field and repair blank legacy values in memory."""

    stored = str(getattr(chunk, "search_text", "") or "")
    if stored.strip():
        return stored
    return build_search_text(
        title=document.title,
        heading_path=chunk.heading_path or [],
        source_path=document.source_path,
        content=chunk.content,
    )


def _full_text_score(question: str, document: Document, chunk: Chunk) -> float:
    terms = _terms(question)
    haystack = _search_text_for(document, chunk)
    if not terms or not haystack:
        return 0.0
    matches = sum(1 for term in terms if _term_occurs(term, haystack))
    score = matches / len(terms)
    if _normalized_text(question) in _normalized_text(haystack):
        score += 0.25
    return min(1.0, score)


def _exact_score(question: str, document: Document, chunk: Chunk) -> float:
    """Score exact query/identifier matches in metadata only."""

    query = _normalized_text(question)
    core = _normalized_text(_core_phrase(question))
    metadata_fields = (
        (document.title, 1.0),
        (" ".join(chunk.heading_path or []), 0.95),
        (document.source_path, 0.75),
    )
    variants = [value for value in dict.fromkeys((query, core)) if value]
    for value, weight in metadata_fields:
        normalized = _normalized_text(value)
        if any(
            candidate == normalized
            or (
                candidate in normalized
                and (
                    " " in candidate
                    or not re.search(r"[a-z0-9]", candidate)
                    or _term_occurs(candidate, normalized)
                )
            )
            for candidate in variants
        ):
            return weight
    terms = _ascii_identifiers(question)
    for value, weight in metadata_fields:
        if any(_term_occurs(term, value) for term in terms):
            return weight * 0.85
    return 0.0


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
    trigram_score: float = 0.0
    full_text_score: float = 0.0
    exact_score: float = 0.0
    rerank_score: float | None = None
    topic_score: float = 0.0
    topic_metadata_match: bool = False
    topic_body_match: bool = False
    topic_anchor_hits: list[str] | None = None
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
        reranker: Reranker | Any | None = None,
        query_planner: QueryPlanner | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings or get_settings()
        self.embedding_provider = embedding_provider or create_embedding_provider(self.settings)
        self.reranker = reranker if reranker is not None else create_reranker(self.settings)
        self.query_planner = query_planner or QueryPlanner(
            TermDictionary.load(self.settings.resolved_term_dictionary_path())
        )
        self.latest_debug: dict[str, Any] = {
            "requested_limit": None,
            "primary_count": 0,
            "channels": {
                "vector": [],
                "keyword": [],
                "phrase": [],
                "full_text": [],
                "exact": [],
                "trigram": [],
            },
            "fusion_order": [],
            "query_plan": {},
            "reranker": {"enabled": bool(self.settings.reranker_enabled), "attempted": False, "status": "disabled"},
            "topic_filter": {"enabled": False, "applied": False, "filtered_count": 0},
            "timings_ms": {},
        }
        self.last_debug = self.latest_debug
        self._debug_context: ContextVar[dict[str, Any]] = ContextVar(
            "rag_retrieval_debug", default=self.latest_debug
        )

    def get_last_debug(self) -> dict[str, Any]:
        """Return a request-local copy of the most recent retrieval trace."""

        return deepcopy(self._debug_context.get(self.latest_debug))

    def _debug_state(self) -> dict[str, Any]:
        """Return the trace owned by the current request context."""

        return self._debug_context.get(self.latest_debug)

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
                .where(distance <= 1.0 - self.settings.retrieval_vector_min_score)
                .order_by(distance)
                .limit(self.settings.retrieval_vector_candidates)
            )
            rows = list(session.execute(statement).all())
            ids = [int(chunk_id) for chunk_id, _ in rows]
            scores = {int(chunk_id): max(0.0, 1.0 - float(distance_value)) for chunk_id, distance_value in rows}
            return ids, scores
        except Exception:
            session.rollback()
            return None

    def _postgres_trigram_scores(
        self, session: Session, question: str
    ) -> tuple[list[int], dict[int, float]] | None:
        """Use pg_trgm as an optional fuzzy lexical channel on PostgreSQL.

        The query uses SQLAlchemy expressions and bound parameters.  SQLite and
        older databases without the extension simply use the Python lexical
        channel instead.
        """

        bind = session.get_bind()
        if bind is None or bind.dialect.name != "postgresql":
            return None
        try:
            similarity = func.greatest(
                func.similarity(Chunk.content, question),
                func.similarity(Document.title, question),
                func.similarity(Document.source_path, question),
            )
            statement = (
                select(Chunk.id, similarity.label("similarity"))
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == "ready")
                .where(similarity >= self.settings.retrieval_trigram_min_similarity)
                .order_by(similarity.desc())
                .limit(self.settings.retrieval_trigram_candidates)
            )
            result = session.execute(statement)
            rows = result.all() if hasattr(result, "all") else list(result)
            ids = [int(chunk_id) for chunk_id, _ in rows]
            scores = {int(chunk_id): max(0.0, min(1.0, float(score))) for chunk_id, score in rows}
            return ids, scores
        except Exception:
            rollback = getattr(session, "rollback", None)
            if callable(rollback):
                rollback()
            return None

    def _postgres_full_text_scores(
        self, session: Session, question: str
    ) -> tuple[list[int], dict[int, float]] | None:
        """Run PostgreSQL simple-dictionary FTS with a bound query parameter."""

        bind = session.get_bind()
        if bind is None or bind.dialect.name != "postgresql":
            return None
        try:
            simple = literal_column("'simple'")
            document_vector = func.to_tsvector(simple, Chunk.search_text)
            query_text = build_full_text_query(question)
            query_vector = func.websearch_to_tsquery(simple, query_text)
            rank = func.ts_rank_cd(document_vector, query_vector)
            statement = (
                select(Chunk.id, rank.label("full_text_score"))
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == "ready")
                .where(rank > 0)
                .order_by(rank.desc(), Chunk.id)
                .limit(self.settings.retrieval_full_text_candidates)
            )
            result = session.execute(statement)
            rows = result.all() if hasattr(result, "all") else list(result)
            ids = [int(chunk_id) for chunk_id, _ in rows]
            scores = {
                int(chunk_id): max(0.0, min(1.0, float(score)))
                for chunk_id, score in rows
            }
            return ids, scores
        except Exception:
            rollback = getattr(session, "rollback", None)
            if callable(rollback):
                rollback()
            return None

    def _exact_scores(
        self,
        session: Session,
        question: str,
        rows: list[tuple[Chunk, Document]] | None = None,
    ) -> tuple[list[int], dict[int, float]]:
        """Return an independent exact metadata channel on every database."""

        rows = rows if rows is not None else self._rows(session)
        scored = [
            (chunk.id, _exact_score(question, document, chunk))
            for chunk, document in rows
        ]
        scored = [(chunk_id, score) for chunk_id, score in scored if score > 0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        scored = scored[: self.settings.retrieval_exact_candidates]
        return [chunk_id for chunk_id, _ in scored], {chunk_id: score for chunk_id, score in scored}

    def _python_full_text_scores(
        self,
        question: str,
        rows: list[tuple[Chunk, Document]],
    ) -> tuple[list[int], dict[int, float]]:
        scored = [
            (chunk.id, _full_text_score(question, document, chunk))
            for chunk, document in rows
        ]
        scored = [(chunk_id, score) for chunk_id, score in scored if score > 0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        scored = scored[: self.settings.retrieval_full_text_candidates]
        return [chunk_id for chunk_id, _ in scored], {chunk_id: score for chunk_id, score in scored}

    def search(self, question: str, limit: int | None = None) -> list[SearchHit]:
        started = time.perf_counter()
        plan = self.query_planner.plan(question)
        normalized = plan.normalized_query
        requested_limit = limit or self.settings.retrieval_context_limit
        self.latest_debug = {
            "requested_limit": requested_limit,
            "primary_count": 0,
            "channels": {
                "vector": [],
                "keyword": [],
                "phrase": [],
                "full_text": [],
                "exact": [],
                "trigram": [],
            },
            "fusion_order": [],
            "query_plan": plan.to_debug_dict(),
            "reranker": {
                "enabled": bool(self.settings.reranker_enabled),
                "attempted": False,
                "status": "disabled" if self.reranker is None else "ready",
            },
            "topic_filter": {"enabled": bool(plan.entities), "applied": False, "filtered_count": 0},
            "timings_ms": {},
        }
        self.last_debug = self.latest_debug
        self._debug_context.set(self.latest_debug)
        if not normalized:
            return []
        limit = requested_limit
        vector_started = time.perf_counter()
        vector_query = plan.rewritten_query if self.settings.retrieval_query_expansion_enabled else normalized
        vector = self.embedding_provider.embed_query(vector_query)
        self._debug_state()["timings_ms"]["embedding"] = round((time.perf_counter() - vector_started) * 1000, 3)
        with self.session_factory() as session:
            retrieval_started = time.perf_counter()
            rows = self._rows(session)
            if not rows:
                self._debug_state()["timings_ms"]["retrieval"] = round((time.perf_counter() - retrieval_started) * 1000, 3)
                return []

            candidates: dict[int, _Candidate] = {}
            postgres_vector = self._postgres_vector_scores(session, vector)
            postgres_trigram = self._postgres_trigram_scores(session, normalized)
            full_text_started = time.perf_counter()
            postgres_full_text = self._postgres_full_text_scores(session, normalized)
            if postgres_full_text is None:
                full_text_ids, full_text_scores = self._python_full_text_scores(normalized, rows)
            else:
                full_text_ids, full_text_scores = postgres_full_text
            self._debug_state()["timings_ms"]["full_text"] = round(
                (time.perf_counter() - full_text_started) * 1000, 3
            )
            exact_started = time.perf_counter()
            exact_ids, exact_scores = self._exact_scores(session, normalized, rows)
            self._debug_state()["timings_ms"]["exact"] = round(
                (time.perf_counter() - exact_started) * 1000, 3
            )
            for chunk, document in rows:
                candidate = _Candidate(chunk=chunk, document=document)
                candidate.vector_score = max(0.0, _cosine(vector, chunk.embedding or []))
                # Keep lexical matching tied to the words the user supplied.
                # Planner expansions are useful for semantic embeddings, but
                # scoring every alias as an exact keyword makes unrelated
                # documents (for example an SQL note for a MySQL question)
                # look relevant.
                lexical_question = normalized
                candidate.keyword_score, candidate.matched_terms = _keyword_details(
                    lexical_question, document, chunk
                )
                candidate.phrase_score = _phrase_score(normalized, document, chunk)
                candidate.full_text_score = full_text_scores.get(chunk.id, 0.0)
                candidate.exact_score = exact_scores.get(chunk.id, 0.0)
                (
                    candidate.topic_score,
                    candidate.topic_metadata_match,
                    candidate.topic_body_match,
                    candidate.topic_anchor_hits,
                ) = _topic_alignment(normalized, plan, document, chunk)
                candidates[chunk.id] = candidate

            # When the query names a concrete topic and at least one candidate
            # identifies that topic in metadata, discard body-only mentions
            # unless they also match at least two independent query anchors.
            # One generic body mention (for example ``JOIN`` in a Redis note)
            # is not enough evidence to enter a MySQL context, while a section
            # that matches several specific anchors can still be retained.
            strong_topic = any(item.topic_metadata_match for item in candidates.values())
            if strong_topic and plan.entities:
                filtered_ids = [
                    chunk_id
                    for chunk_id, item in candidates.items()
                    if not item.topic_metadata_match and len(item.topic_anchor_hits or []) < 2
                ]
                for chunk_id in filtered_ids:
                    candidates.pop(chunk_id, None)
                self._debug_state()["topic_filter"] = {
                    "enabled": True,
                    "applied": bool(filtered_ids),
                    "filtered_count": len(filtered_ids),
                    "filtered_chunk_ids": filtered_ids[:20],
                    "reason": "explicit topic metadata outranked body-only mentions",
                }

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
                    if item.vector_score >= self.settings.retrieval_vector_min_score
                ]
            # A concurrent ingestion can commit between the channel query and
            # the row snapshot. Keep only IDs that have a loaded candidate so
            # debug serialization and fusion cannot dereference stale rows.
            vector_ids = [item_id for item_id in vector_ids if item_id in candidates]
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
            if postgres_trigram is not None:
                trigram_ids, trigram_scores = postgres_trigram
                for chunk_id, score in trigram_scores.items():
                    if chunk_id in candidates:
                        candidates[chunk_id].trigram_score = score
            else:
                trigram_ids = []
            trigram_ids = [item_id for item_id in trigram_ids if item_id in candidates]
            full_text_ids = [item_id for item_id in full_text_ids if item_id in candidates]
            exact_ids = [item_id for item_id in exact_ids if item_id in candidates]
            channels = {
                "vector": vector_ids,
                "keyword": keyword_ids,
                "phrase": phrase_ids,
                "full_text": full_text_ids,
                "exact": exact_ids,
                "trigram": trigram_ids,
            }
            weights = {
                "vector": self.settings.retrieval_vector_rrf_weight,
                "keyword": self.settings.retrieval_keyword_rrf_weight,
                "phrase": self.settings.retrieval_phrase_rrf_weight,
                "full_text": self.settings.retrieval_full_text_rrf_weight,
                "exact": self.settings.retrieval_exact_rrf_weight,
                "trigram": self.settings.retrieval_trigram_rrf_weight,
            }
            fused = reciprocal_rank_fusion(channels, weights=weights)
            self._debug_state()["channels"] = {
                name: [{"chunk_id": item_id, "score": getattr(candidates[item_id], f"{name}_score", 0.0)} for item_id in ids]
                for name, ids in channels.items()
            }
            if not fused:
                return []

            channel_sets = {name: set(ids) for name, ids in channels.items()}
            ranked = sorted(
                fused,
                # RRF remains the dominant signal. A small metadata prior
                # resolves near ties in favor of an exact phrase and metadata
                # hit, keeping a document titled with the queried concept
                # ahead of an incidental body-only mention.
                key=lambda chunk_id: (
                    fused[chunk_id]
                    + 0.01 * candidates[chunk_id].phrase_score
                    + 0.005 * candidates[chunk_id].keyword_score,
                    fused[chunk_id],
                    candidates[chunk_id].phrase_score,
                    candidates[chunk_id].keyword_score,
                    candidates[chunk_id].vector_score,
                ),
                reverse=True,
            )

            ranked = self._apply_reranker(normalized, ranked, candidates, fused)
            self._debug_state()["fusion_order"] = list(ranked)
            self._debug_state()["timings_ms"]["retrieval"] = round(
                (time.perf_counter() - retrieval_started) * 1000, 3
            )
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
                bounded = (
                    0.50 * candidate.vector_score
                    + 0.18 * candidate.keyword_score
                    + 0.25 * candidate.phrase_score
                    + 0.06 * candidate.full_text_score
                    + 0.10 * candidate.exact_score
                    + 0.07 * candidate.trigram_score
                    + 0.10 * min(1.0, fusion * 10)
                )
                if candidate.rerank_score is not None:
                    bounded = 0.75 * bounded + 0.25 * candidate.rerank_score
                bounded = min(1.0, max(0.0, bounded))
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
                        trigram_score=candidate.trigram_score,
                        full_text_score=candidate.full_text_score,
                        exact_score=candidate.exact_score,
                        fusion_score=fusion,
                        matched_channels=matched,
                        matched_terms=list(candidate.matched_terms or []),
                        topic_score=candidate.topic_score,
                        topic_match=(
                            "metadata"
                            if candidate.topic_metadata_match
                            else "body_with_anchor"
                            if candidate.topic_body_match and candidate.topic_anchor_hits
                            else "body_only"
                            if candidate.topic_body_match
                            else "anchor_only"
                            if candidate.topic_anchor_hits
                            else "none"
                        ),
                        rerank_score=candidate.rerank_score,
                    )
                )
                if len(result) >= limit:
                    break

            for rank, hit in enumerate(result, start=1):
                hit.rank = rank
            self._debug_state()["primary_count"] = len(result)

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
                                full_text_score=adjacent.full_text_score,
                                exact_score=adjacent.exact_score,
                                fusion_score=primary.fusion_score * 0.5,
                                matched_channels=["neighbor"],
                                topic_score=adjacent.topic_score,
                                topic_match="neighbor",
                                is_neighbor=True,
                            )
                        )
                        neighbor_budget -= 1
                        if neighbor_budget <= 0:
                            break
                    if neighbor_budget <= 0:
                        break
            return result

    def _apply_reranker(
        self,
        question: str,
        ranked: list[int],
        candidates: dict[int, _Candidate],
        fused: dict[int, float],
    ) -> list[int]:
        """Optionally rerank the fused candidate tail, failing open by default."""

        if self.reranker is None or not ranked:
            return ranked
        candidate_ids = ranked[: self.settings.reranker_candidates]
        documents = [
            " ".join(
                [
                    candidates[item_id].document.title,
                    " ".join(candidates[item_id].chunk.heading_path or []),
                    candidates[item_id].chunk.content,
                ]
            )
            for item_id in candidate_ids
        ]
        debug = self._debug_state()["reranker"]
        debug["attempted"] = True
        debug["candidate_count"] = len(candidate_ids)
        started = time.perf_counter()
        try:
            scores = list(self.reranker.rerank(question, documents))
            if len(scores) != len(candidate_ids):
                raise RuntimeError("reranker returned an unexpected number of scores")
            finite = [float(score) for score in scores if math.isfinite(float(score))]
            if len(finite) != len(scores) or not finite:
                raise RuntimeError("reranker returned non-finite scores")
            low, high = min(finite), max(finite)
            scale = high - low
            if scale == 0:
                debug["status"] = "no_discrimination"
                debug["scores"] = {}
                return ranked
            for item_id, score in zip(candidate_ids, scores):
                value = float(score)
                normalized_score = (value - low) / scale
                candidates[item_id].rerank_score = max(0.0, min(1.0, normalized_score))
            rerank_scores = {
                str(item_id): round(float(candidates[item_id].rerank_score or 0.0), 4)
                for item_id in candidate_ids
            }
            debug["status"] = "ok"
            debug["scores"] = rerank_scores
            return sorted(
                ranked,
                key=lambda item_id: (
                    fused[item_id]
                    + self.settings.reranker_weight * (candidates[item_id].rerank_score or 0.0),
                    fused[item_id],
                    candidates[item_id].phrase_score,
                    candidates[item_id].keyword_score,
                ),
                reverse=True,
            )
        except Exception as exc:
            debug["status"] = "failed_open" if self.settings.reranker_fail_open else "failed"
            debug["error"] = str(exc)[:200]
            if not self.settings.reranker_fail_open:
                raise
            return ranked
        finally:
            self._debug_state()["timings_ms"]["reranker"] = round((time.perf_counter() - started) * 1000, 3)


__all__ = ["RetrievalService", "reciprocal_rank_fusion"]
