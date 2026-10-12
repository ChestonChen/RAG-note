"""Hybrid retrieval over the rebuildable PostgreSQL/pgvector index."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .embeddings import EmbeddingProvider, create_embedding_provider
from .models import Chunk, Document
from .schemas import SearchHit


def reciprocal_rank_fusion(channels: dict[str, list[int]], k: int = 60) -> dict[int, float]:
    """Fuse ranked ids from independent channels using reciprocal rank fusion."""

    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[int, float] = defaultdict(float)
    for ranked_ids in channels.values():
        seen: set[int] = set()
        for rank, item_id in enumerate(ranked_ids, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)


def _terms(question: str) -> list[str]:
    # Keep identifiers such as ``pgvector`` and Chinese runs intact while
    # dropping punctuation that otherwise creates noisy substring matches.
    return [term for term in re.findall(r"[\w\u4e00-\u9fff]+", question.casefold()) if term]


def _keyword_score(question: str, content: str) -> float:
    query = " ".join(question.casefold().split())
    haystack = " ".join(content.casefold().split())
    if not query or not haystack:
        return 0.0
    terms = _terms(question)
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in haystack)
    score = hits / len(terms)
    if query in haystack:
        score = min(1.0, score + 0.25)
    return score


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
        if not normalized:
            return []
        limit = limit or self.settings.retrieval_context_limit
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
                candidate.keyword_score = _keyword_score(normalized, chunk.content)
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
            channels = {"vector": vector_ids, "keyword": keyword_ids}
            fused = reciprocal_rank_fusion(channels)
            if not fused:
                return []

            channel_sets = {name: set(ids) for name, ids in channels.items()}
            ranked = sorted(
                fused,
                key=lambda chunk_id: (fused[chunk_id], candidates[chunk_id].vector_score),
                reverse=True,
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
                        fusion_score=fusion,
                        matched_channels=matched,
                    )
                )
                if len(result) >= limit:
                    break

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
                                fusion_score=primary.fusion_score * 0.5,
                                matched_channels=["neighbor"],
                            )
                        )
                        neighbor_budget -= 1
                        if neighbor_budget <= 0:
                            break
                    if neighbor_budget <= 0:
                        break
            return result


__all__ = ["RetrievalService", "reciprocal_rank_fusion"]
