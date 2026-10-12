"""Bounded context assembly and citation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import Settings, get_settings
from .schemas import SearchHit, SourceCitation


def _rough_tokens(text: str) -> int:
    # A predictable approximation works for both English and Chinese and keeps
    # context assembly independent from a particular tokenizer implementation.
    return max(1, len(text) // 4)


@dataclass
class ContextPackage:
    text: str
    citations: list[SourceCitation]
    hits: list[SearchHit]
    confidence_score: float


class ContextBuilder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def build(self, question: str, hits: Iterable[SearchHit]) -> ContextPackage:
        del question  # reserved for future query-aware context compression
        selected: list[SearchHit] = []
        citations: list[SourceCitation] = []
        citation_by_key: dict[tuple[str, int | None, int | None, tuple[str, ...]], str] = {}
        used_chunks: set[int] = set()
        used_tokens = 0
        lines: list[str] = []

        for hit in hits:
            if hit.chunk_id in used_chunks:
                continue
            chunk_tokens = _rough_tokens(hit.content)
            if selected and used_tokens + chunk_tokens > self.settings.chunk_max_tokens * 3:
                break
            used_chunks.add(hit.chunk_id)
            selected.append(hit)
            used_tokens += chunk_tokens
            key = (hit.source, hit.page_from, hit.page_to, tuple(hit.heading))
            citation_id = citation_by_key.get(key)
            if citation_id is None:
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
            location = hit.source
            if hit.heading:
                location += " > " + " > ".join(hit.heading)
            if hit.page_from is not None:
                location += f" (p.{hit.page_from}"
                if hit.page_to and hit.page_to != hit.page_from:
                    location += f"-{hit.page_to}"
                location += ")"
            lines.append(f"[{citation_id}] {location}\n{hit.content.strip()}")

        # RRF scores are not probabilities.  This bounded value is only used
        # to decide whether an answer has enough evidence to call the LLM.
        top = max((hit.score for hit in selected), default=0.0)
        channel_bonus = 0.05 if selected and any(len(hit.matched_channels) > 1 for hit in selected) else 0.0
        confidence = min(1.0, top + channel_bonus)
        return ContextPackage(
            text="\n\n".join(lines),
            citations=citations,
            hits=selected,
            confidence_score=confidence,
        )


__all__ = ["ContextBuilder", "ContextPackage"]
