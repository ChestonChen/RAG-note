"""Bounded context assembly and citation metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .config import Settings, get_settings
from .schemas import EvidenceReport, SearchHit, SourceCitation, WebSource


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

    def build(self, question: str, hits: Iterable[SearchHit]) -> ContextPackage:
        selected: list[SearchHit] = []
        citations: list[SourceCitation] = []
        citation_by_key: dict[tuple[str, int | None, int | None, tuple[str, ...]], str] = {}
        used_chunks: set[int] = set()
        used_tokens = 0
        lines: list[str] = []
        eligible: list[SearchHit] = []
        terms = list(dict.fromkeys(
            re.findall(r"[a-z0-9][a-z0-9_+-]*|[\u4e00-\u9fff]{2,}", question.casefold())
        ))

        for hit in hits:
            if hit.chunk_id in used_chunks:
                continue
            chunk_tokens = _rough_tokens(hit.content)
            if selected and used_tokens + chunk_tokens > self.settings.chunk_max_tokens * 3:
                break
            used_chunks.add(hit.chunk_id)
            selected.append(hit)
            used_tokens += chunk_tokens
            weak_topic = hit.topic_match in {"body_only", "anchor_only"}
            is_direct = not hit.is_neighbor and "neighbor" not in hit.matched_channels
            is_eligible = (
                is_direct
                and not weak_topic
                and hit.score >= self.settings.evidence_min_citation_score
            )
            if is_eligible:
                eligible.append(hit)
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
        direct = [hit for hit in selected if not hit.is_neighbor and "neighbor" not in hit.matched_channels]
        top = max((hit.score for hit in eligible), default=0.0)
        second = sorted((hit.score for hit in eligible), reverse=True)[1:2]
        score_gap = max(0.0, top - second[0]) if second else top
        channel_agreement = (
            sum(min(1.0, max(0, len(set(hit.matched_channels) - {"neighbor"}) - 1) / 2) for hit in eligible)
            / len(eligible) if eligible else 0.0
        )
        corpus = " ".join(
            " ".join([hit.title, " ".join(hit.heading), hit.content, " ".join(hit.matched_terms)])
            for hit in eligible
        )
        covered = sum(1 for term in terms if term in corpus.casefold())
        term_coverage = covered / len(terms) if terms else 0.0
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
                weak_topic = any(hit.topic_match in {"body_only", "anchor_only"} for hit in direct)
                if weak_topic:
                    caps.append("weak_evidence")
                else:
                    caps.append("no_citation_eligible_hits")
            if eligible:
                reasons.append("direct_evidence")
            if any(hit.phrase_score > 0 or hit.exact_score > 0 for hit in eligible):
                reasons.append("exact_or_phrase_match")
            if channel_agreement:
                reasons.append("independent_channel_agreement")
        confidence = min(1.0, max(0.0, confidence))
        evidence = EvidenceReport(
            score=confidence,
            direct_hit_count=len(direct),
            eligible_citation_count=len(citations),
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
