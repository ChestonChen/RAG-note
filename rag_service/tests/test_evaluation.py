from __future__ import annotations

import json
from pathlib import Path

from rag_service.app.context import ContextBuilder
from rag_service.app.schemas import SearchHit
from scripts.evaluate_retrieval import evaluate_cases, load_cases


class FakeRetrievalService:
    def __init__(self, responses: dict[str, list[SearchHit]]) -> None:
        self.responses = responses

    def search(self, question: str, limit: int | None = None) -> list[SearchHit]:
        hits = self.responses[question]
        return hits if limit is None else hits[:limit]


def _hit(
    chunk_id: int,
    source: str,
    score: float,
    *,
    rank: int | None = None,
    neighbor: bool = False,
    heading: str = "Notes",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        content=f"Evidence from {source}",
        source=source,
        title=source,
        heading=[heading],
        score=score,
        rank=rank,
        is_neighbor=neighbor,
        matched_channels=["keyword"],
    )


def test_evaluate_cases_calculates_source_metrics_and_unknown_false_positives():
    cases = [
        {
            "id": "a",
            "question": "first",
            "relevant_sources": ["a.md"],
            "answerable": True,
        },
        {
            "id": "b",
            "question": "second",
            "relevant_sources": ["b.md", "c.md"],
            "answerable": True,
        },
        {
            "id": "unknown",
            "question": "unknown",
            "relevant_sources": [],
            "answerable": False,
        },
    ]
    retrieval = FakeRetrievalService(
        {
            "first": [
                _hit(1, "d.md", 0.8, rank=1),
                _hit(2, "a.md", 0.7, rank=2),
                _hit(3, "a.md", 0.4, neighbor=True),
            ],
            "second": [
                _hit(4, "b.md", 0.8, rank=1),
                _hit(5, "d.md", 0.6, rank=2),
                _hit(6, "c.md", 0.5, rank=4),
            ],
            "unknown": [_hit(7, "d.md", 0.4, rank=1)],
        }
    )

    report = evaluate_cases(retrieval, ContextBuilder(), cases, limit=5, confidence_threshold=0.28)

    assert report["metrics"] == {
        "hit_at_1": 0.5,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "mrr": 0.75,
        "mean_source_recall_at_5": 1.0,
        "citation_source_coverage": 1.0,
        "unknown_false_positive_rate": 0.0,
    }
    unknown = next(case for case in report["cases"] if case["id"] == "unknown")
    assert unknown["primary_hits"][0]["source"] == "d.md"
    assert unknown["neighbor_hits"] == []
    assert unknown["confidence_score"] == ContextBuilder().build(
        "unknown", retrieval.responses["unknown"]
    ).evidence.score


def test_load_cases_reads_jsonl_and_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"one","question":"Q","relevant_sources":["a.md"],"answerable":true}\n\n',
        encoding="utf-8",
    )

    assert load_cases(path) == [
        {"id": "one", "question": "Q", "relevant_sources": ["a.md"], "answerable": True}
    ]
