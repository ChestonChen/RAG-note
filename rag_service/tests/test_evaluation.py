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

    assert {key: report["metrics"][key] for key in (
        "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "mean_source_recall_at_5",
        "citation_source_coverage", "unknown_false_positive_rate",
        "expected_channel_recall", "relevant_heading_recall",
    )} == {
        "hit_at_1": 0.5,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "mrr": 0.75,
        "mean_source_recall_at_5": 1.0,
            "citation_source_coverage": 0.0,
        "unknown_false_positive_rate": 0.0,
        "expected_channel_recall": 0.0,
        "relevant_heading_recall": 0.0,
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
        '{"id":"one","question":"Q","relevant_sources":["a.md"],"answerable":true,"evidence_groups":[["answer"]]}\n\n',
        encoding="utf-8",
    )

    assert load_cases(path) == [
        {"id": "one", "question": "Q", "relevant_sources": ["a.md"], "answerable": True, "evidence_groups": [["answer"]]}
    ]


def test_evaluation_report_records_profile_and_dataset_metadata():
    cases = [{
        "id": "profile-case",
        "question": "profile question",
        "relevant_sources": ["a.md"],
        "answerable": True,
    }]
    retrieval = FakeRetrievalService({
        "profile question": [_hit(1, "a.md", 0.8, rank=1)],
    })

    report = evaluate_cases(
        retrieval,
        ContextBuilder(),
        cases,
        profile="reference_like",
        dataset={"name": "gitee-note-repository", "version": "2026-10-14"},
    )

    assert report["metadata"]["profile"] == "reference_like"
    assert report["metadata"]["dataset"] == {
        "name": "gitee-note-repository",
        "version": "2026-10-14",
    }
    assert report["metadata"]["profile_applied"] is False
    assert report["metadata"]["profile_semantics"] == "descriptive_label_only"


def test_evaluation_report_sanitizes_urls_in_dataset_metadata():
    cases = [{
        "id": "safe-metadata",
        "question": "metadata question",
        "relevant_sources": [],
        "answerable": False,
    }]
    report = evaluate_cases(
        FakeRetrievalService({"metadata question": []}),
        ContextBuilder(),
        cases,
        dataset={
            "source_dataset_url": "https://user:secret@example.test/data.json?token=hidden&ref=main",
        },
    )

    assert report["metadata"]["dataset"]["source_dataset_url"] == "https://example.test/data.json?ref=main"


def test_evaluation_normalizes_benchmark_root_in_relevant_sources():
    cases = [{
        "id": "benchmark-source",
        "question": "source question",
        "relevant_sources": ["gitee-benchmark/C++/vector复习.pdf"],
        "answerable": True,
    }]
    report = evaluate_cases(
        FakeRetrievalService({"source question": [_hit(1, "C++/vector复习.pdf", 0.8, rank=1)]}),
        ContextBuilder(),
        cases,
    )

    assert report["metrics"]["hit_at_1"] == 1.0
