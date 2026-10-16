from __future__ import annotations

import json
from pathlib import Path

from rag_service.app.config import Settings
from rag_service.app.context import ContextBuilder
from rag_service.app.schemas import SearchHit
from scripts.evaluate_retrieval import evaluate_cases, load_cases
from scripts.validate_embedding import main as validate_embedding_main
from rag_service.app.evaluation import EvaluationJudge


class FakeRetrieval:
    def __init__(self, responses: dict[str, list[SearchHit]], settings: Settings | None = None):
        self.responses = responses
        self.settings = settings

    def search(self, question: str, limit: int | None = None) -> list[SearchHit]:
        hits = self.responses[question]
        return hits if limit is None else hits[:limit]


def hit(chunk_id: int, source: str, score: float, *, channels: list[str], rank: int) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        content=f"Evidence from {source}",
        source=source,
        title=source,
        heading=["Notes"],
        score=score,
        rank=rank,
        matched_channels=channels,
    )


def test_load_cases_preserves_optional_metadata_and_defaults(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "paraphrase",
                "question": "Q",
                "relevant_sources": ["a.md"],
                "answerable": True,
                "evidence_groups": [["answer evidence"]],
                "tags": ["paraphrase", "multi-source"],
                "expected_channels": ["vector", "keyword"],
                "paraphrase_of": "base",
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "base",
                "question": "Base",
                "relevant_sources": [],
                "answerable": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_cases(path)[0]["tags"] == ["paraphrase", "multi-source"]
    assert load_cases(path)[0]["expected_channels"] == ["vector", "keyword"]
    assert load_cases(path)[0]["paraphrase_of"] == "base"


def test_load_cases_rejects_bad_types_and_unknown_paraphrase_reference(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "question": "Q",
                "relevant_sources": ["a.md"],
                "relevant_headings": ["Heading"],
                "answerable": True,
                "evidence_groups": [["answer evidence"]],
                "paraphrase_of": "missing",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="paraphrase_of"):
        load_cases(path)

    path.write_text(
        '{"id": 1, "question": "Q", "relevant_sources": [2], "answerable": "yes"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="id"):
        load_cases(path)


def test_load_cases_rejects_malformed_evidence_groups(tmp_path: Path):
    import pytest

    path = tmp_path / "bad-evidence.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "question": "Q",
                "relevant_sources": ["a.md"],
                "answerable": True,
                "evidence_groups": [["CREATE DATABASE"], [""]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence_groups"):
        load_cases(path)


def test_load_cases_requires_evidence_groups_for_answerable_cases(tmp_path: Path):
    import pytest

    path = tmp_path / "missing-evidence.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "question": "Q",
                "relevant_sources": ["a.md"],
                "answerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence_groups.*required"):
        load_cases(path)


def test_evidence_evaluation_matches_chunk_body_not_title_or_source_path():
    settings = Settings(low_confidence_threshold=0.28)
    retrieval = FakeRetrieval(
        {
            "create database": [
                SearchHit(
                    chunk_id=1,
                    content="A general database overview without the requested SQL.",
                    source="gitee/MySQL CREATE DATABASE guide.pdf",
                    title="CREATE DATABASE demo_db utf8mb4",
                    score=0.9,
                    rank=1,
                    matched_channels=["keyword"],
                ),
                SearchHit(
                    chunk_id=2,
                    content="CREATE DATABASE demo_db DEFAULT CHARSET=utf8mb4;",
                    source="gitee/MySQL CREATE DATABASE guide.pdf",
                    title="MySQL guide",
                    score=0.8,
                    rank=2,
                    matched_channels=["exact", "full_text"],
                ),
            ]
        },
        settings,
    )

    report = evaluate_cases(
        retrieval,
        ContextBuilder(settings),
        [
            {
                "id": "create-database",
                "question": "create database",
                "relevant_sources": ["gitee/MySQL CREATE DATABASE guide.pdf"],
                "answerable": True,
                "evidence_groups": [["CREATE DATABASE demo_db"], ["utf8mb4"]],
            }
        ],
    )

    case = report["cases"][0]
    assert case["evidence_evaluation"]["matched_group_indexes_at_1"] == []
    assert case["evidence_evaluation"]["matched_group_indexes_at_3"] == [0, 1]
    assert case["evidence_evaluation"]["hit_at_1"] is False
    assert case["evidence_evaluation"]["hit_at_3"] is True
    assert report["metrics"]["evidence_hit_at_1"] == 0.0
    assert report["metrics"]["evidence_hit_at_3"] == 1.0
    assert report["metrics"]["evidence_annotation_coverage"] == 1.0
    assert report["metrics"]["answerable_false_rejection_rate"] == 0.0


def test_answerability_metrics_use_evidence_eligibility_not_source_recall_alone():
    settings = Settings(low_confidence_threshold=0.28)
    retrieval = FakeRetrieval(
        {
            "specific answer": [
                SearchHit(
                    chunk_id=1,
                    content="A broad mention of the topic only.",
                    source="a.md",
                    title="Expected source",
                    score=0.8,
                    rank=1,
                    matched_channels=["vector", "keyword"],
                    topic_match="body_only",
                )
            ],
            "unknown": [hit(3, "a.md", 0.8, channels=["keyword"], rank=1)],
        },
        settings,
    )

    report = evaluate_cases(
        retrieval,
        ContextBuilder(settings),
        [
            {
                "id": "known",
                "question": "specific answer",
                "relevant_sources": ["a.md"],
                "answerable": True,
                "evidence_groups": [["The exact answer phrase"]],
            },
            {"id": "unknown", "question": "unknown", "relevant_sources": [], "answerable": False},
        ],
    )

    assert report["cases"][0]["relevant_primary_ranks"] == [1]
    assert report["cases"][0]["local_answer_eligible"] is False
    assert report["cases"][1]["local_answer_eligible"] is False
    assert report["metrics"]["answerable_false_rejection_rate"] == 1.0
    assert report["metrics"]["unknown_false_positive_rate"] == 0.0


def test_gold_evidence_hit_does_not_override_runtime_answerability_gate():
    settings = Settings(low_confidence_threshold=0.28)
    retrieval = FakeRetrieval(
        {
            "specific answer": [
                SearchHit(
                    chunk_id=1,
                    content="Specific answer evidence is present in the body.",
                    source="a.md",
                    title="Expected source",
                    score=0.9,
                    rank=1,
                    matched_channels=["vector", "keyword"],
                )
            ]
        },
        settings,
    )

    report = evaluate_cases(
        retrieval,
        ContextBuilder(settings),
        [
            {
                "id": "known",
                "question": "specific answer",
                "relevant_sources": ["a.md"],
                "answerable": True,
                "evidence_groups": [["phrase not present in retrieved text"]],
            }
        ],
    )

    assert report["cases"][0]["local_answer_eligible"] is True
    assert report["cases"][0]["evidence_evaluation"]["hit_at_5"] is False
    assert report["metrics"]["answerable_false_rejection_rate"] == 0.0


def test_evaluation_report_records_signature_config_buckets_channels_and_citation_precision():
    settings = Settings(
        embedding_provider="hashing",
        embedding_dimensions=384,
        low_confidence_threshold=0.28,
    )
    retrieval = FakeRetrieval(
        {
                "known": [
                SearchHit(chunk_id=1, content="known answer evidence", source="a.md", title="A", score=0.9, rank=1, matched_channels=["vector", "keyword"]),
                hit(2, "d.md", 0.5, channels=["keyword"], rank=2),
            ],
            "unknown": [SearchHit(chunk_id=3, content="unknown unsupported phrase", source="d.md", title="D", score=0.5, rank=1, matched_channels=["keyword"])],
        },
        settings,
    )
    report = evaluate_cases(
        retrieval,
        ContextBuilder(settings),
        [
            {"id": "known", "question": "known", "relevant_sources": ["a.md"], "answerable": True},
            {"id": "unknown", "question": "unknown", "relevant_sources": [], "answerable": False},
        ],
        confidence_threshold=0.28,
    )

    assert report["embedding_signature"] == settings.embedding_signature()
    assert report["embedding_runtime"]["provider"] == "hashing"
    assert report["retrieval_configuration"]["embedding_dimensions"] == 384
    assert set(report["confidence_buckets"]) == {"low", "medium", "high"}
    assert report["citation_source_precision"] == 1.0
    assert "keyword" in report["per_channel_contribution"]
    assert report["metrics"]["unknown_false_positive_rate"] >= 0
    assert report["cases"][0]["confidence_bucket"] in {"low", "medium", "high"}


def test_evaluation_scores_expected_channels_and_relevant_headings():
    settings = Settings()
    retrieval = FakeRetrieval(
        {
            "known": [
                SearchHit(
                    chunk_id=1,
                    content="Embedding evidence",
                    source="a.md",
                    title="A",
                    heading=["Guide", "Embeddings"],
                    score=0.9,
                    rank=1,
                    matched_channels=["vector", "keyword"],
                ),
                SearchHit(
                    chunk_id=2,
                    content="Unrelated evidence",
                    source="b.md",
                    title="B",
                    heading=["Other"],
                    score=0.5,
                    rank=2,
                    matched_channels=["keyword"],
                ),
            ]
        },
        settings,
    )

    report = evaluate_cases(
        retrieval,
        ContextBuilder(settings),
        [
            {
                "id": "known",
                "question": "known",
                "relevant_sources": ["a.md"],
                "relevant_headings": ["Embeddings", "Vector Search"],
                "expected_channels": ["vector", "full_text"],
                "answerable": True,
            }
        ],
    )

    case = report["cases"][0]
    assert case["channel_expectation"] == {
        "expected": ["vector", "full_text"],
        "observed": ["keyword", "vector"],
        "matched": ["vector"],
        "missing": ["full_text"],
        "recall": 0.5,
    }
    assert case["heading_evaluation"] == {
        "expected": ["Embeddings", "Vector Search"],
        "retrieved": ["Embeddings", "Guide"],
        "missing": ["Vector Search"],
        "recall": 0.5,
    }
    assert report["metrics"]["expected_channel_recall"] == 0.5
    assert report["metrics"]["relevant_heading_recall"] == 0.5


def test_unknown_false_positive_marks_high_confidence_unknown_case():
    settings = Settings()
    unknown_hit = hit(1, "d.md", 1.0, channels=["keyword"], rank=1).model_copy(
        update={"content": "The word unknown appears in this passage."}
    )
    retrieval = FakeRetrieval({"unknown": [unknown_hit]}, settings)
    report = evaluate_cases(
        retrieval,
        ContextBuilder(settings),
        [{"id": "unknown", "question": "unknown", "relevant_sources": [], "answerable": False}],
        confidence_threshold=0.28,
    )

    assert report["metrics"]["unknown_false_positive_rate"] == 1.0
    assert report["cases"][0]["unknown_false_positive"] is True


def test_reranker_comparison_is_unavailable_before_a_run():
    settings = Settings(reranker_enabled=True)
    retrieval = FakeRetrieval({"known": []}, settings)
    retrieval.get_last_debug = lambda: {
        "reranker": {"enabled": True, "attempted": False, "status": "no_discrimination"}
    }

    report = evaluate_cases(
        retrieval,
        ContextBuilder(settings),
        [{"id": "known", "question": "known", "relevant_sources": [], "answerable": True}],
    )

    assert report["reranker_comparison"]["available"] is False
    assert report["reranker_comparison"]["comparison_run"] is False


def test_eval_cases_include_real_mysql_and_redis_fixture_sources():
    cases = load_cases(Path("eval/retrieval_cases_knowledge.jsonl"))
    by_id = {case["id"]: case for case in cases}
    assert by_id["mysql-create-database"]["relevant_sources"] == [
        "gitee/MySQL 初始使用指南.pdf"
    ]
    assert by_id["redis-rdb-bgsave"]["relevant_sources"] == [
        "gitee/Redis 持久化机制深度解析（RDB _ AOF）.pdf"
    ]
    assert by_id["unknown-cross-topic-rewrite"]["answerable"] is False
    assert all(case.get("evidence_groups") for case in cases if case["answerable"])
    fixture_cases = {case["id"]: case for case in load_cases(Path("eval/retrieval_cases.jsonl"))}
    assert fixture_cases["mysql-redis-cross-topic"]["answerable"] is True
    assert fixture_cases["mysql-redis-cross-topic"]["evidence_groups"]


def test_reported_false_rejection_requires_gate_score_and_citation():
    settings = Settings(low_confidence_threshold=0.28)
    retrieval = FakeRetrieval(
        {
            "specific answer": [
                SearchHit(
                    chunk_id=1,
                    content="Specific answer evidence.",
                    source="a.md",
                    title="A",
                    score=0.20,
                    rank=1,
                    matched_channels=["keyword"],
                )
            ]
        },
        settings,
    )
    report = evaluate_cases(
        retrieval,
        ContextBuilder(settings),
        [
            {
                "id": "known",
                "question": "specific answer",
                "relevant_sources": ["a.md"],
                "answerable": True,
                "evidence_groups": [["different exact gold phrase"]],
            }
        ],
    )

    assert report["cases"][0]["evidence_gate"]["evidence_gate_passed"] is True
    assert report["cases"][0]["local_answer_eligible"] is False
    assert report["metrics"]["answerable_false_rejection_rate"] == 1.0


def test_real_stl_annotation_matches_pdf_extraction_words():
    cases = {case["id"]: case for case in load_cases(Path("eval/retrieval_cases_knowledge.jsonl"))}
    stl = cases["stl-overview"]
    extracted = "STL(standard template libaray-标准模板库) 数据结构与算法的软件框架"

    assert all(any(phrase.casefold() in extracted.casefold() for phrase in group) for group in stl["evidence_groups"])


def test_validate_embedding_cli_is_one_shot_and_does_not_print_credentials(monkeypatch, capsys):
    class Provider:
        provider_name = "hashing"
        model = "feature-hashing"

        def embed_texts(self, texts):
            assert len(texts) == 1
            return [[0.0] * 384]

    monkeypatch.setattr("scripts.validate_embedding.create_embedding_provider", lambda settings: Provider())
    monkeypatch.setattr(
        "scripts.validate_embedding.Settings",
        lambda: Settings(
            embedding_provider="hashing",
            embedding_dimensions=384,
            embedding_api_key="secret",
        ),
    )

    assert validate_embedding_main([]) == 0
    output = capsys.readouterr().out
    assert "secret" not in output
    assert "feature-hashing" in output


def test_ai_judge_does_not_return_provider_error_body():
    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("provider body contains api_secret=do-not-report")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    result = EvaluationJudge(
        Settings(eval_judge_api_key="configured", eval_judge_model="judge"), client=client
    ).evaluate("q", "a", "c")

    assert result.status == "error"
    assert "do-not-report" not in (result.reason or "")
    assert result.reason == "provider request failed (RuntimeError)"
