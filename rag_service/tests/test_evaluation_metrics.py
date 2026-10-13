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


def test_evaluation_report_records_signature_config_buckets_channels_and_citation_precision():
    settings = Settings(low_confidence_threshold=0.28)
    retrieval = FakeRetrieval(
        {
            "known": [
                hit(1, "a.md", 0.9, channels=["vector", "keyword"], rank=1),
                hit(2, "d.md", 0.5, channels=["keyword"], rank=2),
            ],
            "unknown": [hit(3, "d.md", 0.5, channels=["keyword"], rank=1)],
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
    assert report["citation_source_precision"] == 0.5
    assert "keyword" in report["per_channel_contribution"]
    assert report["metrics"]["unknown_false_positive_rate"] >= 0
    assert report["cases"][0]["confidence_bucket"] in {"low", "medium", "high"}


def test_unknown_false_positive_marks_high_confidence_unknown_case():
    settings = Settings()
    retrieval = FakeRetrieval({"unknown": [hit(1, "d.md", 1.0, channels=["keyword"], rank=1)]}, settings)
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
    cases = load_cases(Path("eval/retrieval_cases.jsonl"))
    by_id = {case["id"]: case for case in cases}
    assert by_id["mysql-basics"]["relevant_sources"] == ["04_mysql_notes.md"]
    assert by_id["redis-persistence"]["relevant_sources"] == ["05_redis_notes.md"]
    assert by_id["mysql-redis-cross-topic"]["answerable"] is False


def test_validate_embedding_cli_is_one_shot_and_does_not_print_credentials(monkeypatch, capsys):
    class Provider:
        provider_name = "hashing"
        model = "feature-hashing"

        def embed_texts(self, texts):
            assert len(texts) == 1
            return [[0.0] * 384]

    monkeypatch.setattr("scripts.validate_embedding.create_embedding_provider", lambda settings: Provider())
    monkeypatch.setattr("scripts.validate_embedding.Settings", lambda: Settings(embedding_api_key="secret"))

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
