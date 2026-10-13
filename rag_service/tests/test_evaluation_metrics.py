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
        + "\n",
        encoding="utf-8",
    )

    assert load_cases(path)[0]["tags"] == ["paraphrase", "multi-source"]
    assert load_cases(path)[0]["expected_channels"] == ["vector", "keyword"]
    assert load_cases(path)[0]["paraphrase_of"] == "base"


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
