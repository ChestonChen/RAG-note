from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_service.app.api import create_app
from rag_service.app.config import Settings
from rag_service.app.context import ContextBuilder
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.tuning import TuningService


def _cases(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "question": "q",
                "relevant_sources": ["a.md"],
                "answerable": True,
                "evidence_groups": [["evidence"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_tuning_rejects_unknown_and_secret_parameters() -> None:
    with pytest.raises(ValueError, match="not tunable"):
        TuningService.allowed_grid({"llm_api_key": ["secret"]})
    with pytest.raises(ValueError, match="not tunable"):
        TuningService.allowed_grid({"made_up_weight": [1.0]})


def test_tuning_selects_deterministic_best_trial_and_safe_env_lines(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'db.sqlite3'}",
        embedding_provider="hashing",
        embedding_dimensions=384,
        llm_api_key="do-not-persist",
        eval_cases_root=tmp_path,
    )

    def evaluate(params, cases, trial_settings):
        quality = float(params["retrieval_exact_rrf_weight"])
        return {
            "metrics": {
                "hit_at_1": quality / 2.0,
                "evidence_hit_at_1": quality / 2.0,
                "unknown_false_positive_rate": 0.0,
                "answerable_false_rejection_rate": 0.0,
            },
            "cases": cases,
        }

    service = TuningService(settings=settings, evaluator=evaluate)
    env_file = tmp_path / ".env"
    env_file.write_text("RETRIEVAL_EXACT_RRF_WEIGHT=original\n", encoding="utf-8")
    result = service.run(
        _cases(tmp_path / "cases.jsonl"),
        {"retrieval_exact_rrf_weight": [1.0, 1.2]},
    )

    assert result["best"]["params"] == {"retrieval_exact_rrf_weight": 1.2}
    assert result["best"]["env_lines"] == ["RETRIEVAL_EXACT_RRF_WEIGHT=1.2"]
    assert env_file.read_text(encoding="utf-8") == "RETRIEVAL_EXACT_RRF_WEIGHT=original\n"
    assert "do-not-persist" not in json.dumps(result)
    assert result["trials"][0]["score"] < result["trials"][1]["score"]


def test_tuning_rejects_oversized_grid_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="too many trials"):
        TuningService.allowed_grid({"retrieval_context_limit": list(range(65))})
    with pytest.raises(ValueError, match="invalid"):
        TuningService.allowed_grid({"retrieval_context_limit": [0]})


def test_tuning_route_is_protected_and_returns_trials_without_mutating_env(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'db.sqlite3'}",
        admin_api_key="admin-secret",
        embedding_provider="hashing",
        embedding_dimensions=384,
        eval_cases_root=tmp_path,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)

    def evaluate(params, cases, trial_settings):
        return {
            "metrics": {
                "hit_at_1": 1.0,
                "evidence_hit_at_1": 1.0,
                "unknown_false_positive_rate": 0.0,
                "answerable_false_rejection_rate": 0.0,
            },
            "cases": cases,
        }

    tuning = TuningService(settings=settings, evaluator=evaluate)
    client = TestClient(
        create_app(
            settings=settings,
            session_factory=factory,
            tuning_service=tuning,
        )
    )
    _cases(tmp_path / "cases.jsonl")
    body = {
        "cases_path": "cases.jsonl",
        "grid": {"retrieval_context_limit": [4]},
    }
    assert client.post("/v1/admin/eval/tune", json=body).status_code == 401
    response = client.post(
        "/v1/admin/eval/tune",
        headers={"X-Admin-API-Key": "admin-secret"},
        json=body,
    )
    assert response.status_code == 200
    assert response.json()["best"]["env_lines"] == ["RETRIEVAL_CONTEXT_LIMIT=4"]


def test_tuning_error_trials_cannot_be_selected_as_best(tmp_path: Path) -> None:
    settings = Settings(eval_cases_root=tmp_path)

    def evaluate(params, cases, trial_settings):
        if params["retrieval_context_limit"] == 4:
            raise RuntimeError("provider unavailable")
        return {
            "metrics": {
                "hit_at_1": 1.0,
                "evidence_hit_at_1": 1.0,
                "unknown_false_positive_rate": 0.0,
                "answerable_false_rejection_rate": 0.0,
            },
            "cases": cases,
        }

    service = TuningService(settings=settings, evaluator=evaluate)
    result = service.run(_cases(tmp_path / "cases.jsonl"), {"retrieval_context_limit": [4, 8]})

    assert result["best"]["status"] == "completed"
    assert result["best"]["params"] == {"retrieval_context_limit": 8}
    assert next(trial for trial in result["trials"] if trial["status"] == "error")["error"]


def test_tuning_returns_no_best_when_all_trials_error(tmp_path: Path) -> None:
    settings = Settings(eval_cases_root=tmp_path)

    def evaluate(params, cases, trial_settings):
        raise RuntimeError("provider unavailable")

    service = TuningService(settings=settings, evaluator=evaluate)
    result = service.run(_cases(tmp_path / "cases.jsonl"), {"retrieval_context_limit": [4, 8]})

    assert result["best"] is None
    assert result["error"] == "all tuning trials failed"
