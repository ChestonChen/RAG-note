from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from rag_service.app.api import create_app
from rag_service.app.config import Settings
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.models import EvalRun
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


def test_tuning_result_is_persisted_on_requested_eval_run(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'db.sqlite3'}",
        admin_api_key="admin-secret",
        embedding_provider="hashing",
        embedding_dimensions=384,
        eval_cases_root=tmp_path,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    with factory() as session:
        run = EvalRun(
            status="completed",
            total_cases=1,
            completed_cases=1,
            progress=1.0,
            profile="ours",
            dataset={"name": "cases"},
        )
        session.add(run)
        session.commit()
        run_id = run.id

    def evaluate(params, cases, trial_settings):
        return {
            "metrics": {
                "hit_at_1": float(params["retrieval_context_limit"]) / 10,
                "evidence_hit_at_1": 1.0,
                "unknown_false_positive_rate": 0.0,
                "answerable_false_rejection_rate": 0.0,
            },
            "cases": cases,
        }

    app = create_app(
        settings=settings,
        session_factory=factory,
        tuning_service=TuningService(settings=settings, evaluator=evaluate),
    )
    client = TestClient(app)
    _cases(tmp_path / "cases.jsonl")
    response = client.post(
        "/v1/admin/eval/tune",
        headers={"X-Admin-API-Key": "admin-secret"},
        json={
            "run_id": run_id,
            "cases_path": "cases.jsonl",
            "grid": {"retrieval_context_limit": [4, 8]},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["best"]["params"]["retrieval_context_limit"] == 8
    with factory() as session:
        stored = session.get(EvalRun, run_id)
        assert stored is not None
        assert stored.tuning["best"]["params"]["retrieval_context_limit"] == 8

    queried = client.get(
        f"/v1/admin/eval/runs/{run_id}/tuning",
        headers={"X-Admin-API-Key": "admin-secret"},
    )
    assert queried.status_code == 200
    assert queried.json()["trial_count"] == 2
