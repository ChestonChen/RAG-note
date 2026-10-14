from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_service.app.api import create_app
from rag_service.app.config import Settings
from rag_service.app.context import ContextBuilder
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.evaluation_runner import EvaluationRunner
from rag_service.app.models import EvalResult, EvalRun
from rag_service.app.schemas import EvalRunRequest, SearchHit
from rag_service.app.job_runner import ConflictError


class FakeRetrieval:
    def __init__(self, settings: Settings, responses: dict[str, list[SearchHit]]) -> None:
        self.settings = settings
        self.responses = responses

    def search(self, question: str, limit: int | None = None) -> list[SearchHit]:
        if question == "boom":
            raise RuntimeError("provider secret should not persist")
        hits = self.responses.get(question, [])
        return hits[:limit] if limit is not None else hits

    def get_last_debug(self) -> dict[str, object]:
        return {}


def _hit(source: str = "a.md") -> SearchHit:
    return SearchHit(
        chunk_id=1,
        content="answer evidence",
        source=source,
        title=source,
        heading=[],
        score=0.9,
        rank=1,
        matched_channels=["keyword"],
    )


def _fixture(tmp_path: Path, cases: list[dict[str, object]]):
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        admin_api_key="test-admin",
        eval_judge_api_key="private-secret-value",
        low_confidence_threshold=0.0,
    )
    engine = init_db(get_engine(settings))
    factory = get_session_factory(engine=engine)
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")
    retrieval = FakeRetrieval(settings, {"known": [_hit()]})
    runner = EvaluationRunner(factory, retrieval, ContextBuilder(settings), settings=settings)
    return settings, factory, path, runner


def test_eval_run_persists_case_results_and_metrics(tmp_path: Path):
    settings, factory, path, runner = _fixture(
        tmp_path,
        [
            {
                "id": "case-1",
                "question": "known",
                "relevant_sources": ["a.md"],
                "answerable": True,
                "evidence_groups": [["answer evidence"]],
            }
        ],
    )
    run = runner.create(EvalRunRequest(cases_path=str(path), run_async=False), run_async=False)
    stored = runner.get(run.id)

    assert stored.status == "completed"
    assert stored.progress == 1.0
    assert stored.metrics["hit_at_1"] == 1.0
    assert stored.started_at is not None and stored.finished_at is not None
    assert "private-secret-value" not in json.dumps(stored.config)
    assert runner.results(run.id)[0].case_id == "case-1"
    with factory() as session:
        assert session.get(EvalRun, run.id) is not None
        assert session.query(EvalResult).count() == 1
    runner.shutdown()


def test_eval_run_continues_after_case_error_and_records_safe_error(tmp_path: Path):
    settings, _, path, runner = _fixture(
        tmp_path,
        [
            {"id": "bad", "question": "boom", "relevant_sources": [], "answerable": False},
            {"id": "good", "question": "known", "relevant_sources": ["a.md"], "answerable": True, "evidence_groups": [["answer evidence"]]},
        ],
    )
    run = runner.create(EvalRunRequest(cases_path=str(path), run_async=False), run_async=False)
    rows = runner.results(run.id)

    assert run.status == "failed"
    assert run.completed_cases == 2
    assert run.failed_cases == 1
    assert rows[0].status == "error"
    assert "provider secret" not in (rows[0].error_message or "")
    assert rows[1].status == "completed"
    runner.shutdown()


def test_eval_run_active_conflict_and_protected_routes(tmp_path: Path):
    settings, _, path, runner = _fixture(
        tmp_path,
        [{"id": "case-1", "question": "known", "relevant_sources": ["a.md"], "answerable": True, "evidence_groups": [["answer evidence"]]}],
    )
    original_run = runner.run
    entered = False

    def blocking(run_id, request):
        nonlocal entered
        entered = True
        time.sleep(0.15)
        return original_run(run_id, request)

    runner.run = blocking
    first = runner.create(EvalRunRequest(cases_path=str(path)), run_async=True)
    deadline = time.monotonic() + 1
    while not entered and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(ConflictError):
        runner.create(EvalRunRequest(cases_path=str(path)), run_async=True)

    client = TestClient(create_app(settings=settings, session_factory=runner.session_factory, retrieval_service=runner.retrieval_service, evaluation_runner=runner))
    assert client.get("/v1/admin/eval/runs").status_code == 401
    response = client.get("/v1/admin/eval/runs", headers={"X-Admin-API-Key": "test-admin"})
    assert response.status_code == 200
    assert response.json()["items"][0]["id"] == first.id
    headers = {"X-Admin-API-Key": "test-admin"}
    assert client.get("/v1/admin/eval/runs/latest", headers=headers).json()["id"] == first.id
    assert client.get(f"/v1/admin/eval/runs/{first.id}", headers=headers).json()["status"] in {"queued", "running", "completed"}
    assert client.get(f"/v1/admin/eval/runs/{first.id}/results", headers={"X-Admin-API-Key": "test-admin"}).status_code == 200
    runner.shutdown()


def test_sync_eval_route_keeps_report_shape_and_run_async_switch(tmp_path: Path):
    settings, factory, path, runner = _fixture(
        tmp_path,
        [{"id": "case-1", "question": "known", "relevant_sources": ["a.md"], "answerable": True, "evidence_groups": [["answer evidence"]]}],
    )
    client = TestClient(create_app(settings=settings, session_factory=factory, retrieval_service=runner.retrieval_service, evaluation_runner=runner))
    headers = {"X-Admin-API-Key": "test-admin"}
    response = client.post(
        "/v1/admin/eval/run",
        headers=headers,
        json={"cases_path": str(path), "run_async": False},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["metrics"]["hit_at_1"] == 1.0
    assert response.json()["cases"][0]["id"] == "case-1"
    assert client.post("/v1/admin/eval/run", headers=headers, json={"cases_path": "missing.jsonl"}).status_code == 404
    runner.shutdown()
