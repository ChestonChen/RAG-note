"""Persistent evaluation run orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .config import Settings
from .context import ContextBuilder
from .evaluation import EvaluationJudge
from .job_runner import ConflictError
from .models import EvalResult, EvalRun
from .schemas import EvalRunRequest


def resolve_eval_cases_path(
    settings: Settings,
    value: str | None,
    *,
    reject_absolute: bool = False,
) -> Path:
    """Resolve a case file beneath the configured evaluation root."""

    raw = Path(value or "retrieval_cases.jsonl")
    if reject_absolute and (raw.is_absolute() or bool(raw.root) or bool(raw.drive)):
        raise ValueError("cases_path must be relative to the configured evaluation cases root")

    root = settings.resolved_eval_cases_root()
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        root_relative = root.relative_to(settings.project_root) if root.is_relative_to(settings.project_root) else None
        relative = raw
        if root_relative and raw.parts[: len(root_relative.parts)] == root_relative.parts:
            relative = Path(*raw.parts[len(root_relative.parts) :])
        candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("cases_path must stay under the configured evaluation cases root")
    return candidate


def _safe_case_path(settings: Settings, value: str | None) -> Path:
    return resolve_eval_cases_path(settings, value)


def _safe_error(exc: Exception) -> str:
    return f"evaluation failed ({type(exc).__name__})"


class EvaluationRunner:
    """Run evaluations serially while persisting every case outcome."""

    def __init__(
        self,
        session_factory: Any,
        retrieval_service: Any,
        context_builder: ContextBuilder,
        *,
        settings: Settings,
        evaluation_judge: EvaluationJudge | Any | None = None,
        answer_generator: Any | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.settings = settings
        self.evaluation_judge = evaluation_judge
        self.answer_generator = answer_generator
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evaluation")
        self._case_paths: dict[int, Path] = {}

    def start(self) -> None:
        with self.session_factory() as session:
            runs = list(session.scalars(select(EvalRun).where(EvalRun.status.in_(("queued", "running")))).all())
            for run in runs:
                run.status = "failed"
                run.error_message = "worker stopped before completion"
                run.active_key = None
                run.finished_at = datetime.now(timezone.utc)
            session.commit()

    def create(self, request: EvalRunRequest, run_async: bool = True) -> EvalRun:
        from scripts.evaluate_retrieval import load_cases

        cases_path = _safe_case_path(self.settings, request.cases_path)
        cases = load_cases(cases_path)
        config = {
            "cases_path": cases_path.name,
            "limit": request.limit,
            "profile": request.profile,
            "generate_answers": request.generate_answers,
            "judge_with_ai": request.judge_with_ai,
            "retrieval": {
                name: getattr(self.settings, name)
                for name in (
                    "embedding_provider",
                    "embedding_model",
                    "embedding_dimensions",
                    "retrieval_context_limit",
                    "retrieval_query_expansion_enabled",
                    "reranker_enabled",
                    "reranker_model",
                    "low_confidence_threshold",
                )
            },
        }
        with self.session_factory() as session:
            active = session.scalar(
                select(EvalRun.id).where(EvalRun.status.in_(("queued", "running"))).limit(1)
            )
            if active is not None:
                raise ConflictError("an evaluation run is already active")
            run = EvalRun(
                status="queued",
                active_key=1,
                total_cases=len(cases),
                config=config,
                profile=request.profile,
                dataset={"name": cases_path.stem},
            )
            session.add(run)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ConflictError("an evaluation run is already active") from exc
            run_id = run.id
            self._case_paths[run_id] = cases_path
        try:
            future = self._executor.submit(self.run, run_id, request)
        except Exception as exc:
            self._finish_failed(run_id, _safe_error(exc))
            return self.get(run_id)
        if not run_async:
            try:
                future.result()
            except Exception as exc:
                self._finish_failed(run_id, _safe_error(exc))
        return self.get(run_id)

    def run(self, run_id: int, request: EvalRunRequest) -> None:
        try:
            self._run(run_id, request)
        except Exception as exc:
            self._finish_failed(run_id, _safe_error(exc))

    def _run(self, run_id: int, request: EvalRunRequest) -> None:
        from scripts.evaluate_retrieval import evaluate_cases, load_cases

        with self.session_factory() as session:
            run = session.get(EvalRun, run_id)
            if run is None:
                return
            run.status = "running"
            run.started_at = datetime.now(timezone.utc)
            session.commit()
            cases_path = self._case_paths.get(run_id) or _safe_case_path(self.settings, run.config.get("cases_path"))

        try:
            cases = load_cases(cases_path)
        except Exception as exc:
            self._finish_failed(run_id, _safe_error(exc))
            return

        successful_cases: list[dict[str, Any]] = []
        case_reports: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            status = "completed"
            result: dict[str, Any] = {}
            error_message: str | None = None
            try:
                report = evaluate_cases(
                    self.retrieval_service,
                    self.context_builder,
                    [case],
                    limit=request.limit,
                    confidence_threshold=self.settings.low_confidence_threshold,
                    profile=request.profile,
                    dataset={"name": cases_path.stem},
                )
                result = report["cases"][0]
                case_reports.append(result)
                successful_cases.append(case)
                if request.generate_answers or request.judge_with_ai:
                    question = str(case["question"])
                    hits = self.retrieval_service.search(question, request.limit)
                    context = self.context_builder.build(question, hits)
                    answer = self.answer_generator(question, context) if self.answer_generator else ""
                    result["answer"] = answer
                    if request.judge_with_ai and self.evaluation_judge is not None:
                        judged = self.evaluation_judge.evaluate(question, answer, context.text)
                        result["judge"] = {
                            "status": judged.status,
                            "passed": judged.passed,
                            "score": judged.score,
                            "reason": judged.reason,
                            "raw": judged.raw,
                        }
            except Exception as exc:
                status = "error"
                error_message = _safe_error(exc)
            with self.session_factory() as session:
                run = session.get(EvalRun, run_id)
                if run is None:
                    return
                session.add(
                    EvalResult(
                        run_id=run_id,
                        case_index=index,
                        case_id=str(case.get("id", index)),
                        status=status,
                        result=result,
                        error_message=error_message,
                    )
                )
                run.completed_cases = index + 1
                run.failed_cases = sum(1 for row in session.scalars(select(EvalResult).where(EvalResult.run_id == run_id)).all() if row.status == "error")
                run.progress = run.completed_cases / run.total_cases if run.total_cases else 1.0
                session.commit()

        metrics: dict[str, Any] = {}
        if successful_cases:
            try:
                metrics = evaluate_cases(
                    self.retrieval_service,
                    self.context_builder,
                    successful_cases,
                    limit=request.limit,
                    confidence_threshold=self.settings.low_confidence_threshold,
                    profile=request.profile,
                    dataset={"name": cases_path.stem},
                ).get("metrics", {})
            except Exception:
                metrics = {}
        with self.session_factory() as session:
            run = session.get(EvalRun, run_id)
            if run is None:
                return
            run.metrics = metrics
            run.status = "failed" if run.failed_cases else "completed"
            run.error_message = "one or more cases failed" if run.failed_cases else None
            run.active_key = None
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
        self._case_paths.pop(run_id, None)

    def _finish_failed(self, run_id: int, error: str) -> None:
        try:
            with self.session_factory() as session:
                run = session.get(EvalRun, run_id)
                if run is not None:
                    run.status = "failed"
                    run.error_message = error
                    run.active_key = None
                    run.finished_at = datetime.now(timezone.utc)
                    session.commit()
        finally:
            self._case_paths.pop(run_id, None)

    def get(self, run_id: int) -> EvalRun:
        with self.session_factory() as session:
            run = session.get(EvalRun, run_id)
            if run is None:
                raise KeyError(run_id)
            return run

    def list(self, limit: int = 100) -> list[EvalRun]:
        with self.session_factory() as session:
            return list(session.scalars(select(EvalRun).order_by(EvalRun.created_at.desc(), EvalRun.id.desc()).limit(limit)).all())

    def latest(self) -> EvalRun:
        with self.session_factory() as session:
            run = session.scalar(select(EvalRun).order_by(EvalRun.created_at.desc(), EvalRun.id.desc()).limit(1))
            if run is None:
                raise KeyError("latest")
            return run

    def results(self, run_id: int) -> list[EvalResult]:
        with self.session_factory() as session:
            if session.get(EvalRun, run_id) is None:
                raise KeyError(run_id)
            return list(session.scalars(select(EvalResult).where(EvalResult.run_id == run_id).order_by(EvalResult.case_index)).all())

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


__all__ = ["EvaluationRunner", "resolve_eval_cases_path"]
