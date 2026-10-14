"""Persistent, single-worker execution for asynchronous ingestion jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from .ingestion import IngestionService
from .models import IngestionFailure, IngestionJob


class ConflictError(RuntimeError):
    """Raised when another ingestion job is queued or running."""


class IngestionJobRunner:
    """Run one ingestion at a time while keeping job state in SQL."""

    def __init__(self, session_factory: Any, ingestion_service: IngestionService) -> None:
        self.session_factory = session_factory
        self.ingestion_service = ingestion_service
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingestion")

    def start(self) -> None:
        """Recover jobs left active by a prior process after DB startup."""
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self) -> None:
        with self.session_factory() as session:
            jobs = list(
                session.scalars(
                    select(IngestionJob).where(IngestionJob.status.in_(("queued", "running")))
                ).all()
            )
            for job in jobs:
                job.status = "failed"
                job.error_message = "worker stopped before completion"
                job.current_stage = "failed"
                job.current_file = None
                job.finished_at = datetime.now(timezone.utc)
                job.active_key = None
            session.commit()

    def create(self, force: bool, run_async: bool = True) -> IngestionJob:
        with self.session_factory() as session:
            active = session.scalar(
                select(IngestionJob.id)
                .where(IngestionJob.status.in_(("queued", "running")))
                .limit(1)
            )
            if active is not None:
                raise ConflictError("an ingestion job is already active")
            job = IngestionJob(
                status="queued",
                force=force,
                active_key=1,
                current_stage="queued",
            )
            session.add(job)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ConflictError("an ingestion job is already active") from exc
            job_id = job.id

        future = self._executor.submit(self.run, job_id, force)
        if not run_async:
            future.result()
        return self.get(job_id)

    def run(self, job_id: int, force: bool) -> None:
        with self.session_factory() as session:
            job = session.get(IngestionJob, job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            job.current_stage = "starting"
            session.commit()

        def record_progress(event: dict[str, Any]) -> None:
            summary = event.get("summary")
            with self.session_factory() as session:
                job = session.get(IngestionJob, job_id)
                if job is None:
                    return
                job.current_stage = str(event.get("stage") or "running")[:32]
                job.current_file = event.get("current_file")
                if summary is not None:
                    for name in (
                        "scanned_files",
                        "processed_files",
                        "skipped_files",
                        "failed_files",
                        "deleted_documents",
                        "chunk_count",
                    ):
                        setattr(job, name, int(getattr(summary, name, 0) or 0))
                if event.get("stage") == "failed":
                    error = event.get("error")
                    error_type = type(error).__name__ if error is not None else "IngestionError"
                    session.add(
                        IngestionFailure(
                            job_id=job_id,
                            source_path=str(event.get("current_file") or ""),
                            error_type=error_type[:128],
                            error_message=f"file ingestion failed ({error_type})",
                        )
                    )
                session.commit()

        try:
            summary = self.ingestion_service.scan(
                force=force,
                progress_callback=record_progress,
                job_id=job_id,
            )
            with self.session_factory() as session:
                job = session.get(IngestionJob, job_id)
                if job is None:
                    return
                job.scanned_files = summary.scanned_files
                job.processed_files = summary.processed_files
                job.skipped_files = summary.skipped_files
                job.failed_files = summary.failed_files
                job.deleted_documents = summary.deleted_documents
                job.chunk_count = summary.chunk_count
                job.status = "failed" if summary.failed_files else "completed"
                job.current_stage = job.status
                job.current_file = None
                job.error_message = None
                job.active_key = None
                job.started_at = job.started_at or datetime.now(timezone.utc)
                job.finished_at = datetime.now(timezone.utc)
                session.commit()
        except Exception as exc:
            with self.session_factory() as session:
                job = session.get(IngestionJob, job_id)
                if job is not None:
                    job.status = "failed"
                    job.current_stage = "failed"
                    job.current_file = None
                    job.error_message = f"ingestion failed ({type(exc).__name__})"
                    job.active_key = None
                    job.finished_at = datetime.now(timezone.utc)
                    session.commit()

    def get(self, job_id: int) -> IngestionJob:
        with self.session_factory() as session:
            job = session.get(IngestionJob, job_id)
            if job is None:
                raise KeyError(job_id)
            return job

    def list(self, limit: int = 100) -> list[IngestionJob]:
        with self.session_factory() as session:
            return list(
                session.scalars(
                    select(IngestionJob).order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc()).limit(limit)
                ).all()
            )

    def get_failures(self, job_id: int) -> list[IngestionFailure]:
        with self.session_factory() as session:
            if session.get(IngestionJob, job_id) is None:
                raise KeyError(job_id)
            return list(
                session.scalars(
                    select(IngestionFailure)
                    .where(IngestionFailure.job_id == job_id)
                    .order_by(IngestionFailure.id)
                ).all()
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
