"""FastAPI routes for ingestion, retrieval, QA, evaluation, and admin work."""

from __future__ import annotations

import hmac
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query
from sqlalchemy import select

from .config import Settings, get_settings
from .context import ContextBuilder, ContextPackage
from .db import get_engine, get_session_factory
from .evaluation import EvaluationJudge
from .generation import ChatGenerationService
from .ingestion import IngestionService
from .models import Conversation, QALog
from .retrieval import RetrievalService
from .schemas import (
    AskRequest,
    AskResponse,
    EvalJudgeRequest,
    EvalRunRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    ConversationResponse,
    MessageResponse,
)
from .web_search import WebSearchService

logger = logging.getLogger(__name__)


def serialize_message(message: Any) -> MessageResponse:
    """Serialize a persisted message using its complete response contract."""

    return MessageResponse.model_validate(message)


def serialize_conversation(conversation: Conversation) -> ConversationResponse:
    """Serialize a conversation with messages in stable chronological order."""

    messages = sorted(
        conversation.messages,
        key=lambda message: (
            message.created_at is None,
            message.created_at,
            message.id,
        ),
    )
    payload = {
        "id": conversation.id,
        "title": conversation.title,
        "archived": conversation.archived,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "messages": [serialize_message(message) for message in messages],
    }
    return ConversationResponse.model_validate(payload)


def _summary_value(summary: Any, name: str, default: int = 0) -> int:
    value = getattr(summary, name, default)
    return int(value or 0)


def create_app(
    settings: Settings | None = None,
    *,
    retrieval_service: RetrievalService | Any | None = None,
    ingestion_service: IngestionService | Any | None = None,
    generation_service: ChatGenerationService | Any | None = None,
    web_search_service: WebSearchService | Any | None = None,
    session_factory: Any | None = None,
    evaluation_judge: EvaluationJudge | Any | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    context_builder = ContextBuilder(settings)

    if session_factory is None:
        session_factory = get_session_factory(settings)
    if retrieval_service is None:
        retrieval_service = RetrievalService(session_factory, settings=settings)
    if ingestion_service is None:
        ingestion_service = IngestionService(settings=settings, session_factory=session_factory)
    if generation_service is None:
        generation_service = ChatGenerationService(settings)
    if web_search_service is None:
        web_search_service = WebSearchService(settings)
    if evaluation_judge is None:
        evaluation_judge = EvaluationJudge(settings)

    app = FastAPI(title="RAG Knowledge Base MVP", version="0.1.0")
    router = APIRouter()

    def _admin_guard(api_key: str | None) -> None:
        configured = settings.admin_api_key
        if not configured:
            raise HTTPException(status_code=503, detail="admin API is not configured")
        if not api_key or not hmac.compare_digest(api_key, configured):
            raise HTTPException(status_code=401, detail="invalid admin API key")

    def _configured_admin_guard(api_key: str | None) -> None:
        """Keep the legacy route usable in development, but protect it when configured."""

        if settings.admin_api_key:
            _admin_guard(api_key)

    def _debug_payload(question: str, hits: list[Any], context: ContextPackage) -> dict[str, Any]:
        retrieval_debug_data = getattr(retrieval_service, "latest_debug", {}) or {}
        payload = {
            "normalized_question": " ".join(question.split()),
            "requested_limit": retrieval_debug_data.get("requested_limit", settings.retrieval_context_limit),
            "primary_count": retrieval_debug_data.get(
                "primary_count", sum(1 for hit in hits if not getattr(hit, "is_neighbor", False))
            ),
            "channels": retrieval_debug_data.get("channels", {}),
            "fusion_order": retrieval_debug_data.get(
                "fusion_order", [hit.chunk_id for hit in hits if not getattr(hit, "is_neighbor", False)]
            ),
            "hits": [hit.model_dump() for hit in hits],
            "context": context.text,
        }
        if context.web_sources:
            payload["web_sources"] = [source.model_dump() for source in context.web_sources]
        return payload

    def _persist_qa_log(
        *,
        question: str,
        answer: str | None,
        confidence: str | None,
        confidence_score: float | None,
        context: ContextPackage,
        retrieval_debug: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Write an audit record without making the answer depend on logging."""

        try:
            with session_factory() as session:
                session.add(
                    QALog(
                        question=question,
                        answer=answer,
                        confidence=confidence,
                        confidence_score=confidence_score,
                        citations=[citation.model_dump() for citation in context.citations],
                        retrieval_debug=retrieval_debug,
                        latency_ms=latency_ms,
                    )
                )
                session.commit()
        except Exception:
            logger.exception("failed to persist QA log")

    def _answer_low_evidence(context: ContextPackage) -> str:
        if context.web_sources:
            return "本地知识库证据不足，联网搜索已返回资料，但当前未生成最终回答。"
        return "我无法从当前知识库中找到足够可靠的依据来回答这个问题。"

    @router.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        database = "configured"
        try:
            engine = get_engine(settings)
            with engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            database = "ready"
        except Exception:
            database = "unavailable"
        return HealthResponse(
            ok=database == "ready",
            database=database,
            knowledge_root=str(settings.resolved_knowledge_root()),
            embedding_provider=settings.embedding_provider,
            llm_configured=bool(settings.llm_api_key),
            web_search_enabled=settings.web_search_enabled,
            web_search_configured=bool(
                (settings.web_search_provider == "tavily" and settings.tavily_api_key)
                or (settings.web_search_provider == "proxy_gateway" and settings.web_search_proxy_url)
            ),
            admin_configured=bool(settings.admin_api_key),
        )

    @router.post("/v1/ingest", response_model=IngestResponse)
    def ingest(
        request: IngestRequest,
        x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
    ) -> IngestResponse:
        _configured_admin_guard(x_admin_api_key)
        try:
            summary = ingestion_service.scan(force=request.force)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc
        return IngestResponse(
            scanned_files=_summary_value(summary, "scanned_files"),
            processed_files=_summary_value(summary, "processed_files"),
            skipped_files=_summary_value(summary, "skipped_files"),
            failed_files=_summary_value(summary, "failed_files"),
            deleted_documents=_summary_value(summary, "deleted_documents"),
            chunk_count=_summary_value(summary, "chunk_count"),
        )

    @router.post("/v1/retrieval/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        hits = retrieval_service.search(request.question, request.limit)
        context = context_builder.build(request.question, hits)
        return SearchResponse(
            question=request.question,
            hits=hits,
            context=context.text,
            citations=context.citations,
        )

    @router.post("/v1/retrieval/debug")
    def retrieval_debug(request: SearchRequest) -> dict[str, Any]:
        hits = retrieval_service.search(request.question, request.limit)
        context = context_builder.build(request.question, hits)
        payload = {
            "question": request.question,
            **_debug_payload(request.question, hits, context),
            "citations": [citation.model_dump() for citation in context.citations],
            "confidence_score": context.confidence_score,
        }
        return payload

    @router.post("/v1/qa/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        hits = retrieval_service.search(request.question, settings.retrieval_context_limit)
        context = context_builder.build(request.question, hits)
        score = context.confidence_score
        used_web_search = False
        web_search_status: dict[str, Any] = {"status": "not_needed"}
        if request.allow_web_fallback and (not hits or score < settings.web_fallback_score_threshold):
            web_result = web_search_service.search(request.question)
            web_search_status = {
                "status": web_result.status,
                **({"reason": web_result.reason} if web_result.reason else {}),
                **({"error": web_result.error} if web_result.error else {}),
            }
            if web_result.sources:
                context.web_sources = web_result.sources
                used_web_search = True

        debug = _debug_payload(request.question, hits, context)
        debug["web_search"] = web_search_status
        should_generate = bool(hits and score >= settings.low_confidence_threshold) or used_web_search
        if not should_generate:
            answer = _answer_low_evidence(context)
            confidence = "low"
            _persist_qa_log(
                question=request.question,
                answer=answer,
                confidence=confidence,
                confidence_score=score,
                context=context,
                retrieval_debug=debug,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return AskResponse(
                answer=answer,
                confidence=confidence,
                confidence_score=score,
                citations=context.citations,
                retrieval_debug=debug,
                mode="kb_only",
                used_web_search=False,
                web_sources=context.web_sources,
                web_search_status=web_search_status,
            )
        try:
            answer = generation_service.generate(request.question, context)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc
        confidence = "high" if score >= 0.65 else "medium"
        mode = "kb_plus_web" if used_web_search and hits else "web_only" if used_web_search else "kb_only"
        _persist_qa_log(
            question=request.question,
            answer=answer,
            confidence=confidence,
            confidence_score=score,
            context=context,
            retrieval_debug=debug,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return AskResponse(
            answer=answer,
            confidence=confidence,
            confidence_score=score,
            citations=context.citations,
            retrieval_debug=debug,
            mode=mode,
            used_web_search=used_web_search,
            web_sources=context.web_sources,
            web_search_status=web_search_status,
        )

    @router.post("/v1/admin/ingest", response_model=IngestResponse)
    def admin_ingest(
        request: IngestRequest,
        x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
    ) -> IngestResponse:
        _admin_guard(x_admin_api_key)
        try:
            summary = ingestion_service.scan(force=request.force)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc
        return IngestResponse(
            scanned_files=_summary_value(summary, "scanned_files"),
            processed_files=_summary_value(summary, "processed_files"),
            skipped_files=_summary_value(summary, "skipped_files"),
            failed_files=_summary_value(summary, "failed_files"),
            deleted_documents=_summary_value(summary, "deleted_documents"),
            chunk_count=_summary_value(summary, "chunk_count"),
        )

    @router.get("/v1/admin/qa-logs")
    def admin_qa_logs(
        x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        _admin_guard(x_admin_api_key)
        try:
            with session_factory() as session:
                logs = list(session.scalars(select(QALog).order_by(QALog.created_at.desc()).limit(limit)).all())
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"cannot read QA logs: {exc}") from exc
        return {
            "items": [
                {
                    "id": log.id,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                    "question": log.question,
                    "answer": log.answer,
                    "confidence": log.confidence,
                    "confidence_score": log.confidence_score,
                    "citations": log.citations,
                    "retrieval_debug": log.retrieval_debug,
                    "latency_ms": log.latency_ms,
                }
                for log in logs
            ],
            "count": len(logs),
        }

    @router.post("/v1/admin/eval/judge")
    def admin_eval_judge(
        request: EvalJudgeRequest,
        x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
    ) -> dict[str, Any]:
        _admin_guard(x_admin_api_key)
        result = evaluation_judge.evaluate(request.question, request.answer, request.context)
        return {
            "status": result.status,
            "passed": result.passed,
            "score": result.score,
            "reason": result.reason,
            "raw": result.raw,
        }

    @router.post("/v1/admin/eval/run")
    def admin_eval_run(
        request: EvalRunRequest,
        x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
    ) -> dict[str, Any]:
        _admin_guard(x_admin_api_key)
        try:
            from scripts.evaluate_retrieval import evaluate_cases, load_cases

            cases_path = Path(request.cases_path or "eval/retrieval_cases.jsonl")
            if not cases_path.is_absolute():
                cases_path = (settings.project_root / cases_path).resolve()
            cases = load_cases(cases_path)
            report = evaluate_cases(
                retrieval_service,
                context_builder,
                cases,
                limit=request.limit,
                confidence_threshold=settings.low_confidence_threshold,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"evaluation cases not found: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"evaluation failed: {exc}") from exc

        if request.generate_answers or request.judge_with_ai:
            answer_reports: list[dict[str, Any]] = []
            for case in cases:
                question = str(case["question"])
                hits = retrieval_service.search(question, request.limit)
                context = context_builder.build(question, hits)
                answer = (
                    generation_service.generate(question, context)
                    if request.generate_answers or request.judge_with_ai
                    else ""
                )
                item: dict[str, Any] = {"id": case["id"], "question": question, "answer": answer}
                if request.judge_with_ai:
                    judged = evaluation_judge.evaluate(question, answer, context.text)
                    item["judge"] = {
                        "status": judged.status,
                        "passed": judged.passed,
                        "score": judged.score,
                        "reason": judged.reason,
                        "raw": judged.raw,
                    }
                answer_reports.append(item)
            report["answers"] = answer_reports
        return report

    app.include_router(router)
    return app


__all__ = ["create_app"]
