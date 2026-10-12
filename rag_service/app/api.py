"""FastAPI routes for ingestion, retrieval debugging, and QA."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from .config import Settings, get_settings
from .context import ContextBuilder
from .db import get_engine, get_session_factory
from .generation import ChatGenerationService
from .ingestion import IngestionService
from .retrieval import RetrievalService
from .schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)


def _summary_value(summary: Any, name: str, default: int = 0) -> int:
    value = getattr(summary, name, default)
    return int(value or 0)


def create_app(
    settings: Settings | None = None,
    *,
    retrieval_service: RetrievalService | Any | None = None,
    ingestion_service: IngestionService | Any | None = None,
    generation_service: ChatGenerationService | Any | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    context_builder = ContextBuilder(settings)

    if retrieval_service is None:
        session_factory = get_session_factory(settings)
        retrieval_service = RetrievalService(session_factory, settings=settings)
    if ingestion_service is None:
        ingestion_service = IngestionService(settings=settings)
    if generation_service is None:
        generation_service = ChatGenerationService(settings)

    app = FastAPI(title="RAG Knowledge Base MVP", version="0.1.0")
    router = APIRouter()

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
        )

    @router.post("/v1/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
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
        retrieval_debug_data = getattr(retrieval_service, "latest_debug", {}) or {}
        return {
            "question": request.question,
            "normalized_question": " ".join(request.question.split()),
            "requested_limit": retrieval_debug_data.get(
                "requested_limit", request.limit or settings.retrieval_context_limit
            ),
            "primary_count": retrieval_debug_data.get(
                "primary_count", sum(1 for hit in hits if not hit.is_neighbor)
            ),
            "channels": retrieval_debug_data.get("channels", {}),
            "fusion_order": retrieval_debug_data.get(
                "fusion_order", [hit.chunk_id for hit in hits if not hit.is_neighbor]
            ),
            "hits": [hit.model_dump() for hit in hits],
            "context": context.text,
            "citations": [citation.model_dump() for citation in context.citations],
            "confidence_score": context.confidence_score,
        }

    @router.post("/v1/qa/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        hits = retrieval_service.search(request.question, settings.retrieval_context_limit)
        context = context_builder.build(request.question, hits)
        score = context.confidence_score
        debug = {
            "normalized_question": " ".join(request.question.split()),
            "hits": [hit.model_dump() for hit in hits],
            "context": context.text,
        }
        if not hits or score < settings.low_confidence_threshold:
            return AskResponse(
                answer="我无法从当前知识库中找到足够可靠的依据来回答这个问题。",
                confidence="low",
                confidence_score=score,
                citations=context.citations,
                retrieval_debug=debug,
            )
        try:
            answer = generation_service.generate(request.question, context)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc
        confidence = "high" if score >= 0.65 else "medium"
        return AskResponse(
            answer=answer,
            confidence=confidence,
            confidence_score=score,
            citations=context.citations,
            retrieval_debug=debug,
        )

    app.include_router(router)
    return app


__all__ = ["create_app"]
