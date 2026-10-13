"""FastAPI routes for ingestion, retrieval, QA, evaluation, and admin work."""

from __future__ import annotations

import hmac
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, Header, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import delete, func, select

from .config import Settings, get_settings
from .context import ContextBuilder, ContextPackage
from .db import get_engine, get_session_factory
from .evaluation import EvaluationJudge
from .generation import ChatGenerationService
from .ingestion import IngestionService
from .models import Chunk, Conversation, Document, Message, QALog
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
    ConversationCreate,
    ConversationPatch,
    ConversationStreamRequest,
    ConversationSummary,
    MessageResponse,
    KnowledgeDocumentResponse,
    KnowledgeUploadResponse,
)
from .web_search import WebSearchService

logger = logging.getLogger(__name__)

_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _is_safe_upload_basename(filename: str) -> bool:
    """Reject names that are unsafe or cannot be represented on Windows."""

    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
        or any(ord(character) < 32 for character in filename)
        or any(character in '<>:"|?*' for character in filename)
        or filename[-1] in {".", " "}
    ):
        return False
    stem = Path(filename).stem.rstrip(" .").upper()
    return stem not in _WINDOWS_RESERVED_NAMES


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
        get_debug = getattr(retrieval_service, "get_last_debug", None)
        retrieval_debug_data = get_debug() if callable(get_debug) else getattr(retrieval_service, "latest_debug", {}) or {}
        query_plan = retrieval_debug_data.get("query_plan", {}) or {}
        safe_query_plan = {
            key: query_plan[key]
            for key in ("intent", "dictionary", "entity_count", "expanded_term_count", "selected_expansion_count")
            if key in query_plan
        }
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
            "query_plan": safe_query_plan,
            "reranker": retrieval_debug_data.get("reranker", {}),
            "evidence": context.evidence.model_dump(),
            "topic_filter": retrieval_debug_data.get("topic_filter", {}),
            "timings_ms": retrieval_debug_data.get("timings_ms", {}),
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

    def _sse_event(name: str, payload: dict[str, Any]) -> str:
        return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _touch_conversation(conversation: Conversation) -> None:
        conversation.updated_at = datetime.now(timezone.utc)

    def _ingest_response(summary: Any) -> IngestResponse:
        return IngestResponse(
            scanned_files=_summary_value(summary, "scanned_files"),
            processed_files=_summary_value(summary, "processed_files"),
            skipped_files=_summary_value(summary, "skipped_files"),
            failed_files=_summary_value(summary, "failed_files"),
            deleted_documents=_summary_value(summary, "deleted_documents"),
            chunk_count=_summary_value(summary, "chunk_count"),
        )

    def _resolved_source(document: Document) -> Path:
        """Resolve an indexed source and reject paths escaping the root."""

        root = settings.resolved_knowledge_root().resolve()
        candidate = (root / document.source_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="document file not found") from exc
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="document file not found")
        return candidate

    @router.get("/v1/conversations", response_model=list[ConversationSummary])
    def list_conversations() -> list[ConversationSummary]:
        with session_factory() as session:
            conversations = list(
                session.scalars(
                    select(Conversation)
                    .where(Conversation.archived.is_(False))
                    .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                ).all()
            )
            return [ConversationSummary.model_validate(item) for item in conversations]

    @router.post("/v1/conversations", response_model=ConversationResponse)
    def create_conversation(request: ConversationCreate) -> ConversationResponse:
        with session_factory() as session:
            conversation = Conversation(title=request.title)
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return serialize_conversation(conversation)

    @router.get("/v1/conversations/{conversation_id}", response_model=ConversationResponse)
    def get_conversation(conversation_id: int) -> ConversationResponse:
        with session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="conversation not found")
            return serialize_conversation(conversation)

    @router.patch("/v1/conversations/{conversation_id}", response_model=ConversationResponse)
    def patch_conversation(conversation_id: int, request: ConversationPatch) -> ConversationResponse:
        with session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="conversation not found")
            changed = False
            if request.title is not None and request.title != conversation.title:
                conversation.title = request.title
                changed = True
            if request.archived is not None and request.archived != conversation.archived:
                conversation.archived = request.archived
                changed = True
            if changed:
                _touch_conversation(conversation)
            session.commit()
            session.refresh(conversation)
            return serialize_conversation(conversation)

    @router.delete("/v1/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: int) -> Response:
        with session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="conversation not found")
            # Explicitly delete children so SQLite test databases behave like
            # PostgreSQL even when foreign-key pragmas are not enabled.
            session.execute(delete(Message).where(Message.conversation_id == conversation_id))
            session.delete(conversation)
            session.commit()
        return Response(status_code=204)

    @router.post("/v1/conversations/{conversation_id}/stream")
    def stream_conversation(
        conversation_id: int,
        request: ConversationStreamRequest,
    ) -> StreamingResponse:
        # Validate the id before returning the lazy StreamingResponse so a
        # missing conversation remains an ordinary HTTP 404 response.
        with session_factory() as session:
            if session.get(Conversation, conversation_id) is None:
                raise HTTPException(status_code=404, detail="conversation not found")

        def events():
            user_message_id: int | None = None
            started = time.perf_counter()
            try:
                with session_factory() as session:
                    conversation = session.get(Conversation, conversation_id)
                    if conversation is None:
                        yield _sse_event("error", {"error": "conversation not found"})
                        return
                    user_message = Message(
                        conversation_id=conversation_id,
                        role="user",
                        content=request.question,
                        status="complete",
                        mode="kb_only",
                    )
                    session.add(user_message)
                    session.flush()
                    user_message_id = user_message.id
                    if conversation.title == "New conversation":
                        conversation.title = request.question.strip()[:80] or conversation.title
                    _touch_conversation(conversation)
                    session.commit()

                yield _sse_event(
                    "message_start",
                    {"user_message_id": user_message_id, "assistant_message_id": None},
                )

                hits = retrieval_service.search(request.question, settings.retrieval_context_limit)
                context = context_builder.build(request.question, hits)
                score = context.confidence_score
                used_web_search = False
                web_search_status: dict[str, Any] = {"status": "not_needed"}
                if request.allow_web_fallback and (not hits or score < settings.web_fallback_score_threshold):
                    web_result = web_search_service.search(request.question)
                    raw_status = str(getattr(web_result, "status", "error"))
                    safe_statuses = {
                        "ok",
                        "disabled",
                        "not_configured",
                        "skipped",
                        "timeout",
                        "error",
                        "unsupported",
                    }
                    safe_status = raw_status if raw_status in safe_statuses else "error"
                    web_search_status = {"status": safe_status}
                    if safe_status == "error":
                        web_search_status["reason"] = "web search unavailable"
                    sources = getattr(web_result, "sources", []) or []
                    if sources:
                        context.web_sources = sources
                        used_web_search = True

                debug = _debug_payload(request.question, hits, context)
                debug["web_search"] = web_search_status
                should_generate = bool(
                    (hits and context.evidence.eligible_citation_count and score >= settings.low_confidence_threshold)
                    or used_web_search
                )
                confidence = "high" if score >= 0.65 else "medium"
                mode = "kb_plus_web" if used_web_search and hits else "web_only" if used_web_search else "kb_only"
                if not should_generate:
                    answer = _answer_low_evidence(context)
                    confidence = "low"
                else:
                    answer_parts: list[str] = []
                    stream_method = getattr(generation_service, "stream", None)
                    if stream_method is None:
                        answer_parts.append(generation_service.generate(request.question, context))
                    else:
                        for fragment in stream_method(request.question, context):
                            text = str(fragment)
                            if text:
                                answer_parts.append(text)
                                yield _sse_event("token", {"text": text, "content": text})
                    answer = "".join(answer_parts).strip()
                    if not answer:
                        raise RuntimeError("generation returned an empty answer")

                if not should_generate:
                    yield _sse_event("token", {"text": answer, "content": answer})

                citations = [citation.model_dump() for citation in context.citations]
                web_sources = [source.model_dump() for source in context.web_sources]
                with session_factory() as session:
                    conversation = session.get(Conversation, conversation_id)
                    assistant = Message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=answer,
                        status="complete",
                        citations=citations,
                        retrieval_debug=debug,
                        confidence=confidence,
                        confidence_score=score,
                        mode=mode,
                        web_sources=web_sources,
                    )
                    session.add(assistant)
                    if conversation is not None:
                        _touch_conversation(conversation)
                    session.commit()
                    assistant_message_id = assistant.id

                _persist_qa_log(
                    question=request.question,
                    answer=answer,
                    confidence=confidence,
                    confidence_score=score,
                    context=context,
                    retrieval_debug=debug,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

                metadata = {
                    "assistant_message_id": assistant_message_id,
                    "citations": citations,
                    "confidence": confidence,
                    "confidence_score": score,
                    "retrieval_debug": debug,
                    "mode": mode,
                    "web_sources": web_sources,
                    "used_web_search": used_web_search,
                    "web_search_status": web_search_status,
                }
                yield _sse_event("metadata", metadata)
                yield _sse_event(
                    "complete",
                    {"assistant_message_id": assistant_message_id, "answer": answer, **metadata},
                )
            except Exception:
                logger.exception("conversation stream failed", extra={"conversation_id": conversation_id})
                if user_message_id is not None:
                    try:
                        with session_factory() as session:
                            user_message = session.get(Message, user_message_id)
                            if user_message is not None:
                                user_message.status = "error"
                                user_message.error_message = "回答生成失败，可重试此问题。"
                                conversation = session.get(Conversation, conversation_id)
                                if conversation is not None:
                                    _touch_conversation(conversation)
                                session.commit()
                    except Exception:
                        logger.exception("failed to persist stream error", extra={"conversation_id": conversation_id})
                yield _sse_event(
                    "error",
                    {
                        "user_message_id": user_message_id,
                        "error": "回答生成失败，可重试此问题。",
                        "recoverable": True,
                    },
                )

        return StreamingResponse(events(), media_type="text/event-stream")

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
        return _ingest_response(summary)

    @router.get("/v1/knowledge/documents", response_model=list[KnowledgeDocumentResponse])
    def list_knowledge_documents(
        x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
    ) -> list[KnowledgeDocumentResponse]:
        _configured_admin_guard(x_admin_api_key)
        with session_factory() as session:
            rows = session.execute(
                select(Document, func.count(Chunk.id).label("chunk_count"))
                .outerjoin(Chunk, Chunk.document_id == Document.id)
                .group_by(Document.id)
                .order_by(Document.updated_at.desc(), Document.id.desc())
            ).all()
            return [
                KnowledgeDocumentResponse(
                    id=document.id,
                    source_path=document.source_path,
                    title=document.title,
                    type=document.source_type,
                    status=document.status,
                    chunk_count=int(chunk_count or 0),
                    updated_at=document.updated_at,
                    error_message=document.error_message,
                )
                for document, chunk_count in rows
            ]

    @router.post("/v1/knowledge/upload", response_model=KnowledgeUploadResponse, status_code=201)
    def upload_knowledge_document(
        file: UploadFile = File(...),
        x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
    ) -> KnowledgeUploadResponse:
        _configured_admin_guard(x_admin_api_key)
        filename = file.filename or ""
        if not _is_safe_upload_basename(filename):
            raise HTTPException(status_code=400, detail="filename must be a safe basename")
        suffix = Path(filename).suffix.lower()
        if suffix not in IngestionService.supported_suffixes:
            raise HTTPException(status_code=400, detail="only Markdown and PDF files are supported")

        root = settings.resolved_knowledge_root().resolve()
        root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        size = 0
        try:
            # Keep the temporary file invisible to ingestion's supported suffix scan
            # until the atomic replacement publishes the final destination.
            with tempfile.NamedTemporaryFile(dir=root, prefix=".upload-", suffix=".uploadtmp", delete=False) as handle:
                temporary_path = Path(handle.name)
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.knowledge_upload_max_bytes:
                        raise HTTPException(status_code=413, detail="uploaded file exceeds size limit")
                    handle.write(chunk)

            stem = Path(filename).stem
            duplicate = 0
            while True:
                candidate_name = filename if duplicate == 0 else f"{stem} ({duplicate}){suffix}"
                destination = (root / candidate_name).resolve()
                try:
                    destination.relative_to(root)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail="filename must be a safe basename") from exc
                reservation_path = root / f".{candidate_name}.uploadlock"
                try:
                    reservation = os.open(
                        reservation_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                except FileExistsError:
                    duplicate += 1
                    continue
                os.close(reservation)
                try:
                    # A destination may have appeared after candidate selection
                    # but before lock creation; never replace an existing file.
                    if destination.exists():
                        duplicate += 1
                        continue
                    os.replace(temporary_path, destination)
                    temporary_path = None
                except Exception:
                    raise
                finally:
                    reservation_path.unlink(missing_ok=True)
                break
            return KnowledgeUploadResponse(
                source_path=destination.relative_to(root).as_posix(),
                filename=destination.name,
                size=size,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            file.file.close()

    @router.post("/v1/knowledge/ingest", response_model=IngestResponse)
    def ingest_knowledge(
        request: IngestRequest,
        x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
    ) -> IngestResponse:
        _configured_admin_guard(x_admin_api_key)
        try:
            summary = ingestion_service.scan(force=request.force)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc
        return _ingest_response(summary)

    @router.get("/v1/knowledge/documents/{document_id}/file")
    def serve_knowledge_document(
        document_id: int,
        x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
    ) -> FileResponse:
        _configured_admin_guard(x_admin_api_key)
        with session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="document not found")
            source = _resolved_source(document)
            return FileResponse(
                source,
                filename=source.name,
                content_disposition_type="inline",
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
            "evidence": context.evidence.model_dump(),
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
        should_generate = bool(
            (hits and context.evidence.eligible_citation_count and score >= settings.low_confidence_threshold)
            or used_web_search
        )
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
        return _ingest_response(summary)

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
