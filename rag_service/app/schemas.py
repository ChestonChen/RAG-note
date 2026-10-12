from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceCitation(BaseModel):
    source: str
    title: str
    heading: list[str] = Field(default_factory=list)
    page_from: int | None = None
    page_to: int | None = None
    score: float
    citation_id: str


class WebSource(BaseModel):
    title: str
    url: str
    snippet: str | None = None


class SearchHit(BaseModel):
    chunk_id: int
    content: str
    source: str
    title: str
    heading: list[str] = Field(default_factory=list)
    page_from: int | None = None
    page_to: int | None = None
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    fusion_score: float = 0.0
    matched_channels: list[str] = Field(default_factory=list)
    rank: int | None = None
    is_neighbor: bool = False
    matched_terms: list[str] = Field(default_factory=list)
    phrase_score: float = 0.0


class IngestRequest(BaseModel):
    force: bool = False


class IngestResponse(BaseModel):
    scanned_files: int
    processed_files: int
    skipped_files: int
    failed_files: int
    deleted_documents: int
    chunk_count: int


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=30)


class SearchResponse(BaseModel):
    question: str
    hits: list[SearchHit]
    context: str
    citations: list[SourceCitation]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    allow_web_fallback: bool = True


class AskResponse(BaseModel):
    answer: str
    confidence: str
    confidence_score: float
    citations: list[SourceCitation]
    retrieval_debug: dict[str, Any]
    mode: str = "kb_only"
    used_web_search: bool = False
    web_sources: list[WebSource] = Field(default_factory=list)
    web_search_status: dict[str, Any] = Field(default_factory=dict)


class EvalJudgeRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    context: str = ""


class EvalRunRequest(BaseModel):
    cases_path: str | None = None
    limit: int = Field(default=5, ge=1, le=30)
    generate_answers: bool = False
    judge_with_ai: bool = False


class HealthResponse(BaseModel):
    ok: bool
    database: str
    knowledge_root: str
    embedding_provider: str
    llm_configured: bool
    web_search_enabled: bool = False
    web_search_configured: bool = False
    admin_configured: bool = False


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=512)


class ConversationPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=512)
    archived: bool | None = None


class ConversationStreamRequest(BaseModel):
    question: str = Field(min_length=1)
    allow_web_fallback: bool = True


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str
    status: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_debug: dict[str, Any] = Field(default_factory=dict)
    confidence: str | None = None
    confidence_score: float | None = None
    mode: str
    web_sources: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    archived: bool
    created_at: datetime
    updated_at: datetime


class ConversationResponse(ConversationSummary):
    messages: list[MessageResponse] = Field(default_factory=list)
