from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    source: str
    title: str
    heading: list[str] = Field(default_factory=list)
    page_from: int | None = None
    page_to: int | None = None
    score: float
    citation_id: str


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


class AskResponse(BaseModel):
    answer: str
    confidence: str
    confidence_score: float
    citations: list[SourceCitation]
    retrieval_debug: dict[str, Any]


class HealthResponse(BaseModel):
    ok: bool
    database: str
    knowledge_root: str
    embedding_provider: str
    llm_configured: bool
