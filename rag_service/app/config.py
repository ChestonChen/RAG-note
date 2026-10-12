from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(
        default="postgresql+psycopg://rag:rag@localhost:5432/ragdb",
        validation_alias="DATABASE_URL",
    )
    knowledge_root: Path = Field(
        default=Path("knowledge"),
        validation_alias="KNOWLEDGE_ROOT",
    )

    embedding_provider: Literal["hashing", "openai_compatible", "sentence_transformers"] = Field(
        default="hashing",
        validation_alias="EMBEDDING_PROVIDER",
    )
    embedding_base_url: str | None = Field(default=None, validation_alias="EMBEDDING_BASE_URL")
    embedding_api_key: str | None = Field(default=None, validation_alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=32, ge=1, validation_alias="EMBEDDING_BATCH_SIZE")
    embedding_device: str = Field(default="cpu", min_length=1, validation_alias="EMBEDDING_DEVICE")
    # Hashing is the zero-credential smoke provider; real providers must set
    # this to the dimension returned by their configured model.
    embedding_dimensions: int = Field(default=384, ge=8, validation_alias="EMBEDDING_DIMENSIONS")

    llm_base_url: str = Field(default="https://api.deepseek.com", validation_alias="LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY")
    llm_model: str = Field(default="deepseek-chat", validation_alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, validation_alias="LLM_TIMEOUT_SECONDS")

    web_search_enabled: bool = Field(default=False, validation_alias="WEB_SEARCH_ENABLED")
    web_search_provider: Literal["none", "tavily", "proxy_gateway"] = Field(
        default="none", validation_alias="WEB_SEARCH_PROVIDER"
    )
    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    web_search_proxy_url: str | None = Field(default=None, validation_alias="WEB_SEARCH_PROXY_URL")
    web_search_proxy_api_key: str | None = Field(default=None, validation_alias="WEB_SEARCH_PROXY_API_KEY")
    web_search_max_results: int = Field(default=5, ge=1, le=20, validation_alias="WEB_SEARCH_MAX_RESULTS")
    web_search_timeout_seconds: float = Field(default=20.0, gt=0, validation_alias="WEB_SEARCH_TIMEOUT_SECONDS")
    web_fallback_score_threshold: float = Field(
        default=0.28, ge=0, le=1, validation_alias="WEB_FALLBACK_SCORE_THRESHOLD"
    )

    eval_judge_base_url: str | None = Field(default=None, validation_alias="EVAL_JUDGE_BASE_URL")
    eval_judge_api_key: str | None = Field(default=None, validation_alias="EVAL_JUDGE_API_KEY")
    eval_judge_model: str | None = Field(default=None, validation_alias="EVAL_JUDGE_MODEL")
    eval_judge_timeout_seconds: float = Field(
        default=60.0, gt=0, validation_alias="EVAL_JUDGE_TIMEOUT_SECONDS"
    )
    admin_api_key: str | None = Field(default=None, validation_alias="ADMIN_API_KEY")

    chunk_max_tokens: int = Field(default=520, ge=64, validation_alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=80, ge=0, validation_alias="CHUNK_OVERLAP_TOKENS")
    chunking_version: str = Field(default="chunk-v1", validation_alias="CHUNKING_VERSION")
    parser_version: str = Field(default="parser-v1", validation_alias="PARSER_VERSION")

    retrieval_vector_candidates: int = Field(default=30, ge=1, validation_alias="RETRIEVAL_VECTOR_CANDIDATES")
    retrieval_keyword_candidates: int = Field(default=30, ge=1, validation_alias="RETRIEVAL_KEYWORD_CANDIDATES")
    retrieval_phrase_candidates: int = Field(default=20, ge=1, validation_alias="RETRIEVAL_PHRASE_CANDIDATES")
    retrieval_vector_rrf_weight: float = Field(default=1.0, ge=0, validation_alias="RETRIEVAL_VECTOR_RRF_WEIGHT")
    retrieval_keyword_rrf_weight: float = Field(default=1.0, ge=0, validation_alias="RETRIEVAL_KEYWORD_RRF_WEIGHT")
    retrieval_phrase_rrf_weight: float = Field(default=1.0, ge=0, validation_alias="RETRIEVAL_PHRASE_RRF_WEIGHT")
    retrieval_context_limit: int = Field(default=8, ge=1, validation_alias="RETRIEVAL_CONTEXT_LIMIT")
    retrieval_neighbor_window: int = Field(default=1, ge=0, validation_alias="RETRIEVAL_NEIGHBOR_WINDOW")
    retrieval_per_document_limit: int = Field(default=3, ge=1, validation_alias="RETRIEVAL_PER_DOCUMENT_LIMIT")
    low_confidence_threshold: float = Field(default=0.28, ge=0, le=1, validation_alias="LOW_CONFIDENCE_THRESHOLD")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolved_knowledge_root(self) -> Path:
        if self.knowledge_root.is_absolute():
            return self.knowledge_root
        return (self.project_root / self.knowledge_root).resolve()

    def embedding_signature(self) -> str:
        return f"{self.embedding_provider}:{self.embedding_model}:dim={self.embedding_dimensions}"

    def chunking_signature(self) -> str:
        # Keep the user-configured version while invalidating indexes produced
        # before section/page-aware chunk boundaries were introduced.
        implementation_version = "section-aware-v2"
        return (
            f"{self.parser_version}:{self.chunking_version}:{implementation_version}"
            f":max={self.chunk_max_tokens}:overlap={self.chunk_overlap_tokens}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
