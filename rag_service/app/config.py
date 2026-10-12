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

    embedding_provider: Literal["hashing", "openai_compatible"] = Field(
        default="hashing",
        validation_alias="EMBEDDING_PROVIDER",
    )
    embedding_base_url: str | None = Field(default=None, validation_alias="EMBEDDING_BASE_URL")
    embedding_api_key: str | None = Field(default=None, validation_alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL")
    # Hashing is the zero-credential smoke provider; real providers must set
    # this to the dimension returned by their configured model.
    embedding_dimensions: int = Field(default=384, ge=8, validation_alias="EMBEDDING_DIMENSIONS")

    llm_base_url: str = Field(default="https://api.deepseek.com", validation_alias="LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY")
    llm_model: str = Field(default="deepseek-chat", validation_alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, validation_alias="LLM_TIMEOUT_SECONDS")

    chunk_max_tokens: int = Field(default=520, ge=64, validation_alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=80, ge=0, validation_alias="CHUNK_OVERLAP_TOKENS")
    chunking_version: str = Field(default="chunk-v1", validation_alias="CHUNKING_VERSION")
    parser_version: str = Field(default="parser-v1", validation_alias="PARSER_VERSION")

    retrieval_vector_candidates: int = Field(default=30, ge=1, validation_alias="RETRIEVAL_VECTOR_CANDIDATES")
    retrieval_keyword_candidates: int = Field(default=30, ge=1, validation_alias="RETRIEVAL_KEYWORD_CANDIDATES")
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
        return f"{self.parser_version}:{self.chunking_version}:max={self.chunk_max_tokens}:overlap={self.chunk_overlap_tokens}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
