"""Embedding validation and explicit rebuilds of derived knowledge indexes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import drop_derived_tables, get_engine, get_session_factory, init_db
from .embeddings import EmbeddingProvider, create_embedding_provider
from .ingestion import IngestionService


@dataclass(frozen=True, slots=True)
class EmbeddingValidationReport:
    """Safe, serializable result of a one-vector provider probe."""

    provider_name: str
    model_name: str
    configured_dimension: int
    observed_dimension: int | None
    ok: bool
    error: str | None = None

    @property
    def provider(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return self.model_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()


def _provider_name(provider: EmbeddingProvider) -> str:
    configured = getattr(provider, "provider_name", None)
    if configured:
        return str(configured)
    name = provider.__class__.__name__.lower()
    if "openai" in name:
        return "openai_compatible"
    if "sentence" in name or "transformer" in name:
        return "sentence_transformers"
    if "hash" in name:
        return "hashing"
    return provider.__class__.__name__


def _model_name(provider: EmbeddingProvider) -> str:
    model = getattr(provider, "model", None) or getattr(provider, "model_path", None)
    return str(model) if model else ""


def validate_embedding_provider(
    provider: EmbeddingProvider,
    expected_dimensions: int,
    sample_text: str = "embedding dimension validation",
) -> EmbeddingValidationReport:
    """Probe one sample and report actual dimension without provider secrets."""

    if expected_dimensions < 1:
        raise ValueError("expected embedding dimensions must be positive")
    observed: int | None = None
    error: str | None = None
    ok = False
    try:
        vectors = provider.embed_texts([sample_text])
        if len(vectors) != 1:
            error = "provider returned an unexpected number of vectors"
        else:
            vector = vectors[0]
            if not isinstance(vector, (list, tuple)):
                error = "provider returned an invalid vector"
            else:
                observed = len(vector)
                ok = observed == expected_dimensions
                if not ok:
                    error = "observed vector dimension does not match configured dimension"
    except Exception as exc:  # pragma: no cover - provider errors vary
        error = f"{exc.__class__.__name__} while probing embedding provider"
    return EmbeddingValidationReport(
        provider_name=_provider_name(provider),
        model_name=_model_name(provider),
        configured_dimension=expected_dimensions,
        observed_dimension=observed,
        ok=ok,
        error=error,
    )


def _factory_engine(settings: Settings, factory: Callable[[], Session] | Any):
    session = None
    try:
        if callable(factory):
            session = factory()
            get_bind = getattr(session, "get_bind", None)
            if callable(get_bind):
                try:
                    bind = get_bind()
                except Exception:
                    bind = None
                if bind is not None:
                    return bind
            bind = getattr(session, "bind", None)
            if bind is not None:
                return bind
        bind = getattr(factory, "bind", None)
        if bind is None:
            bind = getattr(factory, "kw", {}).get("bind")
        if bind is not None:
            return bind
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()
    raise ValueError(
        "session_factory does not expose its actual database bind; refusing destructive rebuild"
    )


def _drop_derived_objects(settings: Settings, engine: Any | None = None) -> list[str]:
    """Drop rebuildable database tables; a seam for controlled tests."""

    return drop_derived_tables(engine or get_engine(settings))


def _summary_mapping(summary: Any) -> dict[str, Any]:
    names = (
        "scanned_files",
        "processed_files",
        "skipped_files",
        "failed_files",
        "deleted_documents",
        "chunk_count",
        "job_id",
    )
    return {name: getattr(summary, name, None) for name in names}


def rebuild_derived_index(
    settings: Settings | None = None,
    *,
    confirm: bool,
    session_factory: Callable[[], Session] | None = None,
) -> dict[str, Any]:
    """Validate embeddings, recreate derived tables, then force a source scan."""

    settings = settings or get_settings()
    if not confirm:
        raise PermissionError(
            "rebuilding the derived index is destructive; pass confirm=True to continue"
        )

    provider = create_embedding_provider(settings)
    validation = validate_embedding_provider(provider, settings.embedding_dimensions)
    source_root = str(settings.resolved_knowledge_root())
    result: dict[str, Any] = {
        "status": "validation_failed" if not validation.ok else "pending",
        "source_root": source_root,
        "embedding_signature": settings.embedding_signature(),
        "active_embedding_signature": None,
        "validation": validation.to_dict(),
        "dropped_tables": [],
        "ingestion": None,
        "recoverability": (
            "source files are preserved; rerun this confirmed command to rebuild derived data"
        ),
    }
    if not validation.ok:
        return result

    factory = session_factory or get_session_factory(settings)
    engine = _factory_engine(settings, factory)
    result["dropped_tables"] = _drop_derived_objects(settings, engine)
    init_db(engine=engine, settings=settings)
    summary = IngestionService(
        settings=settings,
        session_factory=factory,
        embedding_provider=provider,
    ).scan(force=True)
    result["ingestion"] = _summary_mapping(summary)
    scanned = int(getattr(summary, "scanned_files", 0) or 0)
    processed = int(getattr(summary, "processed_files", 0) or 0)
    skipped = int(getattr(summary, "skipped_files", 0) or 0)
    failed = int(getattr(summary, "failed_files", 0) or 0)
    source_root_path = settings.resolved_knowledge_root()
    scan_complete = source_root_path.is_dir() and processed + skipped + failed == scanned
    if scan_complete and failed == 0:
        result["status"] = "completed"
        result["active_embedding_signature"] = settings.embedding_signature()
        result["recoverability"] = None
    else:
        result["status"] = "completed_with_errors" if failed else "incomplete"
        result["active_embedding_signature"] = None
    return result


__all__ = [
    "EmbeddingValidationReport",
    "rebuild_derived_index",
    "validate_embedding_provider",
]
