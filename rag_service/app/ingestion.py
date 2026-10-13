"""Incremental source scanning and index construction."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import get_session_factory
from .embeddings import EmbeddingProvider, create_embedding_provider, safe_embedding_error
from .models import Chunk, Document, IngestionJob
from .search_text import build_search_text

logger = logging.getLogger(__name__)

try:  # Task 2 may be developed independently from this module.
    from .chunking import chunk_document as _chunk_document
except ImportError:  # pragma: no cover - only during partial checkout
    _chunk_document = None

try:
    from .parser import parse_file as _parse_file
except ImportError:  # pragma: no cover - only during partial checkout
    _parse_file = None


@dataclass(slots=True)
class IngestionSummary:
    scanned_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    deleted_documents: int = 0
    chunk_count: int = 0
    job_id: int | None = None


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 digest without loading an entire source file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk_size):
            digest.update(data)
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    """Backward-compatible alias used by small scripts and tests."""

    return sha256_file(path)


compute_file_hash = sha256_file


def should_reindex(
    existing: Any | None,
    file_hash: str,
    parser_signature: str,
    chunking_signature: str,
    embedding_signature: str,
) -> bool:
    """Decide whether one source needs parsing and embedding again."""

    if existing is None:
        return True
    if isinstance(existing, dict):
        get_existing = existing.get
    else:
        get_existing = lambda name, default=None: getattr(existing, name, default)
    if get_existing("status") != "ready":
        return True
    return any(
        (
            get_existing("file_hash") != file_hash,
            get_existing("parser_signature") != parser_signature,
            get_existing("chunking_signature") != chunking_signature,
            get_existing("embedding_signature") != embedding_signature,
        )
    )


def _value(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        for name in names:
            if name in item:
                return item[name]
        return default
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _heading_path(payload: Any) -> list[str]:
    value = _value(payload, "heading_path", "heading", "headings", default=[])
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(part) for part in value]


class IngestionService:
    """Synchronously scan, parse, embed, and persist a knowledge directory."""

    supported_suffixes = frozenset({".md", ".markdown", ".pdf"})

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session_factory: Callable[[], Session] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        parser: Callable[[Path], Any] | None = None,
        chunker: Callable[..., list[Any]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_factory = session_factory or get_session_factory(self.settings)
        self.embedding_provider = embedding_provider or create_embedding_provider(self.settings)
        self.parser = parser or _parse_file
        self.chunker = chunker or _chunk_document

    def _files(self, root: Path) -> list[Path]:
        return sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in self.supported_suffixes),
            key=lambda path: path.as_posix().lower(),
        )

    @staticmethod
    def _get_document(session: Session, source_path: str) -> Document | None:
        # ``Session.scalar`` is available in SQLAlchemy 1.4+; the fallback is
        # useful for simple fake sessions in unit tests.
        if hasattr(session, "scalar"):
            return session.scalar(select(Document).where(Document.source_path == source_path))
        return session.query(Document).filter(Document.source_path == source_path).first()

    @staticmethod
    def _delete_chunks(session: Session, document_id: int) -> None:
        if hasattr(session, "execute"):
            session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        else:  # pragma: no cover - only for very small fake session objects
            session.query(Chunk).filter(Chunk.document_id == document_id).delete()

    def _parse(self, path: Path) -> Any:
        if self.parser is None:
            raise RuntimeError("parser module is unavailable; implement Task 2 first")
        return self.parser(path)

    def _chunk(self, document: Any) -> list[Any]:
        if self.chunker is None:
            raise RuntimeError("chunking module is unavailable; implement Task 2 first")
        # The public contract accepts ``(document, settings)``.  A one-argument
        # fallback keeps injected test doubles pleasantly small.
        try:
            return list(self.chunker(document, self.settings))
        except TypeError as first_error:
            try:
                return list(self.chunker(document))
            except TypeError:
                raise first_error

    def _source_path(self, path: Path, root: Path) -> str:
        return path.relative_to(root).as_posix()

    def _write_file(
        self,
        session: Session,
        *,
        path: Path,
        root: Path,
        digest: str,
        parsed: Any,
        payloads: list[Any],
        vectors: list[list[float]],
    ) -> int:
        source_path = self._source_path(path, root)
        parser_signature = self.settings.parser_version
        chunking_signature = self.settings.chunking_signature()
        embedding_signature = self.settings.embedding_signature()
        title = str(_value(parsed, "title", "name", default=path.stem) or path.stem)
        source_type = str(
            _value(
                parsed,
                "source_type",
                "type",
                default="markdown" if path.suffix.lower() in {".md", ".markdown"} else path.suffix.lower().lstrip("."),
            )
            or path.suffix.lower().lstrip(".")
        )
        document = self._get_document(session, source_path)
        if document is None:
            document = Document(
                source_path=source_path,
                source_type=source_type,
                title=title,
                file_hash=digest,
                parser_signature=parser_signature,
                chunking_signature=chunking_signature,
                embedding_signature=embedding_signature,
                status="ready",
                file_mtime=path.stat().st_mtime,
            )
            session.add(document)
            session.flush()
        else:
            self._delete_chunks(session, document.id)
            document.source_type = source_type
            document.title = title
            document.file_hash = digest
            document.parser_signature = parser_signature
            document.chunking_signature = chunking_signature
            document.embedding_signature = embedding_signature
            document.status = "ready"
            document.error_message = None
            document.file_mtime = path.stat().st_mtime

        inserted_count = 0
        for index, (payload, vector) in enumerate(zip(payloads, vectors, strict=True)):
            content = str(_value(payload, "content", "text", default="") or "").strip()
            if not content:
                continue
            token_count = _value(payload, "token_count", "tokens", default=None)
            if token_count is None:
                token_count = len(content.split())
            chunk = Chunk(
                document_id=document.id,
                chunk_index=int(_value(payload, "chunk_index", "index", default=index)),
                content=content,
                heading_path=_heading_path(payload),
                search_text=build_search_text(
                    title=document.title,
                    heading_path=_heading_path(payload),
                    source_path=document.source_path,
                    content=content,
                ),
                page_from=_value(payload, "page_from", "page_start", "page", default=None),
                page_to=_value(payload, "page_to", "page_end", default=None),
                token_count=int(token_count),
                embedding=[float(value) for value in vector],
            )
            session.add(chunk)
            inserted_count += 1
        return inserted_count

    def _mark_failed(
        self,
        session: Session,
        *,
        path: Path,
        root: Path,
        digest: str,
        error: Exception,
    ) -> None:
        """Persist a failed status in a fresh transaction after rollback."""

        session.rollback()
        source_path = self._source_path(path, root)
        source_type = "markdown" if path.suffix.lower() in {".md", ".markdown"} else path.suffix.lower().lstrip(".")
        document = self._get_document(session, source_path)
        if document is None:
            document = Document(
                source_path=source_path,
                source_type=source_type,
                title=path.stem,
                file_hash=digest,
                parser_signature=self.settings.parser_version,
                chunking_signature=self.settings.chunking_signature(),
                embedding_signature=self.settings.embedding_signature(),
                status="failed",
                error_message=str(error)[:4000],
                file_mtime=path.stat().st_mtime,
            )
            session.add(document)
        else:
            # Remove old chunks so a failed reindex cannot silently leave stale
            # content eligible for retrieval.
            self._delete_chunks(session, document.id)
            document.file_hash = digest
            document.status = "failed"
            document.error_message = str(error)[:4000]
            document.file_mtime = path.stat().st_mtime
        session.commit()

    def scan(self, force: bool = False) -> IngestionSummary:
        """Scan the configured root and return per-run counters.

        Every source file is committed independently.  A malformed PDF or a
        temporary embedding failure therefore marks only that file as failed;
        the remaining files still get indexed.
        """

        root = self.settings.resolved_knowledge_root()
        summary = IngestionSummary()
        # A missing root is not an empty knowledge base.  Avoid deleting every
        # indexed document when a mount or path is temporarily unavailable.
        if not root.exists() or not root.is_dir():
            logger.warning("knowledge root does not exist: %s", root)
            return summary

        paths = self._files(root)
        summary.scanned_files = len(paths)
        current_paths = {self._source_path(path, root) for path in paths}
        session = self.session_factory()
        try:
            job = IngestionJob(status="running", scanned_files=summary.scanned_files)
            session.add(job)
            session.commit()
            summary.job_id = job.id

            parser_signature = self.settings.parser_version
            chunking_signature = self.settings.chunking_signature()
            embedding_signature = self.settings.embedding_signature()
            for path in paths:
                source_path = self._source_path(path, root)
                digest = sha256_file(path)
                existing = self._get_document(session, source_path)
                if not force and not should_reindex(
                    existing, digest, parser_signature, chunking_signature, embedding_signature
                ):
                    summary.skipped_files += 1
                    continue
                try:
                    parsed = self._parse(path)
                    payloads = self._chunk(parsed)
                    payloads = [
                        item
                        for item in payloads
                        if str(_value(item, "content", "text", default="") or "").strip()
                    ]
                    texts = []
                    for item in payloads:
                        content = str(_value(item, "content", "text", default="") or "")
                        heading_path = _heading_path(item)
                        heading_context = " > ".join(heading_path)
                        texts.append(f"{heading_context}\n\n{content}" if heading_context else content)
                    try:
                        vectors = self.embedding_provider.embed_texts(texts) if texts else []
                    except Exception as error:
                        raise safe_embedding_error(error) from None
                    if len(vectors) != len(payloads):
                        raise RuntimeError(
                            f"embedding provider returned {len(vectors)} vectors for {len(payloads)} chunks"
                        )
                    expected_dimensions = self.settings.embedding_dimensions
                    wrong_dimensions = [
                        len(vector)
                        for vector in vectors
                        if len(vector) != expected_dimensions
                    ]
                    if wrong_dimensions:
                        raise RuntimeError(
                            "embedding provider returned a vector dimension that does not match "
                            f"EMBEDDING_DIMENSIONS={expected_dimensions}"
                        )
                    chunk_count = self._write_file(
                        session,
                        path=path,
                        root=root,
                        digest=digest,
                        parsed=parsed,
                        payloads=payloads,
                        vectors=vectors,
                    )
                    session.commit()
                    summary.processed_files += 1
                    summary.chunk_count += chunk_count
                except Exception as error:  # isolate one bad source
                    logger.exception("failed to ingest %s", path)
                    self._mark_failed(session, path=path, root=root, digest=digest, error=error)
                    summary.failed_files += 1

            # Delete only sources that disappeared while the root itself exists
            # (an existing but empty directory is a valid signal that all
            # sources were removed; a missing directory returned above without
            # touching the index).
            documents = list(session.scalars(select(Document)).all())
            for document in documents:
                if document.source_path not in current_paths:
                    session.delete(document)
                    summary.deleted_documents += 1
            session.commit()

            job.status = "completed" if summary.failed_files == 0 else "completed_with_errors"
            job.processed_files = summary.processed_files
            job.skipped_files = summary.skipped_files
            job.failed_files = summary.failed_files
            job.deleted_documents = summary.deleted_documents
            job.chunk_count = summary.chunk_count
            session.commit()
            return summary
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


__all__ = [
    "IngestionService",
    "IngestionSummary",
    "file_hash",
    "compute_file_hash",
    "sha256_file",
    "should_reindex",
]
