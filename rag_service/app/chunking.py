"""Structure-aware token-budget chunking for parsed source documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Any

from .config import Settings
from .parser import ParsedBlock, ParsedDocument

try:  # Keep parser/unit tests importable before optional dependencies are installed.
    import tiktoken
except ImportError:  # pragma: no cover - exercised only in minimal environments
    tiktoken = None  # type: ignore[assignment]


@dataclass
class ChunkPayload:
    """A chunk ready to be embedded and persisted by the ingestion service."""

    content: str
    token_count: int
    chunk_index: int
    heading_path: list[str] = field(default_factory=list)
    page_from: int | None = None
    page_to: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    block_types: list[str] = field(default_factory=list)
    source_block_indexes: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Segment:
    text: str
    tokens: list[Any]
    heading_path: list[str]
    page_number: int | None
    start_offset: int | None
    end_offset: int | None
    block_type: str
    source_block_index: int


class _FallbackTokenizer:
    """Whitespace/punctuation fallback used only when tiktoken is unavailable."""

    _pattern = re.compile(r"\s+|\S+")

    def encode(self, text: str) -> list[str]:
        return self._pattern.findall(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


@lru_cache(maxsize=1)
def _get_tokenizer() -> Any:
    if tiktoken is not None:
        return tiktoken.get_encoding("cl100k_base")
    return _FallbackTokenizer()


def chunk_document(document: ParsedDocument, settings: Settings) -> list[ChunkPayload]:
    """Chunk a parsed document while preserving structural metadata.

    Blocks are kept intact whenever they fit.  Oversized blocks are split into
    token windows, then adjacent segments are packed up to the configured
    budget.  When a new chunk starts, a bounded suffix of the previous chunk is
    carried into it as overlap.  The overlap is reduced as needed so the final
    content never exceeds ``chunk_max_tokens``.
    """

    max_tokens = int(settings.chunk_max_tokens)
    overlap_tokens = int(settings.chunk_overlap_tokens)
    if max_tokens <= 0:
        raise ValueError("chunk_max_tokens must be greater than zero")
    if overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must not be negative")
    overlap_tokens = min(overlap_tokens, max_tokens - 1)

    tokenizer = _get_tokenizer()
    segments: list[_Segment] = []
    # Leave room for the requested suffix when an oversized block is split.
    # Without this reservation, a segment exactly as large as the chunk budget
    # would force ``start_with_segment`` to drop all overlap.
    split_window = max(1, max_tokens - overlap_tokens)
    for block_index, block in enumerate(document.blocks):
        segments.extend(
            _segments_for_block(
                block,
                block_index,
                max_tokens,
                split_window,
                tokenizer,
            )
        )
    if not segments:
        return []

    chunks: list[ChunkPayload] = []
    current_parts: list[str] = []
    current_segments: list[_Segment] = []
    previous_content = ""

    def render(parts: list[str]) -> str:
        return "\n\n".join(part for part in parts if part)

    def flush() -> None:
        nonlocal current_parts, current_segments, previous_content
        if not current_parts or not current_segments:
            current_parts = []
            current_segments = []
            return
        content = render(current_parts).strip()
        if not content:
            current_parts = []
            current_segments = []
            return
        chunk = _payload_from_segments(
            content=content,
            segments=current_segments,
            chunk_index=len(chunks),
            tokenizer=tokenizer,
        )
        chunks.append(chunk)
        previous_content = content
        current_parts = []
        current_segments = []

    def start_with_segment(segment: _Segment) -> None:
        nonlocal current_parts, current_segments
        if not previous_content or overlap_tokens == 0:
            current_parts = [segment.text]
            current_segments = [segment]
            return

        overlap_units = re.findall(r"\S+(?:\s+|$)", previous_content)
        overlap_text = ""
        suffix_units: list[str] = []
        for unit in reversed(overlap_units):
            candidate_units = [unit, *suffix_units]
            candidate_overlap = "".join(candidate_units).strip()
            if len(tokenizer.encode(candidate_overlap)) > overlap_tokens:
                break
            suffix_units = candidate_units
        while suffix_units:
            candidate_overlap = "".join(suffix_units).strip()
            candidate = render([candidate_overlap, segment.text])
            if len(tokenizer.encode(candidate)) <= max_tokens:
                overlap_text = candidate_overlap
                break
            suffix_units.pop(0)
        current_parts = [part for part in (overlap_text, segment.text) if part]
        current_segments = [segment]

    for segment in segments:
        if not current_segments:
            start_with_segment(segment)
            continue

        candidate = render([*current_parts, segment.text])
        if len(tokenizer.encode(candidate)) <= max_tokens:
            current_parts.append(segment.text)
            current_segments.append(segment)
            continue

        flush()
        start_with_segment(segment)

    flush()
    return chunks


def _segments_for_block(
    block: ParsedBlock,
    block_index: int,
    max_tokens: int,
    split_window: int,
    tokenizer: Any,
) -> list[_Segment]:
    text = block.text.strip()
    if not text:
        return []
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return [_segment(block, block_index, text, tokens)]

    segments: list[_Segment] = []
    pieces = _word_window_pieces(text, split_window, tokenizer)
    for piece in pieces:
        piece_tokens = tokenizer.encode(piece)
        segments.append(_segment(block, block_index, piece, piece_tokens))
    return segments


def _word_window_pieces(text: str, window: int, tokenizer: Any) -> list[str]:
    """Split on whitespace boundaries before falling back to token windows."""

    units = re.findall(r"\S+(?:\s+|$)", text)
    if not units:
        return []
    pieces: list[str] = []
    current = ""
    for unit in units:
        stripped_unit = unit.strip()
        if len(tokenizer.encode(stripped_unit)) > window:
            if current.strip():
                pieces.append(current.strip())
                current = ""
            pieces.extend(_token_window_pieces(stripped_unit, window, tokenizer))
            continue
        candidate = current + unit
        if current and len(tokenizer.encode(candidate.strip())) > window:
            pieces.append(current.strip())
            current = unit
        else:
            current = candidate
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _token_window_pieces(text: str, window: int, tokenizer: Any) -> list[str]:
    tokens = tokenizer.encode(text)
    pieces: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + window, len(tokens))
        candidate = tokenizer.decode(tokens[start:end]).strip()
        while candidate and len(tokenizer.encode(candidate)) > window and end > start + 1:
            end -= 1
            candidate = tokenizer.decode(tokens[start:end]).strip()
        if not candidate:
            break
        pieces.append(candidate)
        start = end
    return pieces


def _segment(
    block: ParsedBlock,
    block_index: int,
    text: str,
    tokens: list[Any],
) -> _Segment:
    return _Segment(
        text=text,
        tokens=tokens,
        heading_path=list(block.heading_path),
        page_number=block.page_number,
        start_offset=block.start_offset,
        end_offset=block.end_offset,
        block_type=block.block_type,
        source_block_index=block_index,
    )


def _payload_from_segments(
    *, content: str, segments: list[_Segment], chunk_index: int, tokenizer: Any
) -> ChunkPayload:
    pages = [segment.page_number for segment in segments if segment.page_number is not None]
    starts = [segment.start_offset for segment in segments if segment.start_offset is not None]
    ends = [segment.end_offset for segment in segments if segment.end_offset is not None]
    block_types = list(dict.fromkeys(segment.block_type for segment in segments))
    block_indexes = list(dict.fromkeys(segment.source_block_index for segment in segments))
    return ChunkPayload(
        content=content,
        token_count=len(tokenizer.encode(content)),
        chunk_index=chunk_index,
        heading_path=list(segments[-1].heading_path),
        page_from=min(pages) if pages else None,
        page_to=max(pages) if pages else None,
        start_offset=min(starts) if starts else None,
        end_offset=max(ends) if ends else None,
        block_types=block_types,
        source_block_indexes=block_indexes,
    )
