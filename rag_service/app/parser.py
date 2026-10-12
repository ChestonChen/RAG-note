"""Parse source files into a small, format-independent document model.

The parser deliberately keeps the source text authoritative.  It extracts
structure and location metadata, but does not try to infer or rewrite the
meaning of a note.  Markdown is handled with a small line-oriented parser so
that code fences and heading paths remain predictable without requiring a
Markdown AST dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

try:  # pypdf is optional at import time, which keeps Markdown tooling usable.
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised only in minimal environments
    PdfReader = None  # type: ignore[assignment,misc]


@dataclass
class ParsedBlock:
    """One ordered piece of a parsed source document.

    ``start_offset`` and ``end_offset`` are character offsets for text-based
    formats.  PDF offsets are not meaningful, so PDF blocks leave them unset
    and carry their one-based page number instead.
    """

    text: str
    block_type: str = "paragraph"
    heading_path: list[str] = field(default_factory=list)
    page_number: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_from(self) -> int | None:
        return self.page_number

    @property
    def page_to(self) -> int | None:
        return self.page_number


@dataclass
class ParsedDocument:
    """Common representation consumed by the structural chunker."""

    title: str
    source_path: str
    source_type: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        """Return the source path for callers that need filesystem access."""

        return Path(self.source_path)


@dataclass(frozen=True)
class _SourceLine:
    text: str
    start: int
    end: int


_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)+\|?\s*$"
)


def parse_file(path: Path) -> ParsedDocument:
    """Parse a Markdown, text, or text-based PDF file.

    Supported Markdown suffixes are ``.md`` and ``.markdown``.  ``.txt`` is
    accepted as a plain paragraph document because it is a harmless and useful
    local extension for notes.  Image-only PDFs produce no blocks; OCR is
    intentionally outside this MVP's scope.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file():
        raise ValueError(f"source path is not a file: {source}")

    suffix = source.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return _parse_markdown(source, source_type="markdown" if suffix != ".txt" else "text")
    if suffix == ".pdf":
        return _parse_pdf(source)
    raise ValueError(f"unsupported source type: {source.suffix or '<none>'}")


def _parse_markdown(path: Path, *, source_type: str) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8-sig")
    lines = _source_lines(raw)
    blocks: list[ParsedBlock] = []
    pending: list[_SourceLine] = []
    heading_stack: list[tuple[int, str]] = []
    first_heading: str | None = None
    index = 0

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        text = _lines_to_text(pending)
        if text.strip():
            blocks.append(
                ParsedBlock(
                    text=text,
                    block_type=_classify_block(pending),
                    heading_path=[title for _, title in heading_stack],
                    start_offset=pending[0].start,
                    end_offset=pending[-1].end,
                )
            )
        pending = []

    while index < len(lines):
        line = lines[index]
        content = line.text

        fence_match = _FENCE_RE.match(content)
        if fence_match:
            flush_pending()
            marker = fence_match.group(1)
            marker_char = marker[0]
            marker_length = len(marker)
            code_lines = [line]
            index += 1
            while index < len(lines):
                code_line = lines[index]
                code_lines.append(code_line)
                closing_match = _FENCE_RE.match(code_line.text)
                if (
                    closing_match
                    and closing_match.group(1)[0] == marker_char
                    and len(closing_match.group(1)) >= marker_length
                    and not closing_match.group(2).strip()
                ):
                    index += 1
                    break
                index += 1
            blocks.append(
                ParsedBlock(
                    text=_lines_to_text(code_lines),
                    block_type="code",
                    heading_path=[title for _, title in heading_stack],
                    start_offset=code_lines[0].start,
                    end_offset=code_lines[-1].end,
                    metadata={"fence": marker_char, "fence_length": marker_length},
                )
            )
            continue

        heading_match = _HEADING_RE.match(content)
        if heading_match:
            flush_pending()
            level = len(heading_match.group(1))
            heading = _clean_heading_text(heading_match.group(2) or "")
            if not heading:
                index += 1
                continue
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))
            if first_heading is None:
                first_heading = heading
            blocks.append(
                ParsedBlock(
                    text=heading,
                    block_type="heading",
                    heading_path=[title for _, title in heading_stack],
                    start_offset=line.start,
                    end_offset=line.end,
                    metadata={"level": level},
                )
            )
            index += 1
            continue

        if not content.strip():
            flush_pending()
        else:
            pending.append(line)
        index += 1

    flush_pending()
    title = first_heading or path.stem
    return ParsedDocument(
        title=title,
        source_path=str(path),
        source_type=source_type,
        blocks=blocks,
        metadata={"suffix": path.suffix.lower(), "character_count": len(raw)},
    )


def _parse_pdf(path: Path) -> ParsedDocument:
    if PdfReader is None:
        raise RuntimeError("PDF parsing requires the 'pypdf' package")

    reader = PdfReader(str(path))
    metadata = getattr(reader, "metadata", None)
    title = _pdf_title(metadata) or path.stem
    blocks: list[ParsedBlock] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Some PDF text extractors use ASCII control separators (for example
        # ``\\x01``) between layout fragments. Treat them as line breaks so
        # they cannot leak into citations or embedding input.
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", "\n", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        blocks.append(
            ParsedBlock(
                text=text,
                block_type="page",
                heading_path=[title],
                page_number=page_number,
                metadata={"page": page_number},
            )
        )

    return ParsedDocument(
        title=title,
        source_path=str(path),
        source_type="pdf",
        blocks=blocks,
        metadata={"page_count": len(reader.pages)},
    )


def _pdf_title(metadata: Any) -> str | None:
    if metadata is None:
        return None
    value: Any = None
    if isinstance(metadata, dict):
        value = metadata.get("/Title") or metadata.get("Title")
    else:
        value = getattr(metadata, "title", None)
        if value is None:
            try:
                value = metadata.get("/Title")
            except (AttributeError, TypeError):
                value = None
    if value is None:
        return None
    title = str(value).strip()
    return title or None


def _source_lines(raw: str) -> list[_SourceLine]:
    lines: list[_SourceLine] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        lines.append(_SourceLine(text=content, start=offset, end=offset + len(line)))
        offset += len(line)
    if raw and not lines:
        lines.append(_SourceLine(text=raw, start=0, end=len(raw)))
    return lines


def _lines_to_text(lines: list[_SourceLine]) -> str:
    return "\n".join(line.text for line in lines).strip()


def _clean_heading_text(value: str) -> str:
    return re.sub(r"[ \t]+#+[ \t]*$", "", value).strip()


def _classify_block(lines: list[_SourceLine]) -> str:
    contents = [line.text for line in lines]
    if contents and all(_LIST_RE.match(value) for value in contents):
        return "list"
    if (
        len(contents) >= 2
        and all("|" in value for value in contents)
        and _TABLE_SEPARATOR_RE.match(contents[1])
    ):
        return "table"
    return "paragraph"
