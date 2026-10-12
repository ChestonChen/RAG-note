from pathlib import Path

import pytest

from rag_service.app.config import Settings
from rag_service.app import parser as parser_module
from rag_service.app.chunking import chunk_document
from rag_service.app.parser import ParsedBlock, ParsedDocument, parse_file


def test_markdown_parser_preserves_headings_and_fenced_code(tmp_path: Path):
    source = tmp_path / "rag-notes.md"
    source.write_text(
        "# RAG Notes\n\n"
        "Introductory paragraph.\n\n"
        "## Ingestion\n\n"
        "```python\n"
        "def ingest(path):\n"
        "    return path\n"
        "```\n\n"
        "- Parse the file\n"
        "- Build chunks\n",
        encoding="utf-8",
    )

    document = parse_file(source)

    assert document.title == "RAG Notes"
    assert document.source_type == "markdown"
    assert [block.block_type for block in document.blocks] == [
        "heading",
        "paragraph",
        "heading",
        "code",
        "list",
    ]
    assert document.blocks[3].heading_path == ["RAG Notes", "Ingestion"]
    assert "def ingest(path):" in document.blocks[3].text
    assert document.blocks[3].text.startswith("```python")
    assert document.blocks[4].heading_path == ["RAG Notes", "Ingestion"]
    assert document.blocks[4].start_offset < document.blocks[4].end_offset


def test_markdown_parser_tracks_nested_heading_paths_and_tables(tmp_path: Path):
    source = tmp_path / "nested.md"
    source.write_text(
        "# Top\n\n"
        "## First\n\n"
        "### Detail\n\n"
        "| name | value |\n"
        "| --- | --- |\n"
        "| k | v |\n\n"
        "## Second\n\n"
        "Text\n",
        encoding="utf-8",
    )

    document = parse_file(source)

    table = next(block for block in document.blocks if block.block_type == "table")
    assert table.heading_path == ["Top", "First", "Detail"]
    assert table.text.count("|") >= 6
    second_text = next(block for block in document.blocks if block.text == "Text")
    assert second_text.heading_path == ["Top", "Second"]


def test_pdf_parser_keeps_page_metadata(monkeypatch, tmp_path: Path):
    class FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        metadata = {"/Title": "PDF title"}
        pages = [FakePage("Page one content"), FakePage("  \n"), FakePage("Page three")]

    monkeypatch.setattr(parser_module, "PdfReader", lambda _: FakeReader())
    source = tmp_path / "reference.pdf"
    source.write_bytes(b"fixture")

    document = parse_file(source)

    assert document.title == "PDF title"
    assert document.source_type == "pdf"
    assert len(document.blocks) == 2
    assert document.blocks[0].page_number == 1
    assert document.blocks[1].page_number == 3
    assert document.blocks[0].text == "Page one content"


def test_chunking_preserves_heading_and_page_range():
    document = ParsedDocument(
        title="Guide",
        source_path="guide.md",
        source_type="markdown",
        blocks=[
            ParsedBlock(
                text="A short introduction.",
                block_type="paragraph",
                heading_path=["Guide", "Intro"],
                start_offset=0,
                end_offset=22,
            ),
            ParsedBlock(
                text="Details from page two.",
                block_type="paragraph",
                heading_path=["Guide", "Details"],
                page_number=2,
                start_offset=23,
                end_offset=47,
            ),
        ],
    )

    settings = Settings(chunk_max_tokens=100, chunk_overlap_tokens=0)
    chunks = chunk_document(document, settings)

    assert len(chunks) == 2
    assert chunks[0].heading_path == ["Guide", "Intro"]
    assert chunks[0].page_from is None
    assert chunks[0].page_to is None
    assert chunks[1].heading_path == ["Guide", "Details"]
    assert chunks[1].page_from == 2
    assert chunks[1].page_to == 2
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert all(chunk.token_count > 0 for chunk in chunks)
    assert "introduction" in chunks[0].content


def test_chunking_splits_oversized_block_and_applies_bounded_overlap():
    words = " ".join(f"token-{index}" for index in range(180))
    document = ParsedDocument(
        title="Long",
        source_path="long.md",
        source_type="markdown",
        blocks=[
            ParsedBlock(
                text=words,
                block_type="paragraph",
                heading_path=["Long"],
                start_offset=0,
                end_offset=len(words),
            )
        ],
    )
    settings = Settings(chunk_max_tokens=96, chunk_overlap_tokens=16)

    chunks = chunk_document(document, settings)

    assert len(chunks) > 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(0 < chunk.token_count <= settings.chunk_max_tokens for chunk in chunks)
    assert all(chunk.heading_path == ["Long"] for chunk in chunks)
    assert any(
        set(chunks[index - 1].content.split()) & set(chunks[index].content.split())
        for index in range(1, len(chunks))
    )


def test_chunking_rejects_invalid_budget():
    document = ParsedDocument(
        title="Empty",
        source_path="empty.md",
        source_type="markdown",
        blocks=[ParsedBlock(text="content")],
    )

    class InvalidSettings:
        chunk_max_tokens = 0
        chunk_overlap_tokens = 0

    with pytest.raises(ValueError, match="chunk_max_tokens"):
        chunk_document(document, InvalidSettings())


def test_heading_blocks_define_boundaries_without_becoming_body_evidence(tmp_path: Path):
    source = tmp_path / "sections.md"
    source.write_text(
        "# Guide\n\n"
        "Intro body.\n\n"
        "## Setup\n\n"
        "Setup body.\n\n"
        "## Usage\n\n"
        "Usage body.\n",
        encoding="utf-8",
    )

    document = parse_file(source)
    chunks = chunk_document(document, Settings(chunk_max_tokens=100, chunk_overlap_tokens=0))

    assert [chunk.heading_path for chunk in chunks] == [
        ["Guide"],
        ["Guide", "Setup"],
        ["Guide", "Usage"],
    ]
    assert [chunk.content for chunk in chunks] == ["Intro body.", "Setup body.", "Usage body."]
    assert all("Guide" != chunk.content for chunk in chunks)
