from __future__ import annotations

from types import SimpleNamespace

from rag_service.app.config import Settings
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.ingestion import IngestionService
from rag_service.app.models import Chunk, Document


class FakeEmbeddingProvider:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index == 0) for index in range(self.dimensions)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class RecordingEmbeddingProvider(FakeEmbeddingProvider):
    pass


def _service(tmp_path, parser=None, chunker=None):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    settings = Settings(
        knowledge_root=knowledge,
        database_url="sqlite+pysqlite:///:memory:",
    )
    engine = init_db(get_engine(settings))
    factory = get_session_factory(engine=engine)
    parser = parser or (lambda path: SimpleNamespace(title=path.stem))
    chunker = chunker or (
        lambda document, settings: [
            SimpleNamespace(
                content=f"content for {document.title}",
                heading_path=["Notes"],
                token_count=4,
                page_from=None,
                page_to=None,
            )
        ]
    )
    return knowledge, IngestionService(
        settings,
        session_factory=factory,
        embedding_provider=FakeEmbeddingProvider(),
        parser=parser,
        chunker=chunker,
    ), factory


def test_unchanged_file_is_skipped_and_changed_file_reindexed(tmp_path):
    knowledge, service, factory = _service(tmp_path)
    source = knowledge / "note.md"
    source.write_text("first", encoding="utf-8")

    first = service.scan()
    second = service.scan()
    source.write_text("changed", encoding="utf-8")
    third = service.scan()

    assert (first.processed_files, first.skipped_files) == (1, 0)
    assert (second.processed_files, second.skipped_files) == (0, 1)
    assert (third.processed_files, third.skipped_files) == (1, 0)
    with factory() as session:
        assert session.query(Document).count() == 1
        assert session.query(Chunk).count() == 1


def test_deleted_source_is_removed_from_index(tmp_path):
    knowledge, service, factory = _service(tmp_path)
    source = knowledge / "note.md"
    source.write_text("first", encoding="utf-8")
    service.scan()
    source.unlink()

    summary = service.scan()

    assert summary.deleted_documents == 1
    with factory() as session:
        assert session.query(Document).count() == 0
        assert session.query(Chunk).count() == 0


def test_one_failed_file_does_not_stop_other_files(tmp_path):
    def parser(path):
        if path.stem == "bad":
            raise ValueError("malformed source")
        return SimpleNamespace(title=path.stem)

    knowledge, service, factory = _service(tmp_path, parser=parser)
    (knowledge / "bad.md").write_text("bad", encoding="utf-8")
    (knowledge / "good.md").write_text("good", encoding="utf-8")

    summary = service.scan()

    assert summary.processed_files == 1
    assert summary.failed_files == 1
    with factory() as session:
        statuses = {document.source_path: document.status for document in session.query(Document).all()}
        assert statuses == {"bad.md": "failed", "good.md": "ready"}
        assert session.query(Chunk).count() == 1


def test_embedding_receives_heading_context_but_chunk_content_stays_body(tmp_path):
    provider = RecordingEmbeddingProvider()
    payload = SimpleNamespace(
        content="Body text.",
        heading_path=["Guide", "Setup"],
        token_count=2,
        page_from=None,
        page_to=None,
    )
    knowledge, service, factory = _service(
        tmp_path,
        chunker=lambda document, settings: [payload],
    )
    service.embedding_provider = provider
    source = knowledge / "note.md"
    source.write_text("body", encoding="utf-8")

    summary = service.scan()

    assert summary.processed_files == 1
    assert provider.calls == [["Guide > Setup\n\nBody text."]]
    with factory() as session:
        chunk = session.query(Chunk).one()
        assert chunk.content == "Body text."
        assert chunk.heading_path == ["Guide", "Setup"]
