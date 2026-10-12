from rag_service.app.context import ContextBuilder
from rag_service.app.config import Settings
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.embeddings import HashingEmbeddingProvider
from rag_service.app.models import Chunk, Document
from rag_service.app.retrieval import RetrievalService, reciprocal_rank_fusion
from rag_service.app.schemas import SearchHit


def test_rrf_prefers_an_item_ranked_in_multiple_channels():
    scores = reciprocal_rank_fusion({"vector": [1, 2], "keyword": [2, 3]})
    assert scores[2] > scores[1]
    assert scores[2] > scores[3]


def test_context_builder_adds_stable_citation_labels():
    hits = [
        SearchHit(
            chunk_id=1,
            content="pgvector stores vectors in PostgreSQL.",
            source="knowledge/db.md",
            title="DB",
            heading=["Storage"],
            score=0.8,
            matched_channels=["vector", "keyword"],
        ),
        SearchHit(
            chunk_id=2,
            content="The vector column supports cosine distance.",
            source="knowledge/db.md",
            title="DB",
            heading=["Storage"],
            score=0.7,
            matched_channels=["vector"],
        ),
    ]
    package = ContextBuilder().build("How does pgvector work?", hits)
    assert "[DOC-1]" in package.text
    assert len(package.citations) == 1
    assert package.citations[0].citation_id == "DOC-1"
    assert package.confidence_score > 0


def test_retrieval_adds_a_small_adjacent_chunk_window():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=3,
        retrieval_keyword_candidates=3,
        retrieval_context_limit=1,
        retrieval_neighbor_window=1,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path="note.md",
            source_type="markdown",
            title="Note",
            file_hash="a" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add(document)
        session.flush()
        for index, content in enumerate(("before context", "middle RRF anchor", "after context")):
            session.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=content,
                    heading_path=["Section"],
                    token_count=3,
                    embedding=provider.embed_query(content),
                )
            )
        session.commit()

    hits = RetrievalService(factory, settings=settings, embedding_provider=provider).search("middle RRF")
    assert hits
    assert hits[0].content == "middle RRF anchor"
    assert any(hit.matched_channels == ["neighbor"] for hit in hits[1:])
