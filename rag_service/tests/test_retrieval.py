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


def test_rrf_accepts_optional_channel_weights():
    scores = reciprocal_rank_fusion(
        {"vector": [1], "keyword": [2]}, weights={"vector": 2.0, "keyword": 0.5}
    )
    assert scores[1] > scores[2]


def test_retrieval_matches_technical_identifiers_and_chinese_ngrams_in_metadata():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=3,
        retrieval_keyword_candidates=3,
        retrieval_phrase_candidates=3,
        retrieval_context_limit=2,
        retrieval_neighbor_window=0,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path="knowledge/pgvector.md",
            source_type="markdown",
            title="pgvector RRF 运维指南",
            file_hash="b" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add(document)
        session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                content="向量索引支持 C++ 扩展。",
                heading_path=["pgvector 配置"],
                token_count=5,
                embedding=provider.embed_query("向量索引支持 C++ 扩展"),
            )
        )
        session.commit()

    hits = RetrievalService(factory, settings=settings, embedding_provider=provider).search(
        "pgvector RRF C++ 向量索引"
    )
    assert hits
    assert hits[0].keyword_score > 0
    assert "pgvector" in hits[0].matched_terms
    assert any(term in hits[0].matched_terms for term in ("向量", "向量索引"))


def test_retrieval_prefers_exact_title_match_over_body_only_keyword_match():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=3,
        retrieval_keyword_candidates=3,
        retrieval_phrase_candidates=3,
        retrieval_context_limit=2,
        retrieval_neighbor_window=0,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    shared_vector = [0.0] * settings.embedding_dimensions
    shared_vector[0] = 1.0
    with factory() as session:
        body_document = Document(
            source_path="knowledge/vector.md",
            source_type="markdown",
            title="Vector reference",
            file_hash="d" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        title_document = Document(
            source_path="knowledge/stl.md",
            source_type="markdown",
            title="STL reference",
            file_hash="e" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add_all([body_document, title_document])
        session.flush()
        session.add_all(
            [
                Chunk(
                    document_id=body_document.id,
                    chunk_index=0,
                    content="This body mentions STL as an implementation detail.",
                    heading_path=["Vector"],
                    token_count=8,
                    embedding=shared_vector,
                ),
                Chunk(
                    document_id=title_document.id,
                    chunk_index=0,
                    content="The standard template library provides reusable components.",
                    heading_path=["Overview"],
                    token_count=8,
                    embedding=shared_vector,
                ),
            ]
        )
        session.commit()

    hits = RetrievalService(factory, settings=settings, embedding_provider=provider).search("STL 是什么")

    assert hits[0].source == "knowledge/stl.md"


def test_retrieval_reports_exact_phrase_score_and_primary_neighbor_metadata():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=3,
        retrieval_keyword_candidates=3,
        retrieval_phrase_candidates=3,
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
            title="RRF reference",
            file_hash="c" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add(document)
        session.flush()
        for index, content in enumerate(("before context", "RRF anchor phrase", "after context")):
            session.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=content,
                    heading_path=["RRF"],
                    token_count=3,
                    embedding=provider.embed_query(content),
                )
            )
        session.commit()

    hits = RetrievalService(factory, settings=settings, embedding_provider=provider).search(
        "RRF anchor phrase"
    )
    primary = [hit for hit in hits if not hit.is_neighbor]
    neighbors = [hit for hit in hits if hit.is_neighbor]
    assert primary and primary[0].rank == 1
    assert primary[0].phrase_score > 0
    assert neighbors and all(hit.rank is None for hit in neighbors)
    assert all(hit.is_neighbor for hit in neighbors)


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
