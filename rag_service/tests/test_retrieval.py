from rag_service.app.context import ContextBuilder
from rag_service.app.config import Settings
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.embeddings import HashingEmbeddingProvider
from rag_service.app.models import Chunk, Document
from rag_service.app.retrieval import RetrievalService, reciprocal_rank_fusion
from rag_service.app.schemas import SearchHit
from rag_service.app.query_planning import QueryPlanner
from rag_service.app.search_text import deterministic_cjk_ngrams
from rag_service.app.term_dictionary import TermDictionary


def test_exact_and_full_text_channels_are_independent_on_sqlite():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=3,
        retrieval_keyword_candidates=3,
        retrieval_phrase_candidates=3,
        retrieval_full_text_candidates=3,
        retrieval_exact_candidates=3,
        retrieval_context_limit=3,
        retrieval_neighbor_window=0,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    vector = provider.embed_query("pgvector")
    with factory() as session:
        exact_document = Document(
            source_path="exact.md",
            source_type="markdown",
            title="pgvector exact title",
            file_hash="1" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        body_document = Document(
            source_path="body.md",
            source_type="markdown",
            title="Database notes",
            file_hash="2" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add_all([exact_document, body_document])
        session.flush()
        session.add_all(
            [
                Chunk(
                    document_id=exact_document.id,
                    chunk_index=0,
                    content="Vector storage overview.",
                    heading_path=["Storage"],
                    token_count=3,
                    embedding=vector,
                ),
                Chunk(
                    document_id=body_document.id,
                    chunk_index=0,
                    content="pgvector appears in the body of this note.",
                    heading_path=["Storage"],
                    token_count=7,
                    embedding=vector,
                ),
            ]
        )
        session.commit()

    service = RetrievalService(factory, settings=settings, embedding_provider=provider)
    hits = service.search("pgvector")
    assert hits
    assert any(hit.exact_score > 0 and "exact" in hit.matched_channels for hit in hits)
    assert any(hit.full_text_score > 0 and "full_text" in hit.matched_channels for hit in hits)
    channels = service.get_last_debug()["channels"]
    assert "full_text" in channels
    assert "exact" in channels
    assert service.get_last_debug()["timings_ms"]["full_text"] >= 0
    assert service.get_last_debug()["timings_ms"]["exact"] >= 0


def test_postgres_full_text_query_uses_bound_query_text_without_sql_fragments():
    from sqlalchemy.dialects import postgresql

    settings = Settings(database_url="sqlite+pysqlite:///:memory:")
    service = RetrievalService(lambda: None, settings=settings, embedding_provider=HashingEmbeddingProvider(settings.embedding_dimensions))

    class CaptureSession:
        def __init__(self):
            self.statement = None
            self.dialect = postgresql.dialect()

        def get_bind(self):
            return type("Bind", (), {"dialect": self.dialect})()

        def execute(self, statement):
            self.statement = statement
            return []

    session = CaptureSession()
    query = "PostgreSQL 中文检索 std::vector"
    assert service._postgres_full_text_scores(session, query) == ([], {})
    compiled = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "to_tsvector" in compiled
    assert "websearch_to_tsquery" in compiled
    assert query not in compiled
    params = session.statement.compile(dialect=postgresql.dialect()).params
    bound_query = next(value for value in params.values() if isinstance(value, str))
    assert "postgresql 中文检索 std::vector" in bound_query
    assert all(ngram in bound_query for ngram in deterministic_cjk_ngrams(query))

    injection = "pgvector'); DROP TABLE chunks; --"
    assert service._postgres_full_text_scores(session, injection) == ([], {})
    injection_sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert injection not in injection_sql


def test_exact_channel_keeps_technical_identifier_boundaries():
    settings = Settings(database_url="sqlite+pysqlite:///:memory:")
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path="dataset.md",
            source_type="markdown",
            title="Dataset operations",
            file_hash="3" * 64,
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
                content="dataset operations",
                heading_path=["Collections"],
                token_count=2,
                embedding=provider.embed_query("dataset operations"),
            )
        )
        session.commit()
        service = RetrievalService(factory, settings=settings, embedding_provider=provider)
        exact_ids, exact_scores = service._exact_scores(session, "set")
    assert exact_ids == []
    assert exact_scores == {}


def test_rrf_prefers_an_item_ranked_in_multiple_channels():
    scores = reciprocal_rank_fusion({"vector": [1, 2], "keyword": [2, 3]})
    assert scores[2] > scores[1]
    assert scores[2] > scores[3]


def test_rrf_accepts_optional_channel_weights():
    scores = reciprocal_rank_fusion(
        {"vector": [1], "keyword": [2]}, weights={"vector": 2.0, "keyword": 0.5}
    )
    assert scores[1] > scores[2]


def test_retrieval_debug_accessor_returns_an_isolated_snapshot():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_context_limit=1,
        retrieval_neighbor_window=0,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    service = RetrievalService(factory, settings=settings, embedding_provider=provider)

    service.search("no indexed content")
    snapshot = service.get_last_debug()
    snapshot["timings_ms"]["test"] = 999

    assert service.latest_debug["timings_ms"].get("test") is None


def test_query_planner_normalizes_terms_and_expands_intent_without_remote_calls():
    plan = QueryPlanner().plan("  pgvector 怎么检索语义向量？ ")

    assert plan.normalized_query == "pgvector 怎么检索语义向量?"
    assert plan.intent in {"definition", "implementation", "usage", "general"}
    assert "pgvector" in {term.casefold() for term in plan.expanded_terms}
    assert any(term in plan.expanded_terms for term in ("向量数据库", "embedding", "向量检索"))


def test_query_planner_does_not_expand_pgvector_as_cpp_vector():
    plan = QueryPlanner().plan("pgvector 的余弦相似度")

    assert "std::vector" not in plan.expanded_terms
    assert "C++" not in plan.expanded_terms


class _RecordingEmbeddingProvider:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.0] * self.dimensions


def test_retrieval_embedding_receives_required_terms_and_expansion_can_be_disabled(tmp_path):
    dictionary_path = tmp_path / "terms.json"
    dictionary_path.write_text('{"required_terms":["required-anchor"]}', encoding="utf-8")
    enabled_settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        term_dictionary_path=dictionary_path,
        retrieval_query_expansion_enabled=True,
    )
    engine = init_db(get_engine(enabled_settings), settings=enabled_settings)
    factory = get_session_factory(engine=engine)
    recording = _RecordingEmbeddingProvider(enabled_settings.embedding_dimensions)
    RetrievalService(factory, settings=enabled_settings, embedding_provider=recording).search("pgvector 是什么")
    assert recording.queries and "required-anchor" in recording.queries[-1]

    disabled_settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        term_dictionary_path=dictionary_path,
        retrieval_query_expansion_enabled=False,
    )
    disabled_engine = init_db(get_engine(disabled_settings), settings=disabled_settings)
    disabled_recording = _RecordingEmbeddingProvider(disabled_settings.embedding_dimensions)
    RetrievalService(get_session_factory(engine=disabled_engine), settings=disabled_settings, embedding_provider=disabled_recording).search("pgvector 是什么")
    assert disabled_recording.queries[-1] == "pgvector 是什么"


class _FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        return [1.0 if "preferred" in document else 0.0 for document in documents]


def test_optional_reranker_can_promote_a_late_candidate_and_reports_status():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=3,
        retrieval_keyword_candidates=3,
        retrieval_phrase_candidates=3,
        retrieval_context_limit=2,
        retrieval_neighbor_window=0,
        reranker_enabled=True,
        reranker_candidates=3,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    reranker = _FakeReranker()
    with factory() as session:
        document = Document(
            source_path="rerank.md",
            source_type="markdown",
            title="Rerank",
            file_hash="f" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add(document)
        session.flush()
        for index, content in enumerate(("generic vector context", "preferred vector context")):
            session.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=content,
                    heading_path=["Rerank"],
                    token_count=3,
                    embedding=provider.embed_query("vector context"),
                )
            )
        session.commit()

    service = RetrievalService(
        factory,
        settings=settings,
        embedding_provider=provider,
        reranker=reranker,
    )
    hits = service.search("vector context")

    assert hits[0].content == "preferred vector context"
    assert reranker.calls
    assert service.latest_debug["reranker"]["status"] == "ok"
    assert service.latest_debug["reranker"]["attempted"] is True


def test_reranker_failure_fails_open_to_original_retrieval():
    class BrokenReranker:
        def rerank(self, query: str, documents: list[str]) -> list[float]:
            raise RuntimeError("model unavailable")

    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_context_limit=1,
        retrieval_neighbor_window=0,
        reranker_enabled=True,
        reranker_fail_open=True,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path="fallback.md",
            source_type="markdown",
            title="Fallback",
            file_hash="1" * 64,
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
                content="vector fallback",
                heading_path=["Fallback"],
                token_count=2,
                embedding=provider.embed_query("vector fallback"),
            )
        )
        session.commit()

    service = RetrievalService(
        factory,
        settings=settings,
        embedding_provider=provider,
        reranker=BrokenReranker(),
    )
    assert service.search("vector fallback")
    assert service.latest_debug["reranker"]["status"] == "failed_open"


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


def test_retrieval_filters_body_only_cross_topic_distractors_when_topic_is_explicit():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=5,
        retrieval_keyword_candidates=5,
        retrieval_phrase_candidates=5,
        retrieval_context_limit=5,
        retrieval_neighbor_window=0,
        retrieval_per_document_limit=5,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    shared_vector = [0.0] * settings.embedding_dimensions
    shared_vector[0] = 1.0
    with factory() as session:
        mysql_document = Document(
            source_path="knowledge/mysql.md",
            source_type="markdown",
            title="MySQL JOIN reference",
            file_hash="2" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        redis_document = Document(
            source_path="knowledge/redis.md",
            source_type="markdown",
            title="Redis cache notes",
            file_hash="3" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add_all([mysql_document, redis_document])
        session.flush()
        session.add_all(
            [
                Chunk(
                    document_id=mysql_document.id,
                    chunk_index=0,
                    content="LEFT JOIN keeps rows from the left table and fills missing values with NULL.",
                    heading_path=["MySQL", "JOIN"],
                    token_count=12,
                    embedding=shared_vector,
                ),
                Chunk(
                    document_id=redis_document.id,
                    chunk_index=0,
                    content="A Redis cache note mentions MySQL as an unrelated comparison.",
                    heading_path=["Redis", "Cache"],
                    token_count=14,
                    embedding=shared_vector,
                ),
            ]
        )
        session.commit()

    service = RetrievalService(factory, settings=settings, embedding_provider=provider)
    hits = service.search("MySQL LEFT JOIN 是什么")

    primary = [hit for hit in hits if not hit.is_neighbor]
    assert primary
    assert all(hit.source == "knowledge/mysql.md" for hit in primary)
    assert service.get_last_debug()["topic_filter"]["filtered_count"] == 1


def test_lexical_identifier_matching_does_not_treat_dataset_as_set():
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
        document = Document(
            source_path="lexical.md",
            source_type="markdown",
            title="Lexical",
            file_hash="9" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add(document)
        session.flush()
        session.add_all(
            [
                Chunk(
                    document_id=document.id,
                    chunk_index=0,
                    content="dataset operations",
                    heading_path=["Database"],
                    token_count=2,
                    embedding=shared_vector,
                ),
                Chunk(
                    document_id=document.id,
                    chunk_index=1,
                    content="set operations",
                    heading_path=["Collections"],
                    token_count=2,
                    embedding=shared_vector,
                ),
            ]
        )
        session.commit()

    hits = RetrievalService(factory, settings=settings, embedding_provider=provider).search("set")

    assert hits[0].content == "set operations"


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


def test_query_expansion_does_not_make_an_sql_only_note_match_a_mysql_question():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=3,
        retrieval_keyword_candidates=3,
        retrieval_phrase_candidates=3,
        retrieval_context_limit=3,
        retrieval_neighbor_window=0,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        mysql = Document(
            source_path="mysql.md",
            source_type="markdown",
            title="MySQL 建表指南",
            file_hash="1" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        sql = Document(
            source_path="sql.md",
            source_type="markdown",
            title="SQL 安全笔记",
            file_hash="2" * 64,
            parser_signature="parser-v1",
            chunking_signature="chunk-v1",
            embedding_signature=settings.embedding_signature(),
            status="ready",
        )
        session.add_all([mysql, sql])
        session.flush()
        session.add_all(
            [
                Chunk(
                    document_id=mysql.id,
                    chunk_index=0,
                    content="MySQL CREATE TABLE 用于创建数据库表。",
                    heading_path=["MySQL"],
                    token_count=8,
                    embedding=provider.embed_query("MySQL CREATE TABLE 用于创建数据库表"),
                ),
                Chunk(
                    document_id=sql.id,
                    chunk_index=0,
                    content="SQL 注入是应用安全问题，本文讨论参数化查询。",
                    heading_path=["SQL"],
                    token_count=8,
                    embedding=provider.embed_query("SQL 注入是应用安全问题"),
                ),
            ]
        )
        session.commit()

    hits = RetrievalService(factory, settings=settings, embedding_provider=provider).search(
        "MySQL 如何创建表"
    )
    assert not any(hit.source == "sql.md" and hit.keyword_score > 0 for hit in hits)


def test_low_similarity_vector_candidates_are_not_returned_for_unknown_questions():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=3,
        retrieval_keyword_candidates=3,
        retrieval_phrase_candidates=3,
        retrieval_context_limit=3,
        retrieval_neighbor_window=0,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path="known.md",
            source_type="markdown",
            title="Known topic",
            file_hash="3" * 64,
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
                content="This document only covers a known database topic.",
                heading_path=["Known"],
                token_count=9,
                embedding=provider.embed_query("This document only covers a known database topic"),
            )
        )
        session.commit()

    hits = RetrievalService(factory, settings=settings, embedding_provider=provider).search(
        "量子纠缠的数学证明是什么？"
    )
    assert hits == []


def test_phrase_match_contributes_to_display_confidence():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_vector_candidates=3,
        retrieval_keyword_candidates=3,
        retrieval_phrase_candidates=3,
        retrieval_context_limit=1,
        retrieval_neighbor_window=0,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path="phrase.md",
            source_type="markdown",
            title="Phrase note",
            file_hash="4" * 64,
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
                content="精确短语命中应该被视为可靠证据。",
                heading_path=["Phrase"],
                token_count=6,
                embedding=[0.0] * settings.embedding_dimensions,
            )
        )
        session.commit()

    hits = RetrievalService(factory, settings=settings, embedding_provider=provider).search(
        "精确短语命中应该被视为可靠证据"
    )
    assert hits and hits[0].phrase_score > 0
    assert hits[0].score >= 0.25


def test_reranker_with_non_finite_scores_fails_open():
    class InvalidReranker:
        def rerank(self, query: str, documents: list[str]) -> list[float]:
            return [float("nan")] * len(documents)

    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_context_limit=1,
        retrieval_neighbor_window=0,
        reranker_enabled=True,
        reranker_fail_open=True,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path="rerank-invalid.md",
            source_type="markdown",
            title="Rerank invalid",
            file_hash="5" * 64,
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
                content="reranker fallback",
                heading_path=["Rerank"],
                token_count=2,
                embedding=provider.embed_query("reranker fallback"),
            )
        )
        session.commit()

    service = RetrievalService(
        factory,
        settings=settings,
        embedding_provider=provider,
        reranker=InvalidReranker(),
    )
    assert service.search("reranker fallback")
    assert service.latest_debug["reranker"]["status"] == "failed_open"


def test_reranker_keeps_request_debug_when_shared_latest_snapshot_changes():
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_context_limit=1,
        retrieval_neighbor_window=0,
        reranker_enabled=True,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path="request-debug.md",
            source_type="markdown",
            title="Request debug",
            file_hash="6" * 64,
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
                content="request-local reranker",
                heading_path=["Debug"],
                token_count=2,
                embedding=provider.embed_query("request-local reranker"),
            )
        )
        session.commit()

    service: RetrievalService

    class InterleavingReranker:
        def rerank(self, query: str, documents: list[str]) -> list[float]:
            service.latest_debug = {"other_request": True}
            return [1.0] * len(documents)

    service = RetrievalService(
        factory,
        settings=settings,
        embedding_provider=provider,
        reranker=InterleavingReranker(),
    )
    assert service.search("request-local reranker")
    # A single candidate cannot be reordered. The reranker must leave the
    # score untouched while still keeping this request's debug trace isolated.
    assert service.get_last_debug()["reranker"]["status"] == "no_discrimination"


def test_reranker_with_equal_scores_does_not_inflate_hit_confidence():
    class EqualReranker:
        def rerank(self, query: str, documents: list[str]) -> list[float]:
            return [1.0] * len(documents)

    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        retrieval_context_limit=1,
        retrieval_neighbor_window=0,
        reranker_enabled=True,
    )
    engine = init_db(get_engine(settings), settings=settings)
    factory = get_session_factory(engine=engine)
    provider = HashingEmbeddingProvider(settings.embedding_dimensions)
    with factory() as session:
        document = Document(
            source_path="rerank-equal.md",
            source_type="markdown",
            title="Rerank equal",
            file_hash="7" * 64,
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
                content="equal reranker score",
                heading_path=["Rerank"],
                token_count=3,
                embedding=provider.embed_query("equal reranker score"),
            )
        )
        session.commit()

    service = RetrievalService(
        factory,
        settings=settings,
        embedding_provider=provider,
        reranker=EqualReranker(),
    )
    hits = service.search("equal reranker score")
    assert hits and hits[0].rerank_score is None
    assert service.get_last_debug()["reranker"]["status"] == "no_discrimination"
