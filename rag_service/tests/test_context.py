from rag_service.app.config import Settings
from rag_service.app.context import ContextBuilder
from rag_service.app.schemas import SearchHit


def _hit(chunk_id: int, content: str, *, source: str = "notes.md", title: str = "Notes") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        content=content,
        source=source,
        title=title,
        score=0.95,
        rank=chunk_id,
        matched_channels=["vector", "keyword"],
    )


def test_title_and_source_topic_do_not_make_missing_cluster_election_evidence():
    hit = _hit(
        1,
        "Redis persistence uses RDB snapshots and AOF logs. BGSAVE forks a child process.",
        source="knowledge/Redis Cluster.pdf",
        title="Redis Cluster failover primary election",
    )

    context = ContextBuilder(Settings()).build(
        "How does Redis Cluster elect a new primary during failover?", [hit]
    )

    assert context.citations == []
    assert context.evidence.eligible_citation_count == 0
    assert context.evidence.evidence_gate_passed is False


def test_separate_topic_chunks_do_not_support_cross_topic_relationship():
    hits = [
        _hit(1, "std::vector increases capacity geometrically when more elements are appended."),
        _hit(2, "Redis AOF rewrite compacts the append-only log in the background.", source="redis.md"),
    ]

    context = ContextBuilder(Settings()).build(
        "How does std::vector capacity growth control Redis AOF rewrite scheduling?", hits
    )

    assert context.citations == []
    assert context.evidence.evidence_gate_passed is False
    assert context.evidence.evidence_gate_reason == "relationship_not_supported_in_one_chunk"


def test_one_direct_chunk_with_explicit_relationship_can_be_used():
    hit = _hit(
        1,
        "The scheduler uses std::vector capacity growth to control when Redis AOF rewrite is started.",
    )

    context = ContextBuilder(Settings()).build(
        "How does std::vector capacity growth control Redis AOF rewrite scheduling?", [hit]
    )

    assert len(context.citations) == 1
    assert context.evidence.evidence_gate_passed is True


def test_chinese_paraphrase_uses_body_evidence_and_keeps_multi_source_discovery():
    builder = ContextBuilder(Settings())
    supported = builder.build(
        "Redis 执行 BGSAVE 时怎样避免阻塞主线程？",
        [_hit(1, "Redis BGSAVE creates an RDB snapshot by forking a child process, so the parent can continue serving requests.")],
    )
    discovery = builder.build(
        "哪些笔记分别介绍 MySQL 关系表和 Redis 持久化？",
        [
            _hit(2, "MySQL tables store relational records. CREATE TABLE users (id INT PRIMARY KEY);", source="mysql.pdf", title="MySQL 初始使用指南"),
            _hit(3, "Redis persistence supports RDB snapshots and AOF logs.", source="redis.pdf", title="Redis persistence"),
        ],
    )

    assert len(supported.citations) == 1
    assert supported.evidence.evidence_gate_passed is True
    assert len(discovery.citations) == 2
    assert discovery.evidence.evidence_gate_passed is True
    assert discovery.evidence.evidence_gate_reason == "multi_source_discovery"


def test_english_question_can_use_chinese_pgvector_body_evidence():
    hit = _hit(
        1,
        "pgvector 是 PostgreSQL 的向量扩展。它允许把 embedding 向量和原文元数据放在同一个数据库中，并使用余弦距离等运算检索语义相近的文本。",
    )

    context = ContextBuilder(Settings()).build(
        "What does pgvector store and which distance can it use for semantic retrieval?",
        [hit],
    )

    assert len(context.citations) == 1
    assert context.evidence.evidence_gate_passed is True


def test_english_tcp_udp_comparison_can_use_chinese_body_evidence():
    hit = _hit(
        1,
        "TCP 通过建立连接、按序编号字节、确认数据到达并重传丢失的数据段，提供可靠有序的传输。",
    )
    udp = _hit(2, "UDP 发送独立的数据报，不建立连接。")

    context = ContextBuilder(Settings()).build("How does TCP differ from UDP?", [hit, udp])

    assert context.evidence.evidence_gate_passed is True


def test_redis_election_question_requires_election_mechanism_evidence():
    hit = _hit(
        1,
        "Redis replication and failover promote a replica when a primary becomes unavailable.",
    )

    context = ContextBuilder(Settings()).build(
        "How does Redis Cluster elect a new primary during failover?", [hit]
    )

    assert context.citations == []
    assert context.evidence.evidence_gate_passed is False


def test_english_checked_in_note_discovery_cites_each_body_supported_source():
    context = ContextBuilder(Settings()).build(
        "Which checked-in notes explain both MySQL relational tables and Redis persistence?",
        [
            _hit(1, "MySQL stores relational records in tables with rows and columns.", source="mysql.pdf"),
            _hit(2, "Redis persistence writes RDB snapshots and appends AOF operations.", source="redis.pdf"),
        ],
    )

    assert context.evidence.evidence_gate_reason == "multi_source_discovery"
    assert {citation.source for citation in context.citations} == {"mysql.pdf", "redis.pdf"}


def test_multi_source_discovery_does_not_use_title_as_topic_evidence():
    context = ContextBuilder(Settings()).build(
        "Which notes explain MySQL relational tables and Redis persistence?",
        [
            _hit(1, "CREATE TABLE users (id INT PRIMARY KEY);", source="mysql.pdf", title="MySQL relational tables"),
            _hit(2, "Redis persistence writes RDB snapshots and appends AOF operations.", source="redis.pdf"),
        ],
    )

    assert not any(citation.source == "mysql.pdf" for citation in context.citations)


def test_supported_multi_source_comparison_is_distinct_from_unsupported_causality():
    comparison = ContextBuilder(Settings()).build(
        "How does Redis persistence differ from MySQL table storage?",
        [
            _hit(1, "Redis is an in-memory data store. RDB persistence writes periodic snapshots, while AOF appends write operations.", source="redis.md"),
            _hit(2, "MySQL stores relational records in tables with rows and columns.", source="mysql.md"),
        ],
    )
    causal = ContextBuilder(Settings()).build(
        "How does std::vector capacity growth control Redis AOF rewrite scheduling?",
        [
            _hit(3, "std::vector increases capacity geometrically when elements are appended."),
            _hit(4, "Redis AOF rewrite compacts the append-only log in the background.", source="redis.md"),
        ],
    )

    assert {citation.source for citation in comparison.citations} == {"redis.md", "mysql.md"}
    assert comparison.evidence.evidence_gate_passed is True
    assert causal.citations == []
    assert causal.evidence.evidence_gate_passed is False


def test_comparison_does_not_use_broad_entities_without_requested_subtopic():
    context = ContextBuilder(Settings()).build(
        "How does Redis persistence differ from MySQL MVCC gap locks?",
        [
            _hit(1, "Redis is an in-memory data store with RDB and AOF persistence.", source="redis.md"),
            _hit(2, "MySQL stores relational records in tables with rows and columns.", source="mysql.md"),
        ],
    )

    assert context.citations == []
    assert context.evidence.evidence_gate_passed is False
