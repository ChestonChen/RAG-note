"""Offline retrieval evaluation for the local RAG pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Running ``python scripts/evaluate_retrieval.py`` puts ``scripts`` on
# sys.path, so add the project root for imports from the application package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_service.app.config import Settings
from rag_service.app.context import ContextBuilder
from rag_service.app.db import get_engine, get_session_factory, init_db
from rag_service.app.embeddings import create_embedding_provider
from rag_service.app.ingestion import IngestionService
from rag_service.app.retrieval import RetrievalService
from rag_service.app.schemas import SearchHit


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and lightly validate one evaluation case per JSONL line."""

    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"case on line {line_number} must be an object")
            missing = {"id", "question", "relevant_sources", "answerable"} - value.keys()
            if missing:
                raise ValueError(f"case on line {line_number} is missing: {sorted(missing)}")
            if not isinstance(value["id"], str) or not value["id"].strip():
                raise ValueError(f"id on line {line_number} must be a non-empty string")
            if not isinstance(value["question"], str) or not value["question"].strip():
                raise ValueError(f"question on line {line_number} must be a non-empty string")
            if not isinstance(value["answerable"], bool):
                raise ValueError(f"answerable on line {line_number} must be a boolean")
            if not isinstance(value["relevant_sources"], list):
                raise ValueError(f"relevant_sources on line {line_number} must be a list")
            if any(not isinstance(item, str) for item in value["relevant_sources"]):
                raise ValueError(f"relevant_sources on line {line_number} must contain strings")
            if "relevant_headings" in value:
                if not isinstance(value["relevant_headings"], list) or any(
                    not isinstance(item, str) for item in value["relevant_headings"]
                ):
                    raise ValueError(
                        f"relevant_headings on line {line_number} must be a list of strings"
                    )
            for field in ("tags", "expected_channels"):
                if field in value and not isinstance(value[field], list):
                    raise ValueError(f"{field} on line {line_number} must be a list")
                if field in value and any(not isinstance(item, str) for item in value[field]):
                    raise ValueError(f"{field} on line {line_number} must contain strings")
            if (
                "paraphrase_of" in value
                and value["paraphrase_of"] is not None
                and not isinstance(value["paraphrase_of"], str)
            ):
                raise ValueError(
                    f"paraphrase_of on line {line_number} must be a string"
                )
            cases.append(value)
    ids = {case["id"] for case in cases}
    if len(ids) != len(cases):
        raise ValueError(f"case ids in {path} must be unique")
    for case in cases:
        parent = case.get("paraphrase_of")
        if parent is not None and parent not in ids:
            raise ValueError(
                f"paraphrase_of for case {case['id']!r} references unknown case {parent!r}"
            )
    return cases


def _source_key(source: str) -> str:
    value = str(source).replace("\\", "/").lstrip("./")
    for prefix in ("eval/fixtures/", "fixtures/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value


def _hit_dict(hit: SearchHit) -> dict[str, Any]:
    if hasattr(hit, "model_dump"):
        return hit.model_dump()
    return hit.dict()


def _ranked_primary(hits: Iterable[SearchHit]) -> list[tuple[int, SearchHit]]:
    ranked: list[tuple[int, SearchHit]] = []
    for index, hit in enumerate(hits, start=1):
        ranked.append((int(hit.rank) if hit.rank is not None else index, hit))
    return ranked


def _embedding_runtime(retrieval_service: Any, settings: Settings) -> dict[str, Any]:
    provider = getattr(retrieval_service, "embedding_provider", None)
    if provider is None:
        return {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "signature": settings.embedding_signature(),
        }
    provider_name = str(getattr(provider, "provider_name", settings.embedding_provider))
    model = str(
        getattr(provider, "model", None)
        or getattr(provider, "model_path", None)
        or settings.embedding_model
    )
    dimensions = getattr(provider, "dimensions", None) or settings.embedding_dimensions
    return {
        "provider": provider_name,
        "model": model,
        "dimensions": int(dimensions),
        "signature": f"{provider_name}:{model}:dim={int(dimensions)}",
    }


def evaluate_cases(
    retrieval_service: Any,
    context_builder: ContextBuilder,
    cases: list[dict[str, Any]],
    limit: int = 5,
    confidence_threshold: float = 0.28,
) -> dict[str, Any]:
    """Run cases and calculate source-level retrieval and citation metrics."""

    if limit < 1:
        raise ValueError("limit must be at least 1")
    case_reports: list[dict[str, Any]] = []
    answerable_reports: list[dict[str, Any]] = []
    unknown_reports: list[dict[str, Any]] = []
    settings = (
        getattr(retrieval_service, "settings", None)
        or getattr(context_builder, "settings", None)
        or Settings()
    )
    embedding_runtime = _embedding_runtime(retrieval_service, settings)
    channel_hits: dict[str, int] = {}
    channel_relevant: dict[str, int] = {}

    for case in cases:
        question = str(case["question"])
        hits = list(retrieval_service.search(question, limit=limit))
        primary = [hit for hit in hits if not hit.is_neighbor]
        neighbors = [hit for hit in hits if hit.is_neighbor]
        ranked_primary = _ranked_primary(primary)
        relevant = {_source_key(source) for source in case.get("relevant_sources", [])}
        primary_ranks: dict[str, int] = {}
        for rank, hit in ranked_primary:
            source = _source_key(hit.source)
            if source in relevant:
                primary_ranks.setdefault(source, rank)
        for hit in primary:
            source = _source_key(hit.source)
            for channel in hit.matched_channels:
                if channel == "neighbor":
                    continue
                channel_hits[channel] = channel_hits.get(channel, 0) + 1
                if source in relevant:
                    channel_relevant[channel] = channel_relevant.get(channel, 0) + 1
        relevant_ranks = sorted(primary_ranks.values())
        context = context_builder.build(question, hits)
        cited_sources = {_source_key(citation.source) for citation in context.citations}
        citation_coverage = (
            len(relevant & cited_sources) / len(relevant) if relevant else 0.0
        )
        cited_sources = {_source_key(citation.source) for citation in context.citations}
        citation_precision = (
            len(relevant & cited_sources) / len(cited_sources) if cited_sources else 0.0
        )
        score = float(context.confidence_score)
        low_threshold = float(
            getattr(settings, "low_confidence_threshold", confidence_threshold)
        )
        high_threshold = 0.65
        confidence_bucket = (
            "low"
            if score < low_threshold
            else "high"
            if score >= high_threshold
            else "medium"
        )
        report = {
            "id": case["id"],
            "question": question,
            "answerable": bool(case["answerable"]),
            "relevant_sources": list(case.get("relevant_sources", [])),
            "relevant_headings": list(case.get("relevant_headings", [])),
            "tags": list(case.get("tags", [])),
            "expected_channels": list(case.get("expected_channels", [])),
            "paraphrase_of": case.get("paraphrase_of"),
            "primary_hits": [_hit_dict(hit) for hit in primary],
            "neighbor_hits": [_hit_dict(hit) for hit in neighbors],
            "confidence_score": context.confidence_score,
            "confidence_bucket": confidence_bucket,
            "citation_coverage": citation_coverage,
            "citation_precision": citation_precision,
            "unknown_false_positive": (not bool(case["answerable"]) and score >= confidence_threshold),
            "relevant_primary_ranks": relevant_ranks,
            "primary_ranks": primary_ranks,
            "ranks": {
                "relevant_primary": relevant_ranks,
                "primary_by_source": primary_ranks,
            },
        }
        case_reports.append(report)
        if case["answerable"]:
            answerable_reports.append(report)
        else:
            unknown_reports.append(report)

    def hit_at(k: int) -> float:
        if not answerable_reports:
            return 0.0
        return sum(
            any(rank <= k for rank in report["relevant_primary_ranks"])
            for report in answerable_reports
        ) / len(answerable_reports)

    reciprocal_ranks = [
        (1.0 / report["relevant_primary_ranks"][0])
        if report["relevant_primary_ranks"]
        else 0.0
        for report in answerable_reports
    ]
    recall_reports = [
        report
        for report in answerable_reports
        if report["relevant_sources"]
    ]
    recalls = []
    for report in recall_reports:
        relevant = {_source_key(source) for source in report["relevant_sources"]}
        retrieved = {
            _source_key(hit["source"])
            for hit in report["primary_hits"][:5]
        }
        recalls.append(len(relevant & retrieved) / len(relevant))
    coverage_values = [
        report["citation_coverage"]
        for report in answerable_reports
        if report["relevant_sources"]
    ]
    false_positives = sum(
        report["unknown_false_positive"] for report in unknown_reports
    )
    citation_precision_values = [
        report["citation_precision"] for report in answerable_reports if report["relevant_sources"]
    ]
    all_bucket_counts = {bucket: 0 for bucket in ("low", "medium", "high")}
    for report in case_reports:
        all_bucket_counts[report["confidence_bucket"]] += 1
    channel_contribution = {
        channel: {
            "hit_count": count,
            "relevant_hit_count": channel_relevant.get(channel, 0),
            "relevant_rate": channel_relevant.get(channel, 0) / count if count else 0.0,
        }
        for channel, count in sorted(channel_hits.items())
    }
    reranker_debug = getattr(retrieval_service, "get_last_debug", lambda: {})()
    reranker_state = (reranker_debug or {}).get("reranker", {})
    metrics = {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "mean_source_recall_at_5": sum(recalls) / len(recalls) if recalls else 0.0,
        "citation_source_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        ),
        "unknown_false_positive_rate": (
            false_positives / len(unknown_reports) if unknown_reports else 0.0
        ),
    }
    retrieval_configuration = {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "retrieval_context_limit": settings.retrieval_context_limit,
        "retrieval_neighbor_window": settings.retrieval_neighbor_window,
        "retrieval_per_document_limit": settings.retrieval_per_document_limit,
        "low_confidence_threshold": low_threshold,
        "reranker_enabled": bool(settings.reranker_enabled),
        "reranker_model": settings.reranker_model,
    }
    return {
        "metrics": metrics,
        "cases": case_reports,
        "embedding_signature": embedding_runtime["signature"],
        "embedding_runtime": embedding_runtime,
        "retrieval_configuration": retrieval_configuration,
        "retrieval_config": retrieval_configuration,
        "configured_embedding_signature": settings.embedding_signature(),
        "confidence_buckets": all_bucket_counts,
        "citation_source_precision": (
            sum(citation_precision_values) / len(citation_precision_values)
            if citation_precision_values
            else 0.0
        ),
        "citation_precision": (
            sum(citation_precision_values) / len(citation_precision_values)
            if citation_precision_values
            else 0.0
        ),
        "citation_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        ),
        "per_channel_contribution": channel_contribution,
        "per_channel_metrics": channel_contribution,
        "reranker_comparison": {
            "enabled": bool(settings.reranker_enabled),
            "attempted": bool(reranker_state.get("attempted", False)),
            "status": reranker_state.get("status", "disabled"),
            "available": bool(reranker_state.get("attempted", False)),
            "comparison_run": False,
            "note": "status metadata only; enable an explicit A/B run to compare rankings",
        },
    }


def _run(settings: Settings, cases_path: Path, output_path: Path) -> dict[str, Any]:
    engine = init_db(get_engine(settings), settings=settings)
    try:
        factory = get_session_factory(engine=engine)
        provider = create_embedding_provider(settings)
        summary = IngestionService(
            settings,
            session_factory=factory,
            embedding_provider=provider,
        ).scan()
        report = evaluate_cases(
            RetrievalService(factory, settings=settings, embedding_provider=provider),
            ContextBuilder(settings),
            load_cases(cases_path),
            limit=5,
            confidence_threshold=settings.low_confidence_threshold,
        )
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        report["ingestion"] = {
            "scanned_files": summary.scanned_files,
            "processed_files": summary.processed_files,
            "failed_files": summary.failed_files,
            "chunk_count": summary.chunk_count,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated", action="store_true", help="evaluate copied fixtures in temporary SQLite")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "eval" / "retrieval_cases.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "retrieval_eval.json")
    parser.add_argument("--knowledge-root", type=Path)
    parser.add_argument("--database-url")
    args = parser.parse_args(argv)

    if args.isolated:
        with tempfile.TemporaryDirectory(prefix="rag-retrieval-eval-") as directory:
            temporary_root = Path(directory)
            knowledge_root = temporary_root / "knowledge"
            shutil.copytree(PROJECT_ROOT / "eval" / "fixtures", knowledge_root)
            settings = Settings(
                knowledge_root=knowledge_root,
                database_url=f"sqlite+pysqlite:///{temporary_root / 'index.sqlite3'}",
                embedding_provider="hashing",
            )
            _run(settings, args.cases, args.output)
        return 0

    settings = Settings(
        knowledge_root=args.knowledge_root or Path("knowledge"),
        database_url=args.database_url or Settings().database_url,
    )
    _run(settings, args.cases, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
