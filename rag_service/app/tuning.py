"""Bounded, side-effect-free retrieval parameter tuning."""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import Settings
from .context import ContextBuilder
from .embeddings import create_embedding_provider
from .retrieval import RetrievalService


TUNABLE_ENV: dict[str, str] = {
    "retrieval_vector_candidates": "RETRIEVAL_VECTOR_CANDIDATES",
    "retrieval_vector_min_score": "RETRIEVAL_VECTOR_MIN_SCORE",
    "retrieval_keyword_candidates": "RETRIEVAL_KEYWORD_CANDIDATES",
    "retrieval_phrase_candidates": "RETRIEVAL_PHRASE_CANDIDATES",
    "retrieval_full_text_candidates": "RETRIEVAL_FULL_TEXT_CANDIDATES",
    "retrieval_exact_candidates": "RETRIEVAL_EXACT_CANDIDATES",
    "retrieval_trigram_candidates": "RETRIEVAL_TRIGRAM_CANDIDATES",
    "retrieval_trigram_min_similarity": "RETRIEVAL_TRIGRAM_MIN_SIMILARITY",
    "retrieval_vector_rrf_weight": "RETRIEVAL_VECTOR_RRF_WEIGHT",
    "retrieval_keyword_rrf_weight": "RETRIEVAL_KEYWORD_RRF_WEIGHT",
    "retrieval_phrase_rrf_weight": "RETRIEVAL_PHRASE_RRF_WEIGHT",
    "retrieval_full_text_rrf_weight": "RETRIEVAL_FULL_TEXT_RRF_WEIGHT",
    "retrieval_exact_rrf_weight": "RETRIEVAL_EXACT_RRF_WEIGHT",
    "retrieval_trigram_rrf_weight": "RETRIEVAL_TRIGRAM_RRF_WEIGHT",
    "retrieval_context_limit": "RETRIEVAL_CONTEXT_LIMIT",
    "retrieval_neighbor_window": "RETRIEVAL_NEIGHBOR_WINDOW",
    "retrieval_per_document_limit": "RETRIEVAL_PER_DOCUMENT_LIMIT",
    "reranker_candidates": "RERANKER_CANDIDATES",
    "reranker_weight": "RERANKER_WEIGHT",
    "evidence_weak_score_cap": "EVIDENCE_WEAK_SCORE_CAP",
    "evidence_min_citation_score": "EVIDENCE_MIN_CITATION_SCORE",
    "low_confidence_threshold": "LOW_CONFIDENCE_THRESHOLD",
}
_SECRET_MARKERS = ("key", "secret", "token", "password", "authorization", "cookie")
_MAX_TRIALS = 64
_MAX_VALUES_PER_PARAMETER = 32


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


class TuningService:
    """Run a small deterministic parameter grid without changing process config."""

    tunable_fields = TUNABLE_ENV

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: Any | None = None,
        retrieval_service: Any | None = None,
        context_builder: ContextBuilder | None = None,
        evaluator: Callable[[dict[str, Any], list[dict[str, Any]], Settings], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory or getattr(retrieval_service, "session_factory", None)
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder or ContextBuilder(settings)
        self.evaluator = evaluator

    @classmethod
    def allowed_grid(cls, grid: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
        if not isinstance(grid, Mapping) or not grid:
            raise ValueError("tuning grid must contain at least one parameter")
        normalized: dict[str, list[Any]] = {}
        trial_count = 1
        for name, values in sorted(grid.items(), key=lambda item: str(item[0])):
            field = str(name)
            if field not in cls.tunable_fields or any(marker in field.casefold() for marker in _SECRET_MARKERS):
                raise ValueError(f"{field} is not tunable")
            if not isinstance(values, list) or not values:
                raise ValueError(f"invalid values for {field}")
            if len(values) > _MAX_VALUES_PER_PARAMETER:
                raise ValueError("too many trials")
            deduplicated: list[Any] = []
            for value in values:
                try:
                    probe = Settings.model_validate({field: value})
                    checked = getattr(probe, field)
                except Exception as exc:
                    raise ValueError(f"invalid value for {field}") from None
                if isinstance(checked, float) and not math.isfinite(checked):
                    raise ValueError(f"invalid value for {field}")
                if checked not in deduplicated:
                    deduplicated.append(checked)
            trial_count *= len(deduplicated)
            if trial_count > _MAX_TRIALS:
                raise ValueError("too many trials")
            normalized[field] = deduplicated
        return normalized

    @staticmethod
    def score_metrics(metrics: Mapping[str, Any], latency_ms: float) -> tuple[float, dict[str, float]]:
        """Score retrieval quality with explicit positive and negative terms."""

        source = (
            0.30 * _safe_number(metrics.get("hit_at_1"))
            + 0.12 * _safe_number(metrics.get("hit_at_3"))
            + 0.08 * _safe_number(metrics.get("hit_at_5"))
        )
        evidence = (
            0.22 * _safe_number(metrics.get("evidence_hit_at_1"))
            + 0.10 * _safe_number(metrics.get("evidence_hit_at_3"))
            + 0.05 * _safe_number(metrics.get("evidence_hit_at_5"))
        )
        unknown = 0.08 * (1.0 - _safe_number(metrics.get("unknown_false_positive_rate")))
        false_rejection = 0.05 * (1.0 - _safe_number(metrics.get("answerable_false_rejection_rate")))
        latency_penalty = min(0.05, max(0.0, _safe_number(latency_ms)) / 10000.0)
        components = {
            "source": source,
            "evidence": evidence,
            "unknown": unknown,
            "false_rejection": false_rejection,
            "latency_penalty": latency_penalty,
        }
        return source + evidence + unknown + false_rejection - latency_penalty, components

    def _trial_settings(self, params: dict[str, Any]) -> Settings:
        # Values have already passed Settings validation in allowed_grid. A
        # model copy keeps secrets and unrelated settings in memory only.
        return self.settings.model_copy(update=params)

    def _evaluate_default(
        self,
        params: dict[str, Any],
        cases: list[dict[str, Any]],
        trial_settings: Settings,
        limit: int,
    ) -> dict[str, Any]:
        if self.session_factory is None:
            raise ValueError("a session factory is required for live tuning")
        provider = getattr(self.retrieval_service, "embedding_provider", None)
        if provider is None:
            provider = create_embedding_provider(trial_settings)
        retrieval = RetrievalService(
            self.session_factory,
            settings=trial_settings,
            embedding_provider=provider,
        )
        from scripts.evaluate_retrieval import evaluate_cases

        return evaluate_cases(
            retrieval,
            ContextBuilder(trial_settings),
            cases,
            limit=limit,
            confidence_threshold=trial_settings.low_confidence_threshold,
            profile="ours",
            dataset={"name": "tuning"},
            settings_overrides=params,
        )

    def run(
        self,
        cases_path: Path,
        grid: Mapping[str, list[Any]],
        limit_cases: int | None = None,
        *,
        limit: int = 5,
    ) -> dict[str, Any]:
        from scripts.evaluate_retrieval import load_cases

        if limit < 1 or limit > 30:
            raise ValueError("limit must be between 1 and 30")
        path = Path(cases_path)
        cases = load_cases(path)
        if limit_cases is not None:
            if limit_cases < 1:
                raise ValueError("limit_cases must be positive")
            cases = cases[:limit_cases]
        normalized = self.allowed_grid(grid)
        keys = list(normalized)
        trials: list[dict[str, Any]] = []
        for values in itertools.product(*(normalized[key] for key in keys)):
            params = dict(zip(keys, values))
            trial_settings = self._trial_settings(params)
            started = time.perf_counter()
            status = "completed"
            error: str | None = None
            report: dict[str, Any] = {}
            try:
                evaluator = self.evaluator or (
                    lambda p, c, s: self._evaluate_default(p, c, s, limit)
                )
                report = evaluator(params, cases, trial_settings) or {}
                metrics = report.get("metrics", {})
                if not isinstance(metrics, Mapping):
                    raise ValueError("evaluator returned invalid metrics")
            except Exception as exc:
                status = "error"
                error = f"evaluation failed ({type(exc).__name__})"
                metrics = {}
            latency_ms = (time.perf_counter() - started) * 1000.0
            score, components = self.score_metrics(metrics, latency_ms)
            trial: dict[str, Any] = {
                "params": params,
                "status": status,
                "metrics": dict(metrics),
                "latency_ms": round(latency_ms, 3),
                "score": round(score, 8),
                "score_components": components,
            }
            if error:
                trial["error"] = error
            trials.append(trial)

        successful_trials = [trial for trial in trials if trial["status"] == "completed"]
        ranked = sorted(
            successful_trials,
            key=lambda item: (
                float(item["score"]),
                json.dumps(item["params"], ensure_ascii=False, sort_keys=True),
            ),
            reverse=True,
        )
        best = None
        if ranked:
            best = dict(ranked[0])
            best["env_lines"] = [
                f"{self.tunable_fields[name]}={_format_env_value(best['params'][name])}"
                for name in keys
            ]
        return {
            "dataset": {"name": path.stem, "case_count": len(cases)},
            "trial_count": len(trials),
            "score_formula": {
                "positive": "0.30*hit_at_1 + 0.12*hit_at_3 + 0.08*hit_at_5 + 0.22*evidence_hit_at_1 + 0.10*evidence_hit_at_3 + 0.05*evidence_hit_at_5 + 0.08*(1-unknown_false_positive_rate) + 0.05*(1-answerable_false_rejection_rate)",
                "latency_penalty": "min(0.05, latency_ms/10000)",
            },
            "best": best,
            **({"error": "all tuning trials failed"} if not ranked else {}),
            "trials": trials,
        }


__all__ = ["TUNABLE_ENV", "TuningService"]
