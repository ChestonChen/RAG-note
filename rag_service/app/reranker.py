"""Optional local CrossEncoder reranking with a fail-open boundary."""

from __future__ import annotations

import math
from typing import Any, Protocol


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score for every document, in input order."""


class CrossEncoderReranker:
    """Lazy sentence-transformers CrossEncoder adapter.

    Loading is deferred until the first query so the default service remains
    lightweight and the optional dependency/model cannot block startup.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        batch_size: int = 4,
        max_chars: int = 1200,
    ) -> None:
        self.model_name = model
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_chars = max(128, max_chars)
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is required for reranking") from exc
        self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, document[: self.max_chars]) for document in documents]
        model = self._get_model()
        try:
            values = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except TypeError:  # compatible with older CrossEncoder versions
            values = model.predict(pairs, batch_size=self.batch_size)
        if hasattr(values, "tolist"):
            values = values.tolist()
        scores = [float(value) for value in values]
        if len(scores) != len(documents):
            raise RuntimeError("reranker returned an unexpected number of scores")
        if any(not math.isfinite(value) for value in scores):
            raise RuntimeError("reranker returned non-finite scores")
        return scores


def create_reranker(settings: Any) -> Reranker | None:
    if not getattr(settings, "reranker_enabled", False):
        return None
    return CrossEncoderReranker(
        model=str(settings.reranker_model),
        device=str(settings.reranker_device),
        batch_size=int(settings.reranker_batch_size),
        max_chars=int(settings.reranker_max_chars),
    )


__all__ = ["CrossEncoderReranker", "Reranker", "create_reranker"]
