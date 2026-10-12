"""Embedding provider adapters.

The hashing provider is deterministic and intentionally simple.  It is useful
for local smoke tests and makes the whole RAG pipeline runnable without a
remote credential; the OpenAI-compatible provider is the production option.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol

from .config import Settings, get_settings


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in input order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query text."""


_TOKEN_RE = re.compile(r"[\w]+|[^\W\s]", re.UNICODE)


class HashingEmbeddingProvider:
    """A deterministic feature-hashing embedding adapter.

    It hashes word and character n-gram features into a fixed-size vector and
    L2-normalizes the result.  Including character n-grams gives technical
    identifiers and mixed Chinese/English text a useful lexical signal while
    retaining zero external dependencies.
    """

    def __init__(self, dimensions: int = 384, *, seed: str = "rag-mvp-v1") -> None:
        if dimensions < 8:
            raise ValueError("embedding dimensions must be at least 8")
        self.dimensions = dimensions
        self.seed = seed

    def _features(self, text: str) -> list[tuple[str, float]]:
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return []
        tokens = _TOKEN_RE.findall(normalized)
        features: list[tuple[str, float]] = [(f"w:{token}", 1.0) for token in tokens]
        # Character trigrams retain information for identifiers such as
        # ``pgvector`` and for languages without whitespace segmentation.
        compact = normalized.replace(" ", "")
        features.extend((f"c:{compact[index:index + 3]}", 0.35) for index in range(max(0, len(compact) - 2)))
        return features

    def _index_sign(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(
            f"{self.seed}:{feature}".encode("utf-8"), digest_size=16
        ).digest()
        index = int.from_bytes(digest[:8], "big") % self.dimensions
        sign = 1.0 if digest[8] & 1 else -1.0
        # A small deterministic weight variation avoids excessive collisions
        # among short feature strings while preserving cosine behavior.
        weight = 0.75 + digest[9] / 255.0 * 0.5
        return index, sign * weight

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for feature, feature_weight in self._features(text):
                index, signed_weight = self._index_sign(feature)
                vector[index] += signed_weight * feature_weight
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


# A concise alias is convenient for callers and keeps the provider name clear
# in configuration-independent code.
HashingEmbedding = HashingEmbeddingProvider


class OpenAICompatibleEmbeddingProvider:
    """Embedding adapter for OpenAI-compatible REST APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("an embedding API key is required for openai_compatible")
        if not base_url:
            raise ValueError("embedding base URL is required for openai_compatible")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is optional in tests
                raise RuntimeError("install the openai package to use remote embeddings") from exc
            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "timeout": self.timeout,
            }
            self._client = OpenAI(**kwargs)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("embedding input must contain only strings")
        # Do not pass ``dimensions`` to every gateway.  OpenAI supports that
        # optional field, but many OpenAI-compatible embedding services (for
        # example BGE gateways) reject it; we validate the returned length
        # instead and keep the database schema explicit via Settings.
        request: dict[str, Any] = {"model": self.model, "input": texts}
        response = self.client.embeddings.create(**request)
        data = list(getattr(response, "data", []) or [])
        if len(data) != len(texts):
            raise RuntimeError(
                f"embedding API returned {len(data)} vectors for {len(texts)} inputs"
            )
        # OpenAI-compatible services usually preserve order, but the API
        # contract includes ``index`` and some gateways return sorted data.
        def field(item: Any, name: str, default: Any = None) -> Any:
            if isinstance(item, dict):
                return item.get(name, default)
            return getattr(item, name, default)

        data.sort(key=lambda item: field(item, "index", 0))
        vectors = [list(map(float, field(item, "embedding", []))) for item in data]
        if self.dimensions is not None and any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError("embedding API returned a vector with an unexpected dimension")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0]


def create_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Build the configured provider, defaulting to local hashing embeddings."""

    settings = settings or get_settings()
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(settings.embedding_dimensions)
    if not settings.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY must be set when EMBEDDING_PROVIDER=openai_compatible")
    return OpenAICompatibleEmbeddingProvider(
        base_url=settings.embedding_base_url or settings.llm_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout=settings.llm_timeout_seconds,
    )


# Compatibility aliases make the adapter easy to discover from a REPL and keep
# naming flexible for callers migrating from the Gitee reference project.
HashEmbeddingProvider = HashingEmbeddingProvider
build_embedding_provider = create_embedding_provider


__all__ = [
    "EmbeddingProvider",
    "HashingEmbedding",
    "HashingEmbeddingProvider",
    "HashEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "build_embedding_provider",
    "create_embedding_provider",
]
