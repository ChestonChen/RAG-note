"""Embedding provider adapters.

The hashing provider is deterministic and intentionally simple.  It is useful
for local smoke tests and makes the whole RAG pipeline runnable without a
remote credential; the OpenAI-compatible provider is the production option.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from pathlib import Path
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
        self.provider_name = "hashing"
        self.model = "feature-hashing"

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
        self.provider_name = "openai_compatible"
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


class SentenceTransformersEmbeddingProvider:
    """Local sentence-transformers embedding adapter.

    The optional dependency and model are loaded only when the first batch is
    embedded. Models are shared by path/device within the process so ingestion
    and query handling do not each allocate a separate copy.
    """

    _model_cache: dict[tuple[str, str], Any] = {}
    _model_lock = threading.Lock()

    def __init__(
        self,
        *,
        model_path: str,
        dimensions: int | None = None,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if not model_path or not model_path.strip():
            raise ValueError("a local sentence-transformers model path is required")
        if batch_size < 1:
            raise ValueError("embedding batch size must be at least 1")
        if dimensions is not None and dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        if not device or not device.strip():
            raise ValueError("embedding device must not be empty")
        self.model_path = model_path
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.device = device
        self.provider_name = "sentence_transformers"
        self.model = model_path

    @property
    def _cache_key(self) -> tuple[str, str]:
        return (str(Path(self.model_path).expanduser().resolve()), self.device)

    def _get_model(self) -> Any:
        key = self._cache_key
        with self._model_lock:
            if key in self._model_cache:
                return self._model_cache[key]
            model_dir = Path(key[0])
            if not model_dir.exists():
                raise ValueError(f"sentence-transformers model path does not exist: {self.model_path}")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - exercised with test injection
                raise RuntimeError(
                    "sentence-transformers is required for EMBEDDING_PROVIDER=sentence_transformers; "
                    "install the sentence-transformers package"
                ) from exc
            try:
                model = SentenceTransformer(str(model_dir), device=self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load sentence-transformers model from {self.model_path}: {exc}"
                ) from exc
            self._model_cache[key] = model
            return model

    @staticmethod
    def _as_vectors(encoded: Any) -> list[list[float]]:
        values = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        try:
            return [
                [float(value) for value in (row.tolist() if hasattr(row, "tolist") else row)]
                for row in values
            ]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("sentence-transformers returned invalid embedding vectors") from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("embedding input must contain only strings")
        model = self._get_model()
        try:
            encoded = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RuntimeError(f"sentence-transformers failed to encode input: {exc}") from exc
        vectors = self._as_vectors(encoded)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"sentence-transformers returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        if self.dimensions is not None and any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError("sentence-transformers returned an unexpected vector dimension")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0]


def create_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Build the configured provider, defaulting to local hashing embeddings."""

    settings = settings or get_settings()
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(settings.embedding_dimensions)
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformersEmbeddingProvider(
            model_path=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
        )
    if not settings.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY must be set when EMBEDDING_PROVIDER=openai_compatible")
    if not settings.embedding_base_url:
        raise ValueError("EMBEDDING_BASE_URL must be set when EMBEDDING_PROVIDER=openai_compatible")
    return OpenAICompatibleEmbeddingProvider(
        base_url=settings.embedding_base_url,
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
    "SentenceTransformersEmbeddingProvider",
    "build_embedding_provider",
    "create_embedding_provider",
]
