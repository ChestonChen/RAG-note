import sys
import types
import builtins

import pytest

_ORIGINAL_IMPORT = builtins.__import__


def _fake_sentence_transformers(monkeypatch, model_factory):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = model_factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_create_provider_dispatches_sentence_transformers(tmp_path):
    from rag_service.app.config import Settings
    from rag_service.app.embeddings import (
        SentenceTransformersEmbeddingProvider,
        create_embedding_provider,
    )

    settings = Settings(
        embedding_provider="sentence_transformers",
        embedding_model=str(tmp_path),
        embedding_dimensions=8,
        embedding_batch_size=7,
    )

    provider = create_embedding_provider(settings)

    assert isinstance(provider, SentenceTransformersEmbeddingProvider)
    assert provider.model_path == str(tmp_path)
    assert provider.batch_size == 7
    assert provider.device == "cpu"


def test_sentence_transformers_provider_uses_encode_options_and_caches_model(
    monkeypatch, tmp_path
):
    from rag_service.app.embeddings import SentenceTransformersEmbeddingProvider

    calls = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return [[1, 2, 3] for _ in texts]

    models = []

    def factory(path, device):
        models.append((path, device))
        return FakeModel()

    _fake_sentence_transformers(monkeypatch, factory)
    SentenceTransformersEmbeddingProvider._model_cache.clear()
    provider = SentenceTransformersEmbeddingProvider(
        model_path=str(tmp_path), dimensions=3, batch_size=7
    )

    assert provider.embed_texts(["a", "b"]) == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert provider.embed_query("query") == [1.0, 2.0, 3.0]
    assert models == [(str(tmp_path), "cpu")]
    assert calls == [
        (
            ["a", "b"],
            {
                "batch_size": 7,
                "normalize_embeddings": True,
                "show_progress_bar": False,
                "convert_to_numpy": True,
            },
        ),
        (
            ["query"],
            {
                "batch_size": 7,
                "normalize_embeddings": True,
                "show_progress_bar": False,
                "convert_to_numpy": True,
            },
        )
    ]


def test_sentence_transformers_provider_handles_empty_input(tmp_path):
    from rag_service.app.embeddings import SentenceTransformersEmbeddingProvider

    provider = SentenceTransformersEmbeddingProvider(model_path=str(tmp_path), dimensions=3)

    assert provider.embed_texts([]) == []


def test_sentence_transformers_provider_reports_missing_model_path(tmp_path):
    from rag_service.app.embeddings import SentenceTransformersEmbeddingProvider

    provider = SentenceTransformersEmbeddingProvider(
        model_path=str(tmp_path / "missing"), dimensions=3
    )

    with pytest.raises(ValueError, match="model path does not exist"):
        provider.embed_texts(["hello"])


def test_sentence_transformers_provider_reports_missing_dependency(monkeypatch, tmp_path):
    from rag_service.app.embeddings import SentenceTransformersEmbeddingProvider

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr("builtins.__import__", _reject_sentence_transformers)
    provider = SentenceTransformersEmbeddingProvider(model_path=str(tmp_path), dimensions=3)

    with pytest.raises(RuntimeError, match="sentence-transformers is required"):
        provider.embed_texts(["hello"])


def _reject_sentence_transformers(name, *args, **kwargs):
    if name == "sentence_transformers":
        raise ImportError("test dependency missing")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


def test_sentence_transformers_provider_rejects_unexpected_dimension(monkeypatch, tmp_path):
    from rag_service.app.embeddings import SentenceTransformersEmbeddingProvider

    class FakeModel:
        def encode(self, texts, **kwargs):
            return [[1, 2] for _ in texts]

    _fake_sentence_transformers(monkeypatch, lambda path, device: FakeModel())
    SentenceTransformersEmbeddingProvider._model_cache.clear()
    provider = SentenceTransformersEmbeddingProvider(model_path=str(tmp_path), dimensions=3)

    with pytest.raises(RuntimeError, match="unexpected vector dimension"):
        provider.embed_texts(["hello"])


def test_remote_provider_configuration_keeps_endpoint_and_model_without_secret_in_repr():
    from rag_service.app.config import Settings
    from rag_service.app.embeddings import OpenAICompatibleEmbeddingProvider, create_embedding_provider

    provider = create_embedding_provider(
        Settings(
            embedding_provider="openai_compatible",
            embedding_base_url="https://embedding.example/v1",
            embedding_api_key="super-secret",
            embedding_model="BAAI/bge-m3",
            embedding_dimensions=1024,
        )
    )

    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.base_url == "https://embedding.example/v1"
    assert provider.model == "BAAI/bge-m3"
    assert "super-secret" not in repr(provider)


def test_remote_provider_rejects_returned_dimension_mismatch():
    from rag_service.app.embeddings import OpenAICompatibleEmbeddingProvider

    class FakeEmbeddings:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(index=0, embedding=[0.1, 0.2])]
            )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://embedding.example/v1",
        api_key="secret",
        model="BAAI/bge-m3",
        dimensions=4,
        client=types.SimpleNamespace(embeddings=FakeEmbeddings()),
    )

    with pytest.raises(RuntimeError, match="unexpected dimension"):
        provider.embed_texts(["probe"])


def test_remote_provider_redacts_exception_details():
    from rag_service.app.embeddings import OpenAICompatibleEmbeddingProvider

    class ExplodingEmbeddings:
        def create(self, **kwargs):
            raise RuntimeError(
                "authorization=secret-token https://provider.example/v1?key=secret-key"
            )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://embedding.example/v1",
        api_key="secret-token",
        model="BAAI/bge-m3",
        dimensions=4,
        client=types.SimpleNamespace(embeddings=ExplodingEmbeddings()),
    )

    with pytest.raises(RuntimeError) as error:
        provider.embed_texts(["probe"])
    assert "RuntimeError" in str(error.value)
    assert "secret-token" not in str(error.value)
    assert "secret-key" not in str(error.value)
    assert "provider.example" not in str(error.value)
