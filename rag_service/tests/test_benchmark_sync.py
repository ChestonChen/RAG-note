from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.sync_gitee_benchmark import sync_cases_to_downloads, sync_gitee_benchmark


class FakeResponse:
    def __init__(self, content: bytes, *, url: str = "https://gitee.test/eval_cases.json") -> None:
        self.content = content
        self.url = url

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.requested.append(url)
        return self.responses[url]


class FailingClient:
    def get(self, url: str, **kwargs: object) -> object:
        raise RuntimeError("Authorization: Bearer should-not-leak")


def test_benchmark_sync_maps_only_expected_files() -> None:
    cases = [{"question": "Q", "expected_files": ["C++\\vector复习.pdf"]}]

    result = sync_cases_to_downloads(
        cases,
        "https://gitee.com/yegelute/note-repository/raw/master/NoteRepository/core/pdf",
    )

    assert result == ["C++/vector复习.pdf"]


def test_benchmark_sync_rejects_paths_outside_expected_files() -> None:
    cases = [{"question": "Q", "expected_files": ["../secrets.pdf", "/absolute.pdf"]}]

    assert sync_cases_to_downloads(cases, "https://gitee.test/pdf") == []


def test_benchmark_sync_redacts_network_errors(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"benchmark request failed") as error:
        sync_gitee_benchmark(
            "https://gitee.test/eval_cases.json",
            tmp_path / "downloads",
            tmp_path / "cases.jsonl",
            FailingClient(),
        )

    assert "should-not-leak" not in str(error.value)


def test_benchmark_sync_writes_normalized_cases_and_manifest(tmp_path: Path) -> None:
    source_url = "https://gitee.test/repo/rag_service/eval_cases.json"
    cases = [{
        "question": "What?",
        "expected_files": ["C++\\vector.pdf", "C++/vector.pdf"],
        "expected_terms": ["vector"],
        "should_use_web": False,
    }]
    pdf_url = "https://gitee.test/repo/NoteRepository/core/pdf/C++/vector.pdf"
    payload = b"pdf bytes"
    client = FakeClient({
        source_url: FakeResponse(json.dumps(cases).encode("utf-8")),
        pdf_url: FakeResponse(payload),
    })
    cases_path = tmp_path / "cases.jsonl"
    manifest_path = tmp_path / "manifest.json"

    result = sync_gitee_benchmark(
        source_url,
        tmp_path / "downloads",
        cases_path,
        client,
        manifest_path=manifest_path,
        pdf_source_url="https://gitee.test/repo/NoteRepository/core/pdf",
    )

    assert result["case_count"] == 1
    assert result["download_count"] == 1
    assert client.requested == [source_url, pdf_url]
    normalized = json.loads(cases_path.read_text(encoding="utf-8").strip())
    assert normalized["id"] == "case-001"
    assert normalized["relevant_sources"] == ["gitee-benchmark/C++/vector.pdf"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][0]
    assert entry["url"] == pdf_url
    assert entry["local_path"] == "C++/vector.pdf"
    assert entry["size"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["source_dataset_url"] == source_url
    assert manifest["source_dataset_version"]

