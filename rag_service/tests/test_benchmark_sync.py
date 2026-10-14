from __future__ import annotations

import hashlib
import json
import base64
from pathlib import Path
from urllib.parse import quote

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
    cases = [{"question": "Q", "expected_files": [
        "../secrets.pdf", "/absolute.pdf", "README.md", "data.json",
        "report.pdf?token=secret", "report.pdf#page=2", "notes.txt",
    ]}]

    assert sync_cases_to_downloads(cases, "https://gitee.test/pdf") == []


def test_benchmark_sync_removes_credentials_from_manifest_urls(tmp_path: Path) -> None:
    source_url = "https://user:pass@gitee.test/repo/eval.json?access_token=secret&ref=main"
    pdf_root = "https://reader:secret@gitee.test/repo/pdf?token=hidden&ref=main"
    pdf_url = "https://reader:secret@gitee.test/repo/pdf/C++/vector.pdf"
    clean_pdf_url = "https://gitee.test/repo/pdf/C++/vector.pdf"
    client = FakeClient({
        source_url: FakeResponse(json.dumps([{"question": "Q", "expected_files": ["C++/vector.pdf"]}]).encode()),
        pdf_url: FakeResponse(b"pdf bytes"),
    })

    sync_gitee_benchmark(
        source_url, tmp_path / "downloads", tmp_path / "cases.jsonl", client,
        manifest_path=tmp_path / "manifest.json", pdf_source_url=pdf_root,
    )

    manifest_text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["source_dataset_url"] == "https://gitee.test/repo/eval.json?ref=main"
    assert manifest["files"][0]["url"] == clean_pdf_url
    assert "secret" not in manifest_text
    assert "hidden" not in manifest_text


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


def test_benchmark_sync_replaces_truncated_gitee_pdf_from_contents_api(tmp_path: Path) -> None:
    source_url = "https://gitee.com/yegelute/note-repository/raw/master/NoteRepository/rag_service/eval_cases.json"
    pdf_root = "https://gitee.com/yegelute/note-repository/raw/master/NoteRepository/core/pdf"
    relative_path = "Linux/传输层协议TCP.pdf"
    encoded_path = quote(relative_path, safe="/+-_.~")
    pdf_url = f"{pdf_root}/{encoded_path}"
    contents_url = (
        "https://gitee.com/api/v5/repos/yegelute/note-repository/contents/"
        f"NoteRepository/core/pdf/{quote(relative_path, safe='/')}"
    )
    complete_pdf = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
    cases = [{"question": "What?", "expected_files": [relative_path]}]
    client = FakeClient(
        {
            source_url: FakeResponse(json.dumps(cases).encode("utf-8")),
            pdf_url: FakeResponse(b"%PDF-1.7\n" + b"x" * (10 * 1024 * 1024 - 9)),
            contents_url: FakeResponse(
                json.dumps({"content": base64.b64encode(complete_pdf).decode("ascii")}).encode("utf-8")
            ),
        }
    )

    result = sync_gitee_benchmark(
        source_url,
        tmp_path / "downloads",
        tmp_path / "cases.jsonl",
        client,
        pdf_source_url=pdf_root,
    )

    assert result["download_count"] == 1
    assert client.requested == [source_url, pdf_url, contents_url]
    assert (tmp_path / "downloads" / "Linux" / "传输层协议TCP.pdf").read_bytes() == complete_pdf


def test_benchmark_sync_rejects_truncated_contents_payload_and_uses_full_blob(tmp_path: Path) -> None:
    source_url = "https://gitee.com/yegelute/note-repository/raw/master/NoteRepository/rag_service/eval_cases.json"
    pdf_root = "https://gitee.com/yegelute/note-repository/raw/master/NoteRepository/core/pdf"
    relative_path = "Linux/large.pdf"
    pdf_url = f"{pdf_root}/{relative_path}"
    contents_url = "https://gitee.com/api/v5/repos/yegelute/note-repository/contents/NoteRepository/core/pdf/Linux/large.pdf"
    blob_url = "https://gitee.com/api/v5/repos/yegelute/note-repository/git/blobs/blob-sha"
    complete_pdf = b"%PDF-1.7\n" + b"x" * (10 * 1024 * 1024) + b"\n%%EOF\n"
    truncated = complete_pdf[: 10 * 1024 * 1024]
    cases = [{"question": "What?", "expected_files": [relative_path]}]
    client = FakeClient({
        source_url: FakeResponse(json.dumps(cases).encode()),
        pdf_url: FakeResponse(truncated),
        contents_url: FakeResponse(json.dumps({
            "content": base64.b64encode(truncated).decode("ascii"),
            "size": len(truncated),
            "sha": "blob-sha",
        }).encode()),
        blob_url: FakeResponse(json.dumps({
            "content": base64.b64encode(complete_pdf).decode("ascii"),
            "size": len(complete_pdf),
        }).encode()),
    })

    result = sync_gitee_benchmark(source_url, tmp_path / "downloads", tmp_path / "cases.jsonl", client, pdf_source_url=pdf_root)

    assert result["download_count"] == 1
    assert client.requested == [source_url, pdf_url, contents_url, blob_url]
    assert (tmp_path / "downloads" / "Linux" / "large.pdf").read_bytes() == complete_pdf

