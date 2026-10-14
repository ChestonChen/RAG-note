"""Download only the PDFs referenced by the public note-repository benchmark."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URL = (
    "https://gitee.com/yegelute/note-repository/raw/master/"
    "NoteRepository/rag_service/eval_cases.json"
)
DEFAULT_PDF_SOURCE_URL = (
    "https://gitee.com/yegelute/note-repository/raw/master/NoteRepository/core/pdf"
)


def _safe_relative_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return None
    parts = PurePosixPath(normalized).parts
    if any(part in {"", ".", ".."} for part in parts) or ".." in parts:
        return None
    if not parts[-1].casefold().endswith(".pdf"):
        return None
    return "/".join(parts)


_SECRET_QUERY_MARKERS = ("key", "secret", "token", "auth", "password", "credential", "signature")


def _sanitize_public_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return value
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        try:
            if parsed.port is not None:
                netloc += f":{parsed.port}"
        except ValueError:
            return None
        query = [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not any(marker in key.casefold() for marker in _SECRET_QUERY_MARKERS)
        ]
        return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query), ""))
    except ValueError:
        return None


def sync_cases_to_downloads(cases: Iterable[dict[str, Any]], source_url: str) -> list[str]:
    """Return unique, safe PDF paths from ``expected_files`` in source order.

    ``source_url`` is accepted as part of the public interface and deliberately
    does not influence the returned paths; callers must still build download
    URLs from this allow-list rather than accepting arbitrary URL input.
    """

    del source_url
    paths: list[str] = []
    seen: set[str] = set()
    for case in cases:
        expected = case.get("expected_files", [])
        if not isinstance(expected, list):
            continue
        for value in expected:
            path = _safe_relative_path(value)
            if path is None or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def _response_content(response: Any) -> bytes:
    content = getattr(response, "content", response)
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    raise ValueError("response did not contain bytes")


def _looks_like_complete_pdf(payload: bytes) -> bool:
    """Recognize the cheap PDF markers needed to detect truncated raw files."""

    if not payload.startswith(b"%PDF"):
        return False
    # The EOF marker is normally in the final 1 KiB, but a slightly wider
    # window tolerates small trailing metadata while keeping this inexpensive.
    return b"%%EOF" in payload[-4096:]


def _contents_payload(client: Any, fallback_url: str) -> tuple[bytes, int | None, str | None] | None:
    """Decode a Gitee contents response and retain integrity metadata."""

    try:
        fallback = _client_get(client, fallback_url)
        fallback.raise_for_status()
        payload = json.loads(_response_content(fallback).decode("utf-8"))
        encoded = payload.get("content") if isinstance(payload, dict) else None
        if isinstance(encoded, str):
            size = payload.get("size")
            try:
                size = int(size) if size is not None else None
            except (TypeError, ValueError):
                size = None
            sha = payload.get("sha") if isinstance(payload.get("sha"), str) else None
            return base64.b64decode(encoded, validate=False), size, sha
    except Exception:
        return None
    return None


def _gitee_blob_url(root: str, sha: str | None) -> str | None:
    parsed = urlsplit(root)
    parts = parsed.path.strip("/").split("/")
    if parsed.hostname != "gitee.com" or len(parts) < 5 or parts[2] != "raw" or not sha:
        return None
    owner, repo = parts[0], parts[1]
    return f"https://gitee.com/api/v5/repos/{quote(owner)}/{quote(repo)}/git/blobs/{quote(sha)}"


def _fallback_payload(
    client: Any,
    fallback_url: str | None,
    root: str,
    *,
    expect_pdf: bool = False,
) -> bytes | None:
    if not fallback_url:
        return None
    decoded = _contents_payload(client, fallback_url)
    if decoded is None:
        return None
    payload, declared_size, sha = decoded
    if (
        (declared_size is not None and len(payload) != declared_size)
        or (expect_pdf and not _looks_like_complete_pdf(payload))
    ):
        blob_url = _gitee_blob_url(root, sha)
        if blob_url:
            blob = _contents_payload(client, blob_url)
            if blob is not None:
                payload, blob_size, _ = blob
                if blob_size is not None and len(payload) != blob_size:
                    return None
                declared_size = blob_size or declared_size
    if declared_size is not None and len(payload) != declared_size:
        return None
    return payload


def _client_get(client: Any, url: str) -> Any:
    try:
        return client.get(url, timeout=60, follow_redirects=True)
    except TypeError:
        # Small test doubles and requests-like clients may not expose the
        # httpx-only redirect option.
        return client.get(url, timeout=60)


def _request(
    client: Any,
    url: str,
    *,
    fallback_url: str | None = None,
    expect_pdf: bool = False,
) -> bytes:
    try:
        response = _client_get(client, url)
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        raw_payload = _response_content(response)
    except Exception as exc:
        # Gitee serves some large public LFS files with a login-only raw URL.
        # Its public contents endpoint exposes the same file as base64; use it
        # only for the already allow-listed path and never pass credentials.
        if fallback_url:
            replacement = _fallback_payload(client, fallback_url, url, expect_pdf=expect_pdf)
            if replacement is not None and (not expect_pdf or _looks_like_complete_pdf(replacement)):
                return replacement
        # Never include response text, URLs with query strings, headers, or
        # exception messages because those may contain credentials.
        raise RuntimeError(f"benchmark request failed ({type(exc).__name__})") from None
    if not expect_pdf or not fallback_url or _looks_like_complete_pdf(raw_payload):
        return raw_payload
    replacement = _fallback_payload(client, fallback_url, url, expect_pdf=expect_pdf)
    if replacement is not None and _looks_like_complete_pdf(replacement):
        return replacement
    # A non-Gitee test double may use arbitrary bytes for a PDF. Preserve
    # that compatibility; real Gitee URLs always have a contents fallback.
    if urlsplit(fallback_url).hostname != "gitee.com":
        return raw_payload
    raise RuntimeError("downloaded PDF failed integrity check")


def _join_url(root: str, relative_path: str) -> str:
    parsed = urlsplit(root)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source URL must be an http(s) URL")
    encoded = quote(relative_path, safe="/+-_.~")
    path = parsed.path.rstrip("/") + "/" + encoded
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _gitee_contents_url(root: str, relative_path: str) -> str | None:
    parsed = urlsplit(root)
    parts = parsed.path.strip("/").split("/")
    if parsed.hostname != "gitee.com" or len(parts) < 5 or parts[2] != "raw":
        return None
    owner, repo = parts[0], parts[1]
    repository_path = "/".join(parts[4:] + relative_path.split("/"))
    return (
        f"https://gitee.com/api/v5/repos/{quote(owner)}/{quote(repo)}/contents/"
        f"{quote(repository_path, safe='/')}"
    )


def _dataset_version(response: Any, payload: bytes) -> str:
    headers = getattr(response, "headers", {}) or {}
    for key in ("x-commit-id", "x-gitee-commit", "etag", "last-modified"):
        value = headers.get(key) if hasattr(headers, "get") else None
        if value:
            return str(value).strip('"')
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _load_source_cases(payload: bytes) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"benchmark dataset is not valid JSON ({type(exc).__name__})") from None
    if isinstance(decoded, dict):
        decoded = decoded.get("cases", decoded.get("eval_cases", []))
    if not isinstance(decoded, list) or any(not isinstance(case, dict) for case in decoded):
        raise ValueError("benchmark dataset must be a JSON array of objects")
    return decoded


def _normalized_case(index: int, case: dict[str, Any], paths: set[str]) -> dict[str, Any]:
    question = str(case.get("question", "")).strip()
    if not question:
        raise ValueError(f"benchmark case {index} has no question")
    expected: list[str] = []
    seen_expected: set[str] = set()
    for item in case.get("expected_files", []):
        path = _safe_relative_path(item)
        if path in paths and path not in seen_expected:
            seen_expected.add(path)
            expected.append(path)
    terms = [str(term).strip() for term in case.get("expected_terms", []) if str(term).strip()]
    answerable = bool(expected)
    normalized: dict[str, Any] = {
        "id": str(case.get("id") or f"case-{index:03d}"),
        "question": question,
        "relevant_sources": [f"gitee-benchmark/{path}" for path in expected],
        "expected_files": expected,
        "expected_terms": terms,
        "should_use_web": bool(case.get("should_use_web", False)),
        "answerable": answerable,
    }
    if answerable and terms:
        normalized["evidence_groups"] = [[term] for term in terms]
    elif answerable:
        # Keep the generated file valid for the evaluator even when the source
        # dataset has no term annotation.
        normalized["evidence_groups"] = [[path.rsplit("/", 1)[-1]] for path in expected]
    return normalized


def sync_gitee_benchmark(
    source_url: str,
    destination: Path,
    cases_path: Path,
    client: object,
    *,
    manifest_path: Path | None = None,
    pdf_source_url: str | None = None,
) -> dict[str, object]:
    """Fetch and normalize the benchmark JSON plus only its referenced PDFs."""

    source_response: Any = None
    try:
        source_response = _client_get(client, source_url)
        raise_for_status = getattr(source_response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        source_payload = _response_content(source_response)
        source_cases = _load_source_cases(source_payload)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"benchmark request failed ({type(exc).__name__})") from None

    pdf_root = pdf_source_url or DEFAULT_PDF_SOURCE_URL
    relative_paths = sync_cases_to_downloads(source_cases, pdf_root)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for relative_path in relative_paths:
        target = (destination / Path(*relative_path.split("/"))).resolve()
        if destination not in target.parents:
            raise ValueError("benchmark path escapes destination")
        payload = _request(
            client,
            _join_url(pdf_root, relative_path),
            fallback_url=_gitee_contents_url(pdf_root, relative_path),
            expect_pdf=relative_path.casefold().endswith(".pdf"),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        files.append({
            "url": _sanitize_public_url(_join_url(pdf_root, relative_path)),
            "local_path": relative_path,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    normalized_cases = [_normalized_case(index, case, set(relative_paths)) for index, case in enumerate(source_cases, 1)]
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in normalized_cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": 1,
        "source_dataset_url": _sanitize_public_url(source_url),
        "source_dataset_version": _dataset_version(source_response, source_payload),
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path = manifest_path or cases_path.with_name("gitee_note_repository_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "case_count": len(normalized_cases),
        "download_count": len(files),
        "cases_path": str(cases_path),
        "manifest_path": str(manifest_path),
        "source_dataset_version": manifest["source_dataset_version"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--pdf-source-url", default=DEFAULT_PDF_SOURCE_URL)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "knowledge/gitee-benchmark")
    parser.add_argument("--output-cases", type=Path, default=PROJECT_ROOT / "eval/gitee_note_repository_cases.jsonl")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "eval/gitee_note_repository_manifest.json")
    args = parser.parse_args(argv)
    try:
        import httpx

        result = sync_gitee_benchmark(
            args.source_url,
            args.output_root,
            args.output_cases,
            httpx,
            manifest_path=args.manifest,
            pdf_source_url=args.pdf_source_url,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
