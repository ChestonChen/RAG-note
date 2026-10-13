"""Deterministic text used by lexical retrieval and PostgreSQL FTS indexes."""

from __future__ import annotations

import re
from collections.abc import Iterable


# Keep technical identifiers together when they contain punctuation that a
# normal word tokenizer would discard (for example ``C++``, ``C#`` and
# ``std::vector``).  Boundaries prevent a token from being extracted from the
# middle of a larger identifier.
_TECHNICAL_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*(?:(?:::|[+#./-])[A-Za-z0-9_+#./-]*)*(?![A-Za-z0-9_])"
)
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_WHITESPACE_RE = re.compile(r"\s+")


def technical_identifier_tokens(text: str) -> list[str]:
    """Return stable, case-folded technical tokens from ``text``."""

    tokens: list[str] = []
    for match in _TECHNICAL_IDENTIFIER_RE.finditer(text or ""):
        token = match.group(0).strip("./-")
        if token:
            tokens.append(token.casefold())
    return list(dict.fromkeys(tokens))


def deterministic_cjk_ngrams(text: str, sizes: Iterable[int] = (2, 3)) -> list[str]:
    """Return deterministic character n-grams for every contiguous CJK run."""

    terms: list[str] = []
    normalized_sizes = tuple(sorted({int(size) for size in sizes if int(size) > 0}))
    for run in _CJK_RUN_RE.findall(text or ""):
        for size in normalized_sizes:
            terms.extend(run[index : index + size] for index in range(len(run) - size + 1))
    return list(dict.fromkeys(terms))


def normalize_search_text(text: str) -> str:
    """Collapse whitespace while preserving searchable punctuation and CJK."""

    return _WHITESPACE_RE.sub(" ", (text or "").casefold()).strip()


def build_full_text_query(text: str) -> str:
    """Build the bound query consumed by PostgreSQL's simple FTS parser.

    ``search_text`` stores the readable source followed by technical
    identifiers and deterministic CJK n-grams.  PostgreSQL must receive the
    same derived terms for Chinese questions; retaining the normalized ASCII
    phrase skeleton also keeps ASCII phrases and identifiers searchable.  The returned
    value is data for a bound SQL parameter, never a SQL fragment.
    """

    # A contiguous CJK run is not a stable PostgreSQL ``simple`` token: a
    # source may contain the same concepts separated by another character.
    # Query only its deterministic n-grams, while retaining the ASCII phrase
    # skeleton for ordinary words and technical identifiers.
    ascii_phrase = normalize_search_text(_CJK_RUN_RE.sub(" ", text or ""))
    if not ascii_phrase and not _CJK_RUN_RE.search(text or ""):
        return ""
    identifiers = technical_identifier_tokens(text)
    cjk_terms = deterministic_cjk_ngrams(text)
    if cjk_terms:
        # ``websearch_to_tsquery`` treats plain whitespace as AND.  CJK
        # n-grams from a longer source need an OR clause because an interior
        # query n-gram can be separated by characters in that source.
        cjk_clause = " OR ".join(cjk_terms)
        prefix = " ".join(part for part in (ascii_phrase, *identifiers) if part)
        return f"{prefix} OR {cjk_clause}" if prefix else cjk_clause
    return normalize_search_text(" ".join([ascii_phrase, *identifiers]))


def build_search_text(
    *,
    title: str = "",
    heading_path: Iterable[str] | None = None,
    source_path: str = "",
    content: str = "",
) -> str:
    """Build one deterministic field from metadata, body and search tokens.

    The first portion retains readable source text for simple PostgreSQL
    tokenization.  Identifier and CJK terms are appended as space-delimited
    tokens so ``simple`` full-text search can match them without a language
    specific parser.
    """

    headings = [str(value) for value in (heading_path or ()) if value]
    source_fields = [str(title or ""), *headings, str(source_path or ""), str(content or "")]
    readable = " ".join(value for value in source_fields if value)
    identifiers = technical_identifier_tokens(readable)
    cjk_terms = deterministic_cjk_ngrams(readable)
    return normalize_search_text(" ".join([readable, *identifiers, *cjk_terms]))


# Short aliases make the normalization contract convenient for ingestion and
# tests without exposing the implementation regexes.
extract_technical_identifiers = technical_identifier_tokens
build_deterministic_search_text = build_search_text


__all__ = [
    "build_deterministic_search_text",
    "build_full_text_query",
    "build_search_text",
    "deterministic_cjk_ngrams",
    "extract_technical_identifiers",
    "normalize_search_text",
    "technical_identifier_tokens",
]
