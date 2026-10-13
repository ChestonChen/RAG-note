"""Probe the configured embedding provider once without indexing or secrets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_service.app.config import Settings  # noqa: E402
from rag_service.app.embeddings import create_embedding_provider  # noqa: E402
from rag_service.app.reindex import validate_embedding_provider  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="embedding dimension validation")
    args = parser.parse_args(argv)
    settings = Settings()
    try:
        provider = create_embedding_provider(settings)
        result = validate_embedding_provider(provider, settings.embedding_dimensions, args.sample)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}, ensure_ascii=False))
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
