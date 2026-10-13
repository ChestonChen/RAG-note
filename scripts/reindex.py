"""Explicitly rebuild the derived knowledge index from source files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_service.app.config import get_settings  # noqa: E402
from rag_service.app.reindex import rebuild_derived_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="confirm dropping and recreating derived index tables",
    )
    args = parser.parse_args()
    try:
        result = rebuild_derived_index(get_settings(), confirm=args.confirm)
    except Exception as exc:
        print(f"reindex failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
