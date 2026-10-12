"""Initialize PostgreSQL extensions and tables for the RAG service."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

# Allow ``python scripts/init_db.py`` from the repository root without an
# editable package install.
sys.path.insert(0, str(PROJECT_ROOT))

from rag_service.app.config import get_settings  # noqa: E402
from rag_service.app.db import init_db  # noqa: E402


def main() -> None:
    settings = get_settings()
    engine = init_db(settings=settings)
    print(f"Initialized database schema using {engine.url.render_as_string(hide_password=True)}")


if __name__ == "__main__":
    main()
