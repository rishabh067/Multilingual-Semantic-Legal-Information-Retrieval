"""Build and cache embeddings plus HNSW artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.retriever import LegalSemanticRetriever


def main() -> None:
    settings = get_settings()
    if not settings.embedding_model_dir.exists():
        settings.force_fallback_embeddings = True
    retriever = LegalSemanticRetriever()
    retriever.prepare_artifacts(force_rebuild=True)
    print("Artifacts built successfully.")


if __name__ == "__main__":
    main()
