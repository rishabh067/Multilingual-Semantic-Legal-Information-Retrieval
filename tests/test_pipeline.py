"""Unit tests for the multilingual legal retrieval pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["FORCE_FALLBACK_TRANSLATION"] = "true"
os.environ["FORCE_FALLBACK_EMBEDDINGS"] = "true"

from src.api import app  # noqa: E402
from src.embedder import SentenceBERTEmbedder  # noqa: E402
from src.retriever import LegalSemanticRetriever  # noqa: E402
from src.translator import IndicTranslator  # noqa: E402


def test_translation_and_language_detection() -> None:
    translator = IndicTranslator()
    assert translator.detect_language("છેતરપિંડી કેસ") == "gu"
    assert "cheating" in translator.translate_to_english("છેતરપિંડી કેસ", "gu").lower()


def test_embedding_shape() -> None:
    embedder = SentenceBERTEmbedder()
    vectors = embedder.encode_documents(["legal cheating case", "contract dispute"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0


def test_retrieval_returns_results() -> None:
    retriever = LegalSemanticRetriever()
    output = retriever.retrieve("છેતરપિંડી કેસ", language="gu", top_k=3)
    assert output["language"] == "gu"
    assert len(output["results"]) == 3
    assert output["results"][0]["case_id"]


def test_api_responds() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200

    response = client.post("/search", json={"query": "છેતરપિંડી કેસ", "language": "gu", "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert "results" in payload
