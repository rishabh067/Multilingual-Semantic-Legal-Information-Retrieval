"""BM25 baseline retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils import get_logger


LOGGER = get_logger(__name__)


@dataclass
class BM25SearchResult:
    """Single BM25 result item."""

    case_id: str
    score: float
    title: str
    summary: str


class BM25Baseline:
    """Simple BM25 retrieval baseline."""

    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.tokenized_corpus = [doc["document_text"].lower().split() for doc in documents]
        self.backend = None
        self.build_bm25()

    def build_bm25(self) -> None:
        """Initialize BM25 backend if available."""
        try:
            from rank_bm25 import BM25Okapi

            self.backend = BM25Okapi(self.tokenized_corpus)
            LOGGER.info("BM25 baseline initialized")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("BM25 dependency unavailable, using overlap fallback: %s", exc)
            self.backend = None

    def bm25_search(self, query: str, top_k: int) -> list[BM25SearchResult]:
        """Retrieve top documents for a query."""
        query_tokens = query.lower().split()
        if self.backend is not None:
            scores = np.array(self.backend.get_scores(query_tokens))
        else:
            scores = np.array(
                [sum(token in doc_tokens for token in query_tokens) for doc_tokens in self.tokenized_corpus],
                dtype=float,
            )
        order = scores.argsort()[::-1][:top_k]
        return [
            BM25SearchResult(
                case_id=self.documents[index]["case_id"],
                score=float(scores[index]),
                title=self.documents[index]["title"],
                summary=self.documents[index]["summary"],
            )
            for index in order
        ]
