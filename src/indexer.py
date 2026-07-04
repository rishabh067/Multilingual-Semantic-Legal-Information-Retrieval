"""Vector indexing using HNSW with a numpy fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import get_settings
from src.utils import ensure_parent, get_logger


LOGGER = get_logger(__name__)


class HNSWIndexer:
    """Manage HNSW index lifecycle."""

    def __init__(
        self,
        space: str = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> None:
        self.settings = get_settings()
        self.space = space
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = None
        self.vectors: np.ndarray | None = None
        self.ids: np.ndarray | None = None
        self._use_hnsw = False

    def build_hnsw_index(self, vectors: np.ndarray) -> "HNSWIndexer":
        """Build the vector index."""
        self.vectors = vectors.astype(np.float32)
        self.ids = np.arange(len(vectors))
        dimension = vectors.shape[1]
        try:
            import hnswlib

            self.index = hnswlib.Index(space=self.space, dim=dimension)
            self.index.init_index(max_elements=len(vectors), ef_construction=self.ef_construction, M=self.m)
            self.index.add_items(self.vectors, self.ids)
            self.index.set_ef(self.ef_search)
            self._use_hnsw = True
            LOGGER.info("Built HNSW index with %s vectors", len(vectors))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Using numpy search fallback instead of hnswlib: %s", exc)
            self.index = None
            self._use_hnsw = False
        return self

    def save_index(self, path: str | Path | None = None) -> None:
        """Persist the index or fallback vectors."""
        path = Path(path or self.settings.hnsw_index_path)
        ensure_parent(path)
        if self._use_hnsw and self.index is not None:
            self.index.save_index(str(path))
        elif self.vectors is not None and self.ids is not None:
            np.savez(path.with_suffix(".npz"), vectors=self.vectors, ids=self.ids)
        LOGGER.info("Saved index artifacts for %s", path)

    def load_index(self, dimension: int, path: str | Path | None = None) -> "HNSWIndexer":
        """Load an existing index or fallback vectors."""
        path = Path(path or self.settings.hnsw_index_path)
        try:
            import hnswlib

            self.index = hnswlib.Index(space=self.space, dim=dimension)
            self.index.load_index(str(path))
            self.index.set_ef(self.ef_search)
            self._use_hnsw = True
            LOGGER.info("Loaded HNSW index from %s", path)
            return self
        except Exception:
            payload = np.load(path.with_suffix(".npz"))
            self.vectors = payload["vectors"].astype(np.float32)
            self.ids = payload["ids"]
            self._use_hnsw = False
            LOGGER.info("Loaded numpy fallback index from %s", path.with_suffix(".npz"))
            return self

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search the index and return ids and similarity scores."""
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        if self._use_hnsw and self.index is not None:
            labels, distances = self.index.knn_query(query_vector, k=top_k)
            return labels[0], 1.0 - distances[0]

        if self.vectors is None or self.ids is None:
            raise RuntimeError("Index is not initialized")

        scores = (self.vectors @ query_vector.T).reshape(-1)
        order = np.argsort(scores)[::-1][:top_k]
        return self.ids[order], scores[order]
