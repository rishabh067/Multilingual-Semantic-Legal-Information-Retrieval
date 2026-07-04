"""Embedding model management with caching and offline fallback support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import get_settings
from src.utils import clear_dead_proxy_env, ensure_parent, get_logger


LOGGER = get_logger(__name__)


@dataclass
class FallbackEmbeddingModel:
    """Deterministic lightweight fallback when SentenceTransformer is unavailable."""

    dimension: int = 384
    vectorizer: object | None = None

    def encode(self, texts: list[str], normalize_embeddings: bool = True, **_: object) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=self.dimension, ngram_range=(1, 2))
            vectors = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        else:
            vectors = self.vectorizer.transform(texts).toarray().astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors


class SentenceBERTEmbedder:
    """Sentence-BERT embedding wrapper."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = LOGGER
        self.device = "cpu"
        self.model = self.load_embedding_model()

    def load_embedding_model(self):
        """Load the requested embedding model or a deterministic fallback."""
        if self.settings.force_fallback_embeddings:
            self.logger.info("Using forced fallback embeddings")
            return FallbackEmbeddingModel()

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            clear_dead_proxy_env()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model_source = (
                str(self.settings.embedding_model_dir)
                if self.settings.embedding_model_dir.exists()
                else self.settings.embedding_model_name
            )
            self.logger.info("Loading embedding model %s on %s", model_source, self.device)
            return SentenceTransformer(model_source, device=self.device, local_files_only=self.settings.embedding_model_dir.exists())
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Falling back to hash embeddings: %s", exc)
            self.device = "cpu"
            return FallbackEmbeddingModel()

    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Encode documents into normalized vectors."""
        if batch_size is None:
            batch_size = self.settings.gpu_batch_size if self.device == "cuda" else self.settings.cpu_batch_size

        if isinstance(self.model, FallbackEmbeddingModel):
            return self.model.encode(texts, normalize_embeddings=True)

        embeddings: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding cases"):
            batch = texts[start : start + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            embeddings.append(batch_embeddings.astype(np.float32))
        return np.vstack(embeddings) if embeddings else np.zeros((0, 384), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query into a vector."""
        return self.encode_documents([text], batch_size=1)[0]

    def save_embeddings(self, embeddings: np.ndarray, path: str | Path | None = None) -> None:
        """Persist embeddings to disk."""
        path = Path(path or self.settings.embeddings_path)
        ensure_parent(path)
        np.save(path, embeddings)
        self.logger.info("Saved embeddings to %s", path)

    def load_embeddings(self, path: str | Path | None = None) -> np.ndarray:
        """Load embeddings from disk."""
        path = Path(path or self.settings.embeddings_path)
        embeddings = np.load(path)
        self.logger.info("Loaded embeddings from %s with shape %s", path, embeddings.shape)
        return embeddings.astype(np.float32)
