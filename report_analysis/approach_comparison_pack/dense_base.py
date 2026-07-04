"""Reusable dense retrieval approach implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import CACHE_DIR, SearchResult, QueryTranslator, cosine_top_k


@dataclass
class DenseApproachConfig:
    """Configuration for one dense model comparison approach."""

    name: str
    model_name: str
    cache_key: str
    query_policy: str = "translate_to_english"
    local_dir: str | None = None
    batch_size: int = 8
    max_length: int = 256


class SentenceTransformerApproach:
    """Dense retrieval using sentence-transformers models."""

    def __init__(self, config: DenseApproachConfig, force: bool = False) -> None:
        self.config = config
        self.name = config.name
        self.force = force
        self.records: list[dict] = []
        self.case_ids: list[str] = []
        self.matrix: np.ndarray | None = None
        self.model = None
        self.query_translator: QueryTranslator | None = None

    @property
    def cache_path(self) -> Path:
        """Embedding cache path."""
        return CACHE_DIR / f"{self.config.cache_key}_document_embeddings.npy"

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        source = self.config.local_dir or self.config.model_name
        local_only = self.config.local_dir is not None and Path(self.config.local_dir).exists()
        return SentenceTransformer(source, device="cpu", local_files_only=local_only)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def build(self, records: list[dict]) -> None:
        """Load or compute document embeddings."""
        self.records = records
        self.case_ids = [str(record["case_id"]) for record in records]
        self.model = self._load_model()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists() and not self.force:
            self.matrix = np.load(self.cache_path).astype(np.float32)
            return

        texts = [record["document_text"] for record in records]
        self.matrix = self._encode(texts)
        np.save(self.cache_path, self.matrix)

    def _query_text(self, query: dict) -> str:
        if self.config.query_policy == "original":
            return str(query["query_text"])
        if self.query_translator is None:
            self.query_translator = QueryTranslator()
        return self.query_translator.to_english(query)

    def search(self, query: dict, top_k: int) -> list[SearchResult]:
        """Search by cosine similarity."""
        if self.matrix is None:
            raise RuntimeError(f"{self.name} is not built")
        query_vector = self._encode([self._query_text(query)])[0]
        indices, scores = cosine_top_k(query_vector, self.matrix, top_k)
        return [SearchResult(self.case_ids[int(index)], float(score)) for index, score in zip(indices, scores)]


class HuggingFaceMeanPoolingApproach(SentenceTransformerApproach):
    """Dense retrieval using AutoModel mean-pooled token embeddings."""

    def _load_model(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        source = self.config.local_dir or self.config.model_name
        local_only = self.config.local_dir is not None and Path(self.config.local_dir).exists()
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
        model = AutoModel.from_pretrained(source, local_files_only=local_only)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "torch": torch, "device": device}

    def _encode(self, texts: list[str]) -> np.ndarray:
        torch = self.model["torch"]
        tokenizer = self.model["tokenizer"]
        model = self.model["model"]
        device = self.model["device"]
        batches: list[np.ndarray] = []
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        for start in range(0, len(texts), self.config.batch_size):
            batch_number = (start // self.config.batch_size) + 1
            if batch_number == 1 or batch_number % 100 == 0 or batch_number == total_batches:
                print(f"{self.name}: encoding batch {batch_number}/{total_batches}", flush=True)
            batch = texts[start : start + self.config.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * attention_mask, dim=1)
            counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            vectors = summed / counts
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            batches.append(vectors.cpu().numpy().astype(np.float32))

        return np.vstack(batches) if batches else np.zeros((0, 768), dtype=np.float32)
