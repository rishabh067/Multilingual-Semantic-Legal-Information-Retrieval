"""Central configuration for the legal multilingual IR project."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
ARTIFACT_DIR = MODEL_DIR / "artifacts"

load_dotenv(ROOT_DIR / ".env", override=False)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    app_env: str = "development"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_dir: Path = Field(default=MODEL_DIR / "hf" / "sentence-transformers--all-MiniLM-L6-v2")
    indic_to_en_model: str = "ai4bharat/indictrans2-indic-en-1B"
    en_to_indic_model: str = "ai4bharat/indictrans2-en-indic-1B"
    indic_to_en_model_dir: Path = Field(default=MODEL_DIR / "hf" / "ai4bharat--indictrans2-indic-en-1B")
    en_to_indic_model_dir: Path = Field(default=MODEL_DIR / "hf" / "ai4bharat--indictrans2-en-indic-1B")
    hf_hub_token: str | None = None
    top_k_default: int = 5
    use_lowercase: bool = False
    cpu_batch_size: int = 8
    gpu_batch_size: int = 32
    translation_batch_size: int = 8
    translation_num_beams: int = 2
    semantic_candidate_pool: int = 50
    bm25_candidate_pool: int = 50
    focused_rerank_pool: int = 25
    semantic_weight: float = 0.65
    bm25_weight: float = 0.25
    overlap_weight: float = 0.10
    citation_weight: float = 0.20
    title_overlap_weight: float = 0.15
    focused_semantic_weight: float = 0.20
    phrase_match_weight: float = 0.15
    force_fallback_translation: bool = False
    force_fallback_embeddings: bool = False
    use_lesicin_raw: bool = True
    lesicin_raw_dir: Path = Field(default=DATA_DIR / "lesicin_raw")
    lesicin_train_path: Path = Field(default=DATA_DIR / "lesicin_raw" / "train.jsonl")
    lesicin_dev_path: Path = Field(default=DATA_DIR / "lesicin_raw" / "dev.jsonl")
    lesicin_test_path: Path = Field(default=DATA_DIR / "lesicin_raw" / "test.jsonl")
    lesicin_secs_path: Path = Field(default=DATA_DIR / "lesicin_raw" / "secs.jsonl")
    cases_path: Path = Field(default=DATA_DIR / "lesicin_cases.jsonl")
    eval_queries_path: Path = Field(default=DATA_DIR / "eval_queries.json")
    embeddings_path: Path = Field(default=ARTIFACT_DIR / "case_embeddings.npy")
    focused_embeddings_path: Path = Field(default=ARTIFACT_DIR / "focused_case_embeddings.npy")
    metadata_path: Path = Field(default=ARTIFACT_DIR / "case_metadata.json")
    hnsw_index_path: Path = Field(default=ARTIFACT_DIR / "case_index.bin")
    metrics_plot_path: Path = Field(default=ARTIFACT_DIR / "bm25_vs_semantic.png")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return settings
