"""FastAPI service for multilingual legal retrieval."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.evaluator import evaluate_all_queries
from src.retriever import LegalSemanticRetriever


app = FastAPI(
    title="Multilingual Legal Semantic Retrieval API",
    version="1.0.0",
    description="Semantic retrieval for Indian legal cases using translation, Sentence-BERT, and HNSW.",
)


class SearchRequest(BaseModel):
    """Incoming search request."""

    query: str = Field(..., min_length=2)
    language: str = Field(default="en")
    top_k: int = Field(default=5, ge=1, le=20)


@lru_cache(maxsize=1)
def get_retriever() -> LegalSemanticRetriever:
    """Create a singleton retriever instance."""
    return LegalSemanticRetriever()


@app.get("/")
def root() -> dict[str, str]:
    """Friendly root endpoint for browser visits."""
    return {
        "message": "Multilingual Legal Semantic Retrieval API",
        "health": "/health",
        "docs": "/docs",
        "search": "/search",
        "evaluate": "/evaluate",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Simple service health check."""
    return {"status": "ok"}


@app.post("/search")
def search(request: SearchRequest) -> dict:
    """Retrieve top-k relevant cases."""
    try:
        return get_retriever().retrieve(request.query, request.language, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluate")
def evaluate() -> dict:
    """Run retrieval evaluation."""
    try:
        return evaluate_all_queries(get_retriever())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
