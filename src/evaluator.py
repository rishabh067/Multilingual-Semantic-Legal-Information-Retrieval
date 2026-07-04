"""Evaluation metrics and experiment runner."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from src.bm25_baseline import BM25Baseline
from src.config import get_settings
from src.data_loader import load_eval_queries
from src.retriever import LegalSemanticRetriever
from src.utils import get_logger, safe_divide


LOGGER = get_logger(__name__)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute precision at k."""
    return safe_divide(sum(doc_id in relevant_ids for doc_id in retrieved_ids[:k]), k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute recall at k."""
    return safe_divide(sum(doc_id in relevant_ids for doc_id in retrieved_ids[:k]), len(relevant_ids))


def dcg(relevances: Iterable[int]) -> float:
    """Compute discounted cumulative gain."""
    score = 0.0
    for index, relevance in enumerate(relevances, start=1):
        score += relevance / (1.0 if index == 1 else math.log2(index))
    return score


def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, int], k: int) -> float:
    """Compute normalized DCG at k."""
    actual_relevances = [relevance_map.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    ideal_relevances = sorted(relevance_map.values(), reverse=True)[:k]
    return safe_divide(dcg(actual_relevances), dcg(ideal_relevances))


def evaluate_all_queries(
    retriever: LegalSemanticRetriever | None = None,
    eval_path: str | Path | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Evaluate semantic retrieval and BM25 across evaluation queries."""
    settings = get_settings()
    retriever = retriever or LegalSemanticRetriever()
    eval_df = load_eval_queries(eval_path or settings.eval_queries_path)
    bm25 = BM25Baseline(retriever.case_records)

    semantic_scores: dict[str, list[float]] = {}
    bm25_scores: dict[str, list[float]] = {}
    for k in ks:
        semantic_scores[f"P@{k}"] = []
        semantic_scores[f"R@{k}"] = []
        semantic_scores[f"NDCG@{k}"] = []
        bm25_scores[f"P@{k}"] = []
        bm25_scores[f"R@{k}"] = []
        bm25_scores[f"NDCG@{k}"] = []

    for query in eval_df.to_dict("records"):
        relevant_ids = {item["case_id"] for item in query["relevant_cases"]}
        relevance_map = {item["case_id"]: item["relevance"] for item in query["relevant_cases"]}

        semantic_results = retriever.retrieve(
            query["query_text"],
            query["language"],
            top_k=max(ks),
            translate_results=False,
        )
        semantic_ids = [item["case_id"] for item in semantic_results["results"]]

        bm25_results = bm25.bm25_search(query["query_text"], top_k=max(ks))
        bm25_ids = [item.case_id for item in bm25_results]

        for k in ks:
            semantic_scores[f"P@{k}"].append(precision_at_k(semantic_ids, relevant_ids, k))
            semantic_scores[f"R@{k}"].append(recall_at_k(semantic_ids, relevant_ids, k))
            semantic_scores[f"NDCG@{k}"].append(ndcg_at_k(semantic_ids, relevance_map, k))

            bm25_scores[f"P@{k}"].append(precision_at_k(bm25_ids, relevant_ids, k))
            bm25_scores[f"R@{k}"].append(recall_at_k(bm25_ids, relevant_ids, k))
            bm25_scores[f"NDCG@{k}"].append(ndcg_at_k(bm25_ids, relevance_map, k))

    semantic_avg = {metric: round(sum(values) / len(values), 4) for metric, values in semantic_scores.items()}
    bm25_avg = {metric: round(sum(values) / len(values), 4) for metric, values in bm25_scores.items()}
    plot_metric_comparison(semantic_avg, bm25_avg, settings.metrics_plot_path)
    LOGGER.info("Evaluation completed for %s queries", len(eval_df))
    return {"semantic": semantic_avg, "bm25": bm25_avg, "plot_path": str(settings.metrics_plot_path)}


def plot_metric_comparison(semantic_metrics: dict[str, float], bm25_metrics: dict[str, float], output_path: str | Path) -> None:
    """Create a comparison chart for semantic retrieval vs BM25."""
    output_path = Path(output_path)
    metrics = list(semantic_metrics.keys())
    semantic_values = [semantic_metrics[key] for key in metrics]
    bm25_values = [bm25_metrics[key] for key in metrics]

    plt.figure(figsize=(12, 5))
    x_positions = range(len(metrics))
    width = 0.35
    plt.bar([x - width / 2 for x in x_positions], semantic_values, width=width, label="Semantic")
    plt.bar([x + width / 2 for x in x_positions], bm25_values, width=width, label="BM25")
    plt.xticks(list(x_positions), metrics, rotation=45)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Semantic Retrieval vs BM25")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
