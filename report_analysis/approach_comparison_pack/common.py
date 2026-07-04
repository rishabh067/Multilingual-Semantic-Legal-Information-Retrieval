"""Shared utilities for isolated approach comparison experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import pandas as pd


PACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACK_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.data_loader import load_eval_queries, load_jsonl_cases, load_lesicin_corpus
from src.preprocess import preprocess_case, preprocess_lesicin_record
from src.translator import IndicTranslator
from src.utils import load_json, normalize_language_code


K_VALUES = (1, 3, 5)
METRIC_COLUMNS = ["P@1", "P@3", "P@5", "R@1", "R@3", "R@5", "NDCG@1", "NDCG@3", "NDCG@5"]
CACHE_DIR = PACK_DIR / "cache"
RESULTS_DIR = PACK_DIR / "results"


@dataclass(frozen=True)
class SearchResult:
    """One ranked retrieval result."""

    case_id: str
    score: float


class RetrievalApproach(Protocol):
    """Interface implemented by every comparison approach."""

    name: str

    def build(self, records: list[dict]) -> None:
        """Prepare model/index state for a shared corpus."""

    def search(self, query: dict, top_k: int) -> list[SearchResult]:
        """Return ranked case IDs for a query."""


def parse_common_args(description: str) -> argparse.Namespace:
    """Parse shared CLI flags for a single approach runner."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--eval-path", default=None, help="Evaluation query JSON/JSONL path.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional document limit for smoke tests.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query limit for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Rebuild cached approach artifacts.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    return parser.parse_args()


def load_case_records(max_docs: int | None = None) -> list[dict]:
    """Load the same corpus representation used by the project pipeline."""
    settings = get_settings()
    if settings.metadata_path.exists() and max_docs is None:
        records = load_json(settings.metadata_path)
    else:
        use_raw_lesicin = (
            settings.use_lesicin_raw
            and settings.lesicin_train_path.exists()
            and settings.lesicin_dev_path.exists()
            and settings.lesicin_test_path.exists()
            and settings.lesicin_secs_path.exists()
        )
        if use_raw_lesicin:
            dataframe = load_lesicin_corpus(
                settings.lesicin_train_path,
                settings.lesicin_dev_path,
                settings.lesicin_test_path,
                settings.lesicin_secs_path,
            )
            records = [
                preprocess_lesicin_record(row, settings.use_lowercase)
                for row in dataframe.to_dict("records")
            ]
        else:
            dataframe = load_jsonl_cases(settings.cases_path)
            records = [preprocess_case(row, settings.use_lowercase) for row in dataframe.to_dict("records")]

    if max_docs is not None:
        records = records[:max_docs]
    return records


def load_queries(eval_path: str | Path | None = None, max_queries: int | None = None) -> list[dict]:
    """Load benchmark queries for all approaches."""
    settings = get_settings()
    dataframe = load_eval_queries(eval_path or settings.eval_queries_path)
    if max_queries is not None:
        dataframe = dataframe.head(max_queries)
    return dataframe.to_dict("records")


def safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division by zero in metric computation."""
    return numerator / denominator if denominator else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute precision at k."""
    return safe_divide(sum(case_id in relevant_ids for case_id in retrieved_ids[:k]), k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute recall at k."""
    return safe_divide(sum(case_id in relevant_ids for case_id in retrieved_ids[:k]), len(relevant_ids))


def dcg(relevances: Iterable[int]) -> float:
    """Compute discounted cumulative gain."""
    total = 0.0
    for index, relevance in enumerate(relevances, start=1):
        total += relevance / (1.0 if index == 1 else math.log2(index))
    return total


def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, int], k: int) -> float:
    """Compute normalized discounted cumulative gain at k."""
    actual = [relevance_map.get(case_id, 0) for case_id in retrieved_ids[:k]]
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    return safe_divide(dcg(actual), dcg(ideal))


def evaluate_approach(
    approach: RetrievalApproach,
    records: list[dict],
    queries: list[dict],
    output_path: str | Path | None = None,
) -> dict[str, float | str]:
    """Evaluate a single approach with identical metrics and benchmark queries."""
    approach.build(records)
    metric_values: dict[str, list[float]] = {metric: [] for metric in METRIC_COLUMNS}
    max_k = max(K_VALUES)

    for query in queries:
        relevant_ids = {item["case_id"] for item in query["relevant_cases"]}
        relevance_map = {item["case_id"]: item["relevance"] for item in query["relevant_cases"]}
        retrieved_ids = [result.case_id for result in approach.search(query, max_k)]

        for k in K_VALUES:
            metric_values[f"P@{k}"].append(precision_at_k(retrieved_ids, relevant_ids, k))
            metric_values[f"R@{k}"].append(recall_at_k(retrieved_ids, relevant_ids, k))
            metric_values[f"NDCG@{k}"].append(ndcg_at_k(retrieved_ids, relevance_map, k))

    row: dict[str, float | str] = {"approach": approach.name}
    row.update({metric: round(sum(values) / len(values), 4) if values else 0.0 for metric, values in metric_values.items()})

    if output_path:
        write_single_result(row, output_path)
    return row


def write_single_result(row: dict[str, float | str], output_path: str | Path) -> None:
    """Write one approach metric row to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["approach", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerow(row)


def write_all_results(rows: list[dict[str, float | str]], output_path: str | Path) -> None:
    """Write all approach results to a single comparable CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe = dataframe[["approach", *METRIC_COLUMNS]]
    dataframe.to_csv(output_path, index=False)


def save_run_metadata(path: str | Path, records: list[dict], queries: list[dict], approaches: list[str]) -> None:
    """Save benchmark metadata for report reproducibility."""
    payload = {
        "document_count": len(records),
        "query_count": len(queries),
        "metrics": METRIC_COLUMNS,
        "top_k_values": list(K_VALUES),
        "approaches": approaches,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class QueryTranslator:
    """Translate non-English queries once and cache them per process."""

    def __init__(self) -> None:
        self.translator = IndicTranslator()
        self.cache: dict[tuple[str, str], str] = {}

    def to_english(self, query: dict) -> str:
        """Return an English query string using the project translator."""
        text = str(query["query_text"])
        language = normalize_language_code(str(query.get("language") or "en"))
        cache_key = (language, text)
        if cache_key not in self.cache:
            self.cache[cache_key] = self.translator.translate_to_english(text, language)
        return self.cache[cache_key]


def cosine_top_k(query_vector: np.ndarray, matrix: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and scores using cosine/dot product on normalized vectors."""
    scores = matrix @ query_vector.reshape(-1)
    order = np.argsort(scores)[::-1][:top_k]
    return order, scores[order]
