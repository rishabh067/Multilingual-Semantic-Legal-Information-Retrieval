"""Dataset loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import pandas as pd

from src.utils import get_logger


LOGGER = get_logger(__name__)

CASE_REQUIRED_FIELDS = {"case_id", "sentences", "citations"}
EVAL_REQUIRED_FIELDS = {"query_id", "language", "query_text", "relevant_cases"}
LESICIN_REQUIRED_FIELDS = {"id", "text"}


def validate_schema(record: dict[str, Any], required_fields: set[str]) -> bool:
    """Validate that a record contains required fields."""
    return required_fields.issubset(record.keys())


def load_jsonl_cases(path: str | Path) -> pd.DataFrame:
    """Load case records from a JSONL file with malformed row handling."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    malformed_rows = 0

    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                LOGGER.warning("Skipping malformed JSON case row %s in %s", index, path.name)
                continue
            if not isinstance(row, dict) or not validate_schema(row, CASE_REQUIRED_FIELDS):
                malformed_rows += 1
                LOGGER.warning("Skipping malformed case row %s in %s", index, path.name)
                continue
            rows.append(row)

    dataframe = pd.DataFrame(rows)
    LOGGER.info(
        "Loaded %s valid case rows from %s with %s malformed rows skipped",
        len(dataframe),
        path,
        malformed_rows,
    )
    return dataframe


def load_lesicin_jsonl(path: str | Path, record_type: str) -> pd.DataFrame:
    """Load an official LeSICiN JSONL split."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    malformed_rows = 0

    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                LOGGER.warning("Skipping malformed JSON row %s in %s", index, path.name)
                continue
            if not isinstance(row, dict) or not validate_schema(row, LESICIN_REQUIRED_FIELDS):
                malformed_rows += 1
                LOGGER.warning("Skipping malformed LeSICiN row %s in %s", index, path.name)
                continue

            rows.append(
                {
                    "id": row["id"],
                    "text": row.get("text", []),
                    "labels": row.get("labels") or [],
                    "record_type": record_type,
                }
            )

    dataframe = pd.DataFrame(rows)
    LOGGER.info(
        "Loaded %s valid LeSICiN %s rows from %s with %s malformed rows skipped",
        len(dataframe),
        record_type,
        path,
        malformed_rows,
    )
    return dataframe


def load_lesicin_corpus(train_path: str | Path, dev_path: str | Path, test_path: str | Path, secs_path: str | Path) -> pd.DataFrame:
    """Load and combine the official LeSICiN corpus splits."""
    frames = [
        load_lesicin_jsonl(train_path, "fact_train"),
        load_lesicin_jsonl(dev_path, "fact_dev"),
        load_lesicin_jsonl(test_path, "fact_test"),
        load_lesicin_jsonl(secs_path, "section"),
    ]
    dataframe = pd.concat(frames, ignore_index=True)
    LOGGER.info("Combined LeSICiN corpus contains %s records", len(dataframe))
    return dataframe


def load_eval_queries(path: str | Path) -> pd.DataFrame:
    """Load evaluation queries from a JSON or JSONL file."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipping malformed JSON eval row %s in %s", index, path.name)
                    continue
                if not isinstance(row, dict) or not validate_schema(row, EVAL_REQUIRED_FIELDS):
                    LOGGER.warning("Skipping malformed eval row %s in %s", index, path.name)
                    continue
                rows.append(row)
        dataframe = pd.DataFrame(rows)
    else:
        dataframe = pd.read_json(path)
        if not dataframe.empty:
            dataframe = dataframe[
                dataframe.apply(
                    lambda row: validate_schema(row.to_dict(), EVAL_REQUIRED_FIELDS),
                    axis=1,
                )
            ]

    LOGGER.info("Loaded %s evaluation queries from %s", len(dataframe), path)
    return dataframe


def preview_dataset(dataframe: pd.DataFrame, rows: int = 3) -> pd.DataFrame:
    """Return a small preview of a dataframe."""
    LOGGER.info("Previewing first %s rows", rows)
    return dataframe.head(rows)
