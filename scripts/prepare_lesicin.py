"""Prepare the official LeSICiN dataset for retrieval use."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.data_loader import load_lesicin_corpus
from src.preprocess import preprocess_lesicin_record


def main() -> None:
    settings = get_settings()
    corpus = load_lesicin_corpus(
        settings.lesicin_train_path,
        settings.lesicin_dev_path,
        settings.lesicin_test_path,
        settings.lesicin_secs_path,
    )
    records = [preprocess_lesicin_record(row, settings.use_lowercase) for row in corpus.to_dict("records")]

    settings.cases_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.cases_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Prepared {len(records)} LeSICiN records at {settings.cases_path}")


if __name__ == "__main__":
    main()
