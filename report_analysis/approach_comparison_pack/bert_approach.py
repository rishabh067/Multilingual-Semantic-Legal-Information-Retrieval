"""BERT exact comparison approach."""

from __future__ import annotations

from pathlib import Path

from common import PROJECT_ROOT, RESULTS_DIR, evaluate_approach, load_case_records, load_queries, parse_common_args
from dense_base import DenseApproachConfig, HuggingFaceMeanPoolingApproach


def build_approach(force: bool = False) -> HuggingFaceMeanPoolingApproach:
    """Create the BERT approach."""
    local_dir = PROJECT_ROOT / "models" / "hf" / "bert-base-uncased"
    return HuggingFaceMeanPoolingApproach(
        DenseApproachConfig(
            name="BERT",
            model_name="bert-base-uncased",
            local_dir=str(local_dir) if Path(local_dir).exists() else None,
            cache_key="bert",
            query_policy="translate_to_english",
            batch_size=8,
        ),
        force=force,
    )


def main() -> None:
    """Run BERT as a standalone comparable experiment."""
    args = parse_common_args("Evaluate BERT mean-pooled dense retrieval on the shared benchmark.")
    records = load_case_records(args.max_docs)
    queries = load_queries(args.eval_path, args.max_queries)
    output = args.output or RESULTS_DIR / "bert_results.csv"
    row = evaluate_approach(build_approach(args.force), records, queries, output)
    print(row)


if __name__ == "__main__":
    main()
