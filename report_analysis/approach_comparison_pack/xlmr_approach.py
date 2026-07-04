"""XLM-R exact comparison approach."""

from __future__ import annotations

from pathlib import Path

from common import PROJECT_ROOT, RESULTS_DIR, evaluate_approach, load_case_records, load_queries, parse_common_args
from dense_base import DenseApproachConfig, HuggingFaceMeanPoolingApproach


def build_approach(force: bool = False) -> HuggingFaceMeanPoolingApproach:
    """Create the XLM-R approach."""
    local_dir = PROJECT_ROOT / "models" / "hf" / "xlm-roberta-base"
    return HuggingFaceMeanPoolingApproach(
        DenseApproachConfig(
            name="XLM-R",
            model_name="xlm-roberta-base",
            local_dir=str(local_dir) if Path(local_dir).exists() else None,
            cache_key="xlmr",
            query_policy="original",
            batch_size=8,
        ),
        force=force,
    )


def main() -> None:
    """Run XLM-R as a standalone comparable experiment."""
    args = parse_common_args("Evaluate XLM-R direct multilingual dense retrieval on the shared benchmark.")
    records = load_case_records(args.max_docs)
    queries = load_queries(args.eval_path, args.max_queries)
    output = args.output or RESULTS_DIR / "xlmr_results.csv"
    row = evaluate_approach(build_approach(args.force), records, queries, output)
    print(row)


if __name__ == "__main__":
    main()
