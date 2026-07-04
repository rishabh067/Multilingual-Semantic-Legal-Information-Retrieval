"""Sentence-BERT exact comparison approach."""

from __future__ import annotations

from pathlib import Path

from common import PROJECT_ROOT, RESULTS_DIR, evaluate_approach, load_case_records, load_queries, parse_common_args
from dense_base import DenseApproachConfig, SentenceTransformerApproach


def build_approach(force: bool = False) -> SentenceTransformerApproach:
    """Create the Sentence-BERT approach."""
    local_dir = PROJECT_ROOT / "models" / "hf" / "sentence-transformers--all-MiniLM-L6-v2"
    return SentenceTransformerApproach(
        DenseApproachConfig(
            name="Sentence-BERT",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            local_dir=str(local_dir) if Path(local_dir).exists() else None,
            cache_key="sentence_bert",
            query_policy="translate_to_english",
            batch_size=16,
        ),
        force=force,
    )


def main() -> None:
    """Run Sentence-BERT as a standalone comparable experiment."""
    args = parse_common_args("Evaluate Sentence-BERT dense retrieval on the shared benchmark.")
    records = load_case_records(args.max_docs)
    queries = load_queries(args.eval_path, args.max_queries)
    output = args.output or RESULTS_DIR / "sentence_bert_results.csv"
    row = evaluate_approach(build_approach(args.force), records, queries, output)
    print(row)


if __name__ == "__main__":
    main()
