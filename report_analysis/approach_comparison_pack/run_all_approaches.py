"""Run all retrieval approaches on the same benchmark for exact comparison."""

from __future__ import annotations

import argparse
import json
import traceback

from bm25_approach import BM25Approach
from bert_approach import build_approach as build_bert
from common import RESULTS_DIR, evaluate_approach, load_case_records, load_queries, save_run_metadata, write_all_results
from legalbert_approach import build_approach as build_legalbert
from mbert_approach import build_approach as build_mbert
from proposed_approach import ProposedHybridApproach
from sentence_bert_approach import build_approach as build_sentence_bert
from xlmr_approach import build_approach as build_xlmr


def parse_args() -> argparse.Namespace:
    """Parse orchestrator flags."""
    parser = argparse.ArgumentParser(description="Evaluate all retrieval approaches on one shared benchmark.")
    parser.add_argument("--eval-path", default=None, help="Evaluation query JSON/JSONL path.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional document limit for smoke tests.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query limit for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Rebuild cached dense embeddings.")
    parser.add_argument("--keep-going", action="store_true", help="Continue if a model is unavailable or fails.")
    parser.add_argument("--output", default=str(RESULTS_DIR / "all_approach_results.csv"), help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    """Evaluate all approaches and save one comparable result table."""
    args = parse_args()
    records = load_case_records(args.max_docs)
    queries = load_queries(args.eval_path, args.max_queries)
    approaches = [
        BM25Approach(),
        build_bert(args.force),
        build_legalbert(args.force),
        build_sentence_bert(args.force),
        build_mbert(args.force),
        build_xlmr(args.force),
        ProposedHybridApproach(),
    ]

    rows = []
    failures = []
    for approach in approaches:
        print(f"Running {approach.name}...")
        try:
            rows.append(evaluate_approach(approach, records, queries))
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append({"approach": approach.name, "error": str(exc)})
            print(f"FAILED {approach.name}: {exc}")
            traceback.print_exc()

    write_all_results(rows, args.output)
    save_run_metadata(RESULTS_DIR / "run_metadata.json", records, queries, [row["approach"] for row in rows])
    if failures:
        failure_path = RESULTS_DIR / "failed_approaches.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
