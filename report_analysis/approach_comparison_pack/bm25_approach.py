"""BM25 exact comparison approach."""

from __future__ import annotations

from common import QueryTranslator, SearchResult, evaluate_approach, load_case_records, load_queries, parse_common_args, RESULTS_DIR


class BM25Approach:
    """BM25 over translated queries and the shared legal corpus."""

    name = "BM25"

    def __init__(self) -> None:
        self.backend = None
        self.records: list[dict] = []
        self.query_translator = QueryTranslator()

    def build(self, records: list[dict]) -> None:
        from src.bm25_baseline import BM25Baseline

        self.records = records
        self.backend = BM25Baseline(records)

    def search(self, query: dict, top_k: int) -> list[SearchResult]:
        if self.backend is None:
            raise RuntimeError("BM25 is not built")
        query_text = self.query_translator.to_english(query)
        results = self.backend.bm25_search(query_text, top_k)
        return [SearchResult(item.case_id, item.score) for item in results]


def main() -> None:
    """Run BM25 as a standalone comparable experiment."""
    args = parse_common_args("Evaluate BM25 on the shared benchmark.")
    records = load_case_records(args.max_docs)
    queries = load_queries(args.eval_path, args.max_queries)
    output = args.output or RESULTS_DIR / "bm25_results.csv"
    row = evaluate_approach(BM25Approach(), records, queries, output)
    print(row)


if __name__ == "__main__":
    main()
