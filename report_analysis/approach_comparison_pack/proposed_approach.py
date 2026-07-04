"""Proposed hybrid retrieval exact comparison approach."""

from __future__ import annotations

from common import RESULTS_DIR, SearchResult, evaluate_approach, load_case_records, load_queries, parse_common_args


class ProposedHybridApproach:
    """Wrapper around the implemented project retrieval pipeline."""

    name = "Proposed Multilingual Hybrid Retrieval"

    def __init__(self) -> None:
        self.retriever = None

    def build(self, records: list[dict]) -> None:
        from src.retriever import LegalSemanticRetriever

        self.retriever = LegalSemanticRetriever()

    def search(self, query: dict, top_k: int) -> list[SearchResult]:
        if self.retriever is None:
            raise RuntimeError("Proposed retriever is not built")
        payload = self.retriever.retrieve(
            query["query_text"],
            query.get("language"),
            top_k=top_k,
            translate_results=False,
        )
        return [SearchResult(item["case_id"], float(item["score"])) for item in payload["results"]]


def main() -> None:
    """Run the proposed hybrid system as a standalone comparable experiment."""
    args = parse_common_args("Evaluate the proposed multilingual hybrid retrieval system.")
    records = load_case_records(args.max_docs)
    queries = load_queries(args.eval_path, args.max_queries)
    output = args.output or RESULTS_DIR / "proposed_results.csv"
    row = evaluate_approach(ProposedHybridApproach(), records, queries, output)
    print(row)


if __name__ == "__main__":
    main()
