# Retrieval Approach Comparison Pack

This folder is separate from the working application code. It is meant only for project-report comparison, result presentation, and analysis.

Included approaches:

- BM25
- BERT
- LegalBERT
- Sentence-BERT
- mBERT
- XLM-R
- Proposed Multilingual Hybrid Retrieval

Files:

- `approach_results_table.csv`: report table with measured project results where available.
- `report_ready_analysis.md`: concise report-ready discussion and interpretation.
- `methodology_matrix.md`: how each method should be described, evaluated, and compared.
- `bm25_approach.py`: BM25 baseline runner.
- `bert_approach.py`: BERT mean-pooled dense retrieval runner.
- `legalbert_approach.py`: LegalBERT mean-pooled dense retrieval runner.
- `sentence_bert_approach.py`: Sentence-BERT dense retrieval runner.
- `mbert_approach.py`: mBERT direct multilingual dense retrieval runner.
- `xlmr_approach.py`: XLM-R direct multilingual dense retrieval runner.
- `proposed_approach.py`: proposed project pipeline runner.
- `run_all_approaches.py`: orchestrator that evaluates every approach on the same benchmark.

## Exact Comparison Workflow

Run every approach on the same corpus, same evaluation queries, and same metrics:

```powershell
.\.venv\Scripts\python.exe report_analysis\approach_comparison_pack\run_all_approaches.py --keep-going
```

Outputs:

- `results/all_approach_results.csv`
- `results/run_metadata.json`
- `results/failed_approaches.json` if any model is unavailable

Run one approach separately:

```powershell
.\.venv\Scripts\python.exe report_analysis\approach_comparison_pack\bm25_approach.py
.\.venv\Scripts\python.exe report_analysis\approach_comparison_pack\sentence_bert_approach.py
.\.venv\Scripts\python.exe report_analysis\approach_comparison_pack\proposed_approach.py
```

For a quick smoke test before a full run:

```powershell
.\.venv\Scripts\python.exe report_analysis\approach_comparison_pack\run_all_approaches.py --max-docs 100 --max-queries 5 --keep-going
```

Dense model embeddings are cached inside:

```text
report_analysis/approach_comparison_pack/cache/
```

This keeps comparison artifacts separate from the working project pipeline.

Important note:

Only BM25 and the proposed hybrid retrieval pipeline have project-specific measured scores in `approach_results_table.csv`. After running `run_all_approaches.py`, use `results/all_approach_results.csv` as the exact experimental comparison table.
