# Report Analysis Workspace

This folder is intentionally separate from the working application pipeline.

Purpose:

- compare multiple retrieval approaches for your project report
- keep report-specific artifacts away from the production app code
- allow you to maintain manual, literature-based, or experiment-based metrics in one place

Approaches included:

- BM25
- BERT
- LegalBERT
- Sentence-BERT
- mBERT
- XLM-R
- Proposed Multilingual Hybrid

Files:

- `model_comparison_100_query.csv`
  Stores the comparison metrics table.
- `analysis_notes.md`
  Report-ready interpretation notes and writing guidance.
- `generate_report_assets.py`
  Reads the CSV and creates a markdown summary and comparison plots.

Recommended workflow:

1. Update `model_comparison_100_query.csv` with measured or literature-backed metrics.
2. Run:

```powershell
.\.venv\Scripts\python.exe report_analysis\generate_report_assets.py
```

3. Use the generated files in your mini-project report.

Generated outputs:

- `comparison_summary.md`
- `comparison_p_at_1.png`
- `comparison_ndcg_at_3.png`

Note:

Only BM25 and Proposed Multilingual Hybrid are currently filled with project-specific measured values from the restored 100-query custom benchmark. The remaining models are left blank so you can fill them with either:

- your own experimental results, or
- values cited from literature with proper attribution in the report.
