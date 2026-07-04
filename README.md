# Democratizing Legal Information in India through Multilingual Semantic Retrieval

Production-ready mini project inspired by the MEDCOM 2025 IEEE paper, "Democratizing Legal Information in India through Multilingual Semantic Retrieval".

The system accepts a legal query in an Indian language, translates it into English, embeds it into a shared semantic space, retrieves the most relevant Indian legal cases using HNSW vector search, and returns the results back in the user's language.

## Project Overview

- Multilingual query handling for Indian languages
- IndicTrans2-style translation layer with offline-safe fallback support
- Sentence-BERT embeddings using `all-MiniLM-L6-v2`
- HNSW vector retrieval with cacheable artifacts
- BM25 baseline for comparison
- FastAPI backend and Streamlit frontend
- Evaluation with Precision@K, Recall@K, and NDCG@K

## Architecture

```text
User Query (Indic Language)
        |
        v
+------------------------+
|  IndicTrans2 / Fallback|
+------------------------+
        |
        v
Translated English Query
        |
        v
+------------------------+
| Sentence-BERT Encoder  |
+------------------------+
        |
        v
  Query Embedding Vector
        |
        v
+------------------------+
| HNSW Semantic Index    |
+------------------------+
        |
        v
Top-K Relevant Cases
        |
        v
+------------------------+
| EN -> Indic Translation|
+------------------------+
        |
        v
Localized Results + Scores
```

## Folder Structure

```bash
legal_multilingual_ir/
│── app.py
│── requirements.txt
│── README.md
│── .env.example
│
├── data/
│   ├── lesicin_cases.jsonl
│   ├── eval_queries.json
│
├── models/
│
├── notebooks/
│   └── experimentation.ipynb
│
├── scripts/
│   └── build_index.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── translator.py
│   ├── embedder.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── evaluator.py
│   ├── bm25_baseline.py
│   ├── api.py
│   └── utils.py
│
└── tests/
    └── test_pipeline.py
```

## Dataset Format

### Legal cases JSONL

```json
{
  "case_id": "123",
  "sentences": ["fact sentence", "argument sentence"],
  "citations": ["IPC 420", "Contract Act"]
}
```

### Evaluation query JSON

```json
{
  "query_id": "Q1",
  "language": "Gujarati",
  "query_text": "છેતરપિંડી કેસ",
  "relevant_cases": [
    {
      "case_id": "123",
      "relevance": 3
    }
  ]
}
```

If the original LeSICiN corpus is unavailable, the included synthetic sample dataset allows the entire pipeline to run end-to-end.

## Using The Official LeSICiN Dataset

The official LeSICiN corpus is published on Zenodo: [10.5281/zenodo.6053791](https://zenodo.org/records/6053791) and described in the original repository: [Law-AI/LeSICiN](https://github.com/Law-AI/LeSICiN).

Place these files under `data/lesicin_raw/`:

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `secs.jsonl`

Then run:

```bash
python scripts/prepare_lesicin.py
python scripts/build_index.py
```

Notes:

- The official LeSICiN files use `id`, `text`, and `labels`, not the simplified `case_id`, `sentences`, `citations` schema.
- This project now supports both formats automatically.
- The multilingual ranked query evaluation set from the MEDCOM-style retrieval paper is not part of the public LeSICiN release, so `data/eval_queries.json` remains the retrieval benchmark unless you provide the authors' curated query set.

## Installation

```bash
cd legal_multilingual_ir
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## How to Run

### 0. Download and cache models locally

```bash
python scripts/download_models.py
```

### 1. Build embeddings and HNSW index

```bash
python scripts/build_index.py
```

### 2. Run FastAPI backend

```bash
uvicorn src.api:app --reload
```

### 3. Run Streamlit frontend

```bash
streamlit run app.py
```

## API Usage

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Search

```bash
curl -X POST http://127.0.0.1:8000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"છેતરપિંડી કેસ\",\"language\":\"gu\",\"top_k\":5}"
```

### Evaluate

```bash
curl -X POST http://127.0.0.1:8000/evaluate
```

## Metrics

The project evaluates:

- Precision@K
- Recall@K
- NDCG@K

It also creates a comparison plot for:

- Semantic retrieval
- BM25 baseline

Expected behavior, aligned with the paper:

- Strong Precision@1
- Competitive NDCG@3
- Semantic retrieval outperforming BM25 on multilingual queries

## Caching and Robustness

- Embeddings are cached in `models/artifacts/case_embeddings.npy`
- Metadata is cached in `models/artifacts/case_metadata.json`
- HNSW index is cached in `models/artifacts/case_index.bin`
- Translation and embedding fall back gracefully when models are unavailable
- CPU-friendly smaller batch sizes are used when GPU is not present

## Testing

```bash
pytest -q
```

The tests force fallback translation and embeddings, making the suite lightweight and easier to run locally.

## Screenshots

- `docs/ui-home.png` - add frontend screenshot here
- `docs/results-view.png` - add results screenshot here
- `docs/evaluation-chart.png` - add evaluation screenshot here

## Future Scope

- Replace fallback translation with fully hosted IndicTrans2 checkpoints
- Integrate domain-specific legal embedding models
- Add reranking with cross-encoders or LLM-based legal summarization
- Support larger LeSICiN subsets and court metadata filters
- Add Docker, CI, and deployment manifests

## Model Notes

- The project now expects local model caches under `models/hf/`.
- Translation uses the official AI4Bharat IndicTrans2 Hugging Face interface with `IndicTransToolkit`.
- The AI4Bharat IndicTrans2 Hugging Face repositories currently require authenticated access. Add a Hugging Face token to `.env` as `HF_HUB_TOKEN=...` after requesting access on the model pages, then run `python scripts/download_models.py`.
- If local model files are missing, translation and embedding fall back gracefully instead of crashing.

## Commands to Run

```bash
pip install -r requirements.txt
python scripts/download_models.py
python scripts/build_index.py
python scripts/prepare_lesicin.py
uvicorn src.api:app --reload
streamlit run app.py
```
