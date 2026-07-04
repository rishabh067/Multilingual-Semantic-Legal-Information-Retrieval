# Methodology Matrix

| Approach | What it represents | Evaluation setup for fair comparison | Strength | Limitation |
|---|---|---|---|---|
| BM25 | Classical sparse lexical retrieval baseline | Index the same legal corpus and search translated or original queries using BM25 | Simple, fast, explainable | Weak for paraphrases, cross-lingual queries, and semantic similarity |
| BERT | General-purpose transformer encoder | Encode query-document pairs or use pooled embeddings with cosine similarity; preferably fine-tune for retrieval | Captures contextual meaning better than lexical matching | Plain BERT is not naturally optimized for sentence embedding retrieval |
| LegalBERT | Domain-specific legal transformer | Use legal-domain embeddings or reranking over the same candidate pool | Better legal vocabulary and legal phrase understanding | Mostly English/legal-domain focused; multilingual support requires translation or adaptation |
| Sentence-BERT | Sentence-level semantic embedding model | Encode queries and legal documents into a shared vector space and rank by cosine similarity | Designed for semantic similarity and dense retrieval | General SBERT may miss legal-specific cues without domain adaptation |
| mBERT | Multilingual BERT encoder | Encode Indian-language queries and English legal documents directly or after translation | Multilingual coverage across many languages | Legal-domain ranking quality may be weaker without fine-tuning |
| XLM-R | Robust multilingual transformer encoder | Use as a cross-lingual dense retriever or reranker on the same benchmark | Strong multilingual representations | Computationally heavier and still needs retrieval-oriented adaptation |
| Proposed Multilingual Hybrid Retrieval | Implemented project pipeline | Translate query, retrieve dense candidates, combine BM25, semantic score, overlap, citations, and legal-domain reranking | Combines semantic, lexical, multilingual, and legal-specific signals | More complex pipeline; dependent on translation quality and weighting choices |

## Fair Comparison Protocol

Use the same corpus, query set, relevance judgments, and top-k values for every method. Report Precision@1, Precision@3, Precision@5, Recall@1, Recall@3, Recall@5, NDCG@1, NDCG@3, and NDCG@5.

For models not implemented in this project, do not present blank rows as experimental results. Mark them as either `Not measured in this project` or `Reported from literature`, and cite the source if using literature values.
