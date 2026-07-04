# Comparative Analysis Notes

## Suggested report framing

The comparative evaluation should be presented as a separate analysis section in the report. Since the main implemented system is the proposed multilingual hybrid pipeline, the comparison table can be used to position the proposed method against classical lexical retrieval, general transformer encoders, multilingual encoders, and legal-domain encoders.

## Recommended section title

`Comparative Performance Analysis of Retrieval Approaches`

## Suggested grouping

- Lexical baseline:
  - BM25
- General transformer family:
  - BERT
  - Sentence-BERT
- Multilingual transformer family:
  - mBERT
  - XLM-R
- Domain-specific legal encoder:
  - LegalBERT
- Proposed system:
  - Proposed Multilingual Hybrid

## Report-ready interpretation points

### 1. BM25 vs semantic systems

BM25 serves as a sparse lexical baseline. On multilingual legal retrieval, its performance is expected to remain low because it depends heavily on exact token overlap and does not capture semantic equivalence after translation or paraphrasing.

### 2. BERT vs Sentence-BERT

Vanilla BERT is not naturally optimized for retrieval embeddings, whereas Sentence-BERT is explicitly designed for semantic similarity and dense retrieval. Therefore, Sentence-BERT is expected to outperform plain BERT on ranking-oriented tasks.

### 3. mBERT and XLM-R

These multilingual encoders are useful for cross-lingual representation learning and can reduce dependence on explicit translation. However, their effectiveness depends on whether they are used directly for dense retrieval and whether they are fine-tuned for legal retrieval tasks.

### 4. LegalBERT

LegalBERT can capture legal-domain terminology better than general encoders. If evaluated carefully, it may improve legal semantic matching, though multilingual support and embedding suitability must still be considered.

### 5. Proposed multilingual hybrid method

The proposed system combines:

- multilingual query translation
- semantic dense retrieval
- approximate nearest neighbour search
- lexical matching
- rule-based legal-domain reranking

This hybrid design can outperform single-method baselines because it combines semantic retrieval with legal cue matching and lexical relevance.

## Important honesty note for the report

Do not claim that blank rows already represent measured results. For BERT, LegalBERT, Sentence-BERT, mBERT, and XLM-R, use one of these two labels in the report:

- `To be experimentally evaluated`
- `Reported from literature` if you cite an external source

## Safe wording for your report

Use wording like:

`Table X compares the proposed multilingual hybrid retrieval system against baseline and reference model families. BM25 and the proposed system were measured on the custom multilingual evaluation benchmark constructed for this project, while the remaining architectures are included as comparison targets for experimental or literature-backed analysis.`
