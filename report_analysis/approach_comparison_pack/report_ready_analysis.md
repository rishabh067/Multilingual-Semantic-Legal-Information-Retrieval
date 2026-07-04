# Comparative Analysis of Retrieval Approaches

## Result Summary

The project compares seven retrieval approaches: BM25, BERT, LegalBERT, Sentence-BERT, mBERT, XLM-R, and the proposed multilingual hybrid retrieval system. BM25 and the proposed method have measured values on the custom 100-query multilingual benchmark. BERT, LegalBERT, Sentence-BERT, mBERT, and XLM-R are included as comparison approaches for future experimental evaluation or literature-backed discussion.

| Approach | Status | P@1 | P@5 | NDCG@3 | Main interpretation |
|---|---:|---:|---:|---:|---|
| BM25 | Measured | 0.0200 | 0.0300 | 0.0129 | Lexical matching performs poorly on multilingual legal queries. |
| BERT | Not measured | - | - | - | General contextual encoder, but not ideal as a retrieval embedding model without adaptation. |
| LegalBERT | Not measured | - | - | - | Useful for legal-domain understanding, especially English legal text. |
| Sentence-BERT | Not separately measured | - | - | - | Strong semantic retrieval baseline because it is trained for sentence similarity. |
| mBERT | Not measured | - | - | - | Multilingual encoder suitable for cross-lingual comparison. |
| XLM-R | Not measured | - | - | - | Stronger multilingual transformer candidate than mBERT in many cross-lingual settings. |
| Proposed Multilingual Hybrid Retrieval | Measured | 0.8200 | 0.3920 | 0.6104 | Best measured approach due to combined translation, semantic search, lexical evidence, and legal reranking. |

## Analysis

BM25 acts as the classical retrieval baseline. Its low measured scores show the limitation of exact term matching for multilingual legal search. Legal queries often contain translated, paraphrased, or layperson wording, while case documents contain formal legal language. Because BM25 depends on token overlap, it struggles when the query and relevant case express the same concept using different words.

BERT improves over lexical matching conceptually because it captures contextual meaning. However, vanilla BERT is not directly optimized for producing sentence embeddings for retrieval. Without fine-tuning or a pairwise reranking setup, BERT embeddings may not rank legal documents reliably.

LegalBERT is important as a domain-specific comparison because it is trained on legal text and can better represent legal vocabulary, statutory phrases, and formal case language. Its main limitation for this project is multilingual access: it is strongest for English legal documents, so Indian-language queries still require translation or multilingual adaptation.

Sentence-BERT is a stronger dense retrieval baseline than vanilla BERT because it is designed for sentence similarity. The implemented project already uses a Sentence-BERT style embedding model inside the proposed system, but the full proposed method is not only Sentence-BERT. It adds query translation, HNSW vector search, BM25 evidence, overlap scoring, citation matching, and legal-domain reranking.

mBERT and XLM-R are multilingual transformer baselines. They are useful for evaluating whether direct multilingual representations can reduce reliance on translation. XLM-R is generally a stronger multilingual comparison candidate, while mBERT is a widely used baseline. Both should be evaluated on the same benchmark before claiming measured superiority.

The proposed multilingual hybrid retrieval system obtains the best measured performance in this project. Its P@1 of 0.8200 and NDCG@3 of 0.6104 indicate that the system is often able to place highly relevant cases near the top of the ranked list. The improvement over BM25 is explained by the combined use of multilingual query handling, semantic embeddings, lexical matching, citation cues, and legal-domain reranking rules.

## Report Wording

Use this wording to stay accurate:

`Table X compares the proposed multilingual hybrid retrieval system with classical, transformer-based, multilingual, and legal-domain retrieval approaches. BM25 and the proposed method were measured on the custom 100-query benchmark used in this project. Other transformer models are included as reference approaches and should be reported as future experimental baselines unless separate experiments or cited literature values are added.`

## Conclusion

The comparison supports the project objective: multilingual legal information retrieval benefits from combining semantic retrieval with translation and legal-domain signals. A pure lexical baseline is insufficient, while a hybrid pipeline provides better top-ranked relevance for multilingual legal queries.
