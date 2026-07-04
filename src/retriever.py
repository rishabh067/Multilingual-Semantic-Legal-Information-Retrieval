"""End-to-end multilingual semantic retrieval pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.bm25_baseline import BM25Baseline
from src.config import get_settings
from src.data_loader import load_jsonl_cases, load_lesicin_corpus
from src.embedder import FallbackEmbeddingModel, SentenceBERTEmbedder
from src.indexer import HNSWIndexer
from src.preprocess import preprocess_case, preprocess_lesicin_record
from src.translator import IndicTranslator
from src.utils import get_logger, load_json, normalize_language_code, save_json


LOGGER = get_logger(__name__)


class LegalSemanticRetriever:
    """Complete multilingual retrieval pipeline."""

    QUERY_EXPANSION_RULES = [
        {
            "trigger_terms": {"dowry", "cruelty", "harassment", "wife", "marriage", "violence", "bride"},
            "expansion_terms": [
                "dowry death",
                "dowry harassment",
                "matrimonial cruelty",
                "mental cruelty",
                "marital harassment",
                "section 498a",
                "section 304b",
            ],
            "citation_targets": ["498A", "304B"],
        },
        {
            "trigger_terms": {"cheating", "forged", "forgery", "receipt", "fraud", "extorting", "money"},
            "expansion_terms": [
                "forged documents",
                "criminal cheating",
                "fake receipt",
                "forged receipt",
                "valuable security",
                "section 420",
                "section 467",
                "section 468",
                "section 471",
                "section 406",
            ],
            "citation_targets": ["420", "467", "468", "471", "406"],
        },
        {
            "trigger_terms": {"contract", "construction", "breach", "compensation", "contractor"},
            "expansion_terms": [
                "contractual breach",
                "damages",
                "contractor dispute",
                "construction agreement",
                "deficiency compensation",
            ],
            "citation_targets": [],
        },
        {
            "trigger_terms": {"termination", "employment", "dismissal", "salary", "justice", "inquiry"},
            "expansion_terms": [
                "service law",
                "departmental inquiry",
                "natural justice",
                "reinstatement",
                "back wages",
                "wrongful dismissal",
            ],
            "citation_targets": [],
        },
        {
            "trigger_terms": {"disability", "access", "transport", "bus", "rights", "public"},
            "expansion_terms": [
                "accessibility rights",
                "public transport",
                "disabled passenger",
                "regional transport officer",
                "driving licence",
            ],
            "citation_targets": [],
        },
        {
            "trigger_terms": {"murder", "homicide", "conviction", "death"},
            "expansion_terms": [
                "section 302",
                "culpable homicide",
                "murder conviction",
                "criminal appeal",
            ],
            "citation_targets": ["302"],
        },
        {
            "trigger_terms": {"accident", "vehicle", "motor", "compensation", "injury", "claim"},
            "expansion_terms": [
                "motor accident claims tribunal",
                "motor vehicles act",
                "injury compensation",
                "permanent disability",
                "section 166",
            ],
            "citation_targets": ["279", "337", "338"],
        },
        {
            "trigger_terms": {"robbery", "loot", "looted", "stolen", "cash", "assault"},
            "expansion_terms": [
                "section 394",
                "section 392",
                "armed robbery",
                "loot cash ornaments",
            ],
            "citation_targets": ["392", "394"],
        },
        {
            "trigger_terms": {"rape", "sexual", "assault", "victim", "minor", "prosecutrix"},
            "expansion_terms": [
                "section 376",
                "sexual assault",
                "victim statement",
                "minor prosecutrix",
            ],
            "citation_targets": ["376"],
        },
        {
            "trigger_terms": {"electricity", "meter", "power", "theft", "tampering"},
            "expansion_terms": [
                "electric meter tampering",
                "dishonestly abstracting power",
                "electricity stolen",
                "power loom factory",
            ],
            "citation_targets": ["379"],
        },
        {
            "trigger_terms": {"property", "sale", "deed", "forged", "transfer", "land"},
            "expansion_terms": [
                "forged sale deed",
                "property transfer fraud",
                "section 420",
                "section 467",
                "section 468",
                "section 471",
            ],
            "citation_targets": ["420", "467", "468", "471"],
        },
    ]

    TOKEN_EQUIVALENTS = {
        "receipts": "receipt",
        "forged": "forgery",
        "forging": "forgery",
        "fraudulent": "fraud",
        "cheated": "cheating",
        "cheat": "cheating",
        "extorting": "extortion",
        "extorted": "extortion",
        "wives": "wife",
        "spouse": "wife",
        "harassed": "harassment",
        "harassing": "harassment",
        "violence": "cruelty",
        "domestic": "matrimonial",
        "dismissed": "dismissal",
        "terminating": "termination",
        "terminated": "termination",
        "employment": "service",
        "job": "service",
        "compensatory": "compensation",
        "disabled": "disability",
        "disabilities": "disability",
        "buses": "bus",
        "vehicles": "vehicle",
        "injuries": "injury",
        "claims": "claim",
        "looted": "robbery",
        "stolen": "theft",
        "victims": "victim",
        "prosecutrixs": "prosecutrix",
        "meters": "meter",
    }

    DOMAIN_BONUS_RULES = [
        {
            "query_terms": {"receipt", "forgery", "fraud", "cheating", "extortion"},
            "record_terms": {"receipt", "forgery", "valuable security", "consumer", "loan application"},
        },
        {
            "query_terms": {"dowry", "cruelty", "matrimonial", "wife", "harassment"},
            "record_terms": {"dowry", "cruelty", "harassment", "death", "wife", "marriage"},
        },
        {
            "query_terms": {"dismissal", "termination", "service", "justice", "inquiry"},
            "record_terms": {
                "dismissal",
                "termination",
                "departmental inquiry",
                "natural justice",
                "salary",
                "reinstatement",
                "back wages",
                "service",
            },
        },
        {
            "query_terms": {"contract", "compensation", "breach", "agreement", "contractor"},
            "record_terms": {"contract", "agreement", "compensation", "damages", "contractor", "construction"},
        },
        {
            "query_terms": {"disability", "access", "bus", "transport", "public"},
            "record_terms": {"disability", "disabled", "access", "bus", "transport", "public"},
        },
        {
            "query_terms": {"murder", "homicide", "conviction", "death"},
            "record_terms": {"murder", "section 302", "convicted", "conviction", "death"},
        },
        {
            "query_terms": {"motor", "accident", "compensation", "vehicle", "injury", "claim"},
            "record_terms": {
                "motor accident",
                "compensation",
                "injury",
                "tribunal",
                "motor vehicles act",
                "disability",
            },
        },
        {
            "query_terms": {"robbery", "loot", "cash", "assault", "hurt"},
            "record_terms": {"robbery", "looted", "cash", "assaulted", "pistol", "ornaments"},
        },
        {
            "query_terms": {"rape", "sexual", "assault", "victim", "minor"},
            "record_terms": {"rape", "sexual assault", "victim", "prosecutrix", "minor"},
        },
        {
            "query_terms": {"electricity", "theft", "meter", "power", "tampering"},
            "record_terms": {"electricity", "meter", "tampered", "stolen", "power", "wire"},
        },
        {
            "query_terms": {"property", "sale", "deed", "transfer", "land", "forgery"},
            "record_terms": {"sale deed", "forged", "transfer", "property", "section 420", "section 467"},
        },
    ]

    def __init__(self, cases_path: str | Path | None = None) -> None:
        self.settings = get_settings()
        self.cases_path = Path(cases_path or self.settings.cases_path)
        self.logger = LOGGER
        self.translator = IndicTranslator()
        self.embedder = SentenceBERTEmbedder()
        self.indexer = HNSWIndexer()
        self.case_records: list[dict[str, Any]] = []
        self.cases_df = pd.DataFrame()
        self.case_id_to_index: dict[str, int] = {}
        self.bm25: BM25Baseline | None = None
        self.focused_embeddings: np.ndarray | None = None
        self.ready = False
        self.prepare_artifacts(force_rebuild=False)

    def _build_runtime_helpers(self) -> None:
        """Build helper structures used during retrieval."""
        self.case_id_to_index = {
            str(record["case_id"]): index for index, record in enumerate(self.case_records)
        }
        self.bm25 = BM25Baseline(self.case_records)

    def _prepare_case_records(self) -> list[dict[str, Any]]:
        use_raw_lesicin = (
            self.settings.use_lesicin_raw
            and self.settings.lesicin_train_path.exists()
            and self.settings.lesicin_dev_path.exists()
            and self.settings.lesicin_test_path.exists()
            and self.settings.lesicin_secs_path.exists()
        )

        if use_raw_lesicin:
            self.logger.info("Using official LeSICiN raw splits from %s", self.settings.lesicin_raw_dir)
            raw_df = load_lesicin_corpus(
                self.settings.lesicin_train_path,
                self.settings.lesicin_dev_path,
                self.settings.lesicin_test_path,
                self.settings.lesicin_secs_path,
            )
            records = [
                preprocess_lesicin_record(row, self.settings.use_lowercase)
                for row in raw_df.to_dict("records")
            ]
        else:
            raw_df = load_jsonl_cases(self.cases_path)
            records = [preprocess_case(row, self.settings.use_lowercase) for row in raw_df.to_dict("records")]

        self.case_records = records
        self.cases_df = pd.DataFrame(records)
        return records

    def prepare_artifacts(self, force_rebuild: bool = False) -> None:
        """Prepare embeddings and index, reusing caches when present."""
        records = self._prepare_case_records()
        embeddings_exist = self.settings.embeddings_path.exists()
        focused_embeddings_exist = self.settings.focused_embeddings_path.exists()
        metadata_exists = self.settings.metadata_path.exists()
        index_exists = self.settings.hnsw_index_path.exists() or self.settings.hnsw_index_path.with_suffix(".npz").exists()

        if not force_rebuild and embeddings_exist and metadata_exists and index_exists:
            embeddings = self.embedder.load_embeddings(self.settings.embeddings_path)
            if focused_embeddings_exist:
                self.focused_embeddings = self.embedder.load_embeddings(self.settings.focused_embeddings_path)
            self.case_records = load_json(self.settings.metadata_path)
            self.cases_df = pd.DataFrame(self.case_records)
            if isinstance(self.embedder.model, FallbackEmbeddingModel):
                self.embedder.model.encode([record["document_text"] for record in self.case_records])
            self.indexer.load_index(dimension=embeddings.shape[1], path=self.settings.hnsw_index_path)
            self.indexer.vectors = embeddings
            if self.focused_embeddings is None:
                focused_texts = [self._build_focused_candidate_text(record) for record in self.case_records]
                self.focused_embeddings = self.embedder.encode_documents(focused_texts)
                self.embedder.save_embeddings(self.focused_embeddings, self.settings.focused_embeddings_path)
            self._build_runtime_helpers()
            self.ready = True
            self.logger.info("Reused cached embeddings and index")
            return

        texts = [record["document_text"] for record in records]
        embeddings = self.embedder.encode_documents(texts)
        self.embedder.save_embeddings(embeddings, self.settings.embeddings_path)
        focused_texts = [self._build_focused_candidate_text(record) for record in records]
        self.focused_embeddings = self.embedder.encode_documents(focused_texts)
        self.embedder.save_embeddings(self.focused_embeddings, self.settings.focused_embeddings_path)
        save_json(records, self.settings.metadata_path)
        self.indexer.build_hnsw_index(embeddings)
        self.indexer.save_index(self.settings.hnsw_index_path)
        self._build_runtime_helpers()
        self.ready = True
        self.logger.info("Prepared fresh retrieval artifacts")

    @staticmethod
    def _tokenize_for_overlap(text: str) -> set[str]:
        """Tokenize text for simple lexical overlap scoring."""
        return {
            LegalSemanticRetriever._normalize_token(token)
            for token in text.split()
            if len(LegalSemanticRetriever._normalize_token(token)) > 2
        }

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        """Collapse small token variants into a common legal form."""
        token = token.strip(".,;:()[]{}\"'").lower()
        if token.endswith("'s"):
            token = token[:-2]
        if token in cls.TOKEN_EQUIVALENTS:
            return cls.TOKEN_EQUIVALENTS[token]
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        return cls.TOKEN_EQUIVALENTS.get(token, token)

    @staticmethod
    def _normalize_scores(score_map: dict[int, float]) -> dict[int, float]:
        """Min-max normalize scores into [0, 1]."""
        if not score_map:
            return {}
        values = list(score_map.values())
        low = min(values)
        high = max(values)
        if high - low < 1e-9:
            return {key: 1.0 for key in score_map}
        return {key: (value - low) / (high - low) for key, value in score_map.items()}

    def _pick_display_title(self, record: dict[str, Any]) -> str:
        """Create a more meaningful display title than the generic fact id."""
        title = record.get("title", "").strip()
        if not title.startswith("Legal Fact"):
            return title

        sentences = record.get("sentences", []) or []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                words = sentence.split()
                return " ".join(words[:16]) + ("..." if len(words) > 16 else "")
        return title

    def _pick_snippet(self, record: dict[str, Any], translated_query: str) -> str:
        """Choose the sentence(s) most aligned with the query."""
        sentences = [sentence.strip() for sentence in (record.get("sentences", []) or []) if sentence.strip()]
        if not sentences:
            return record.get("summary", "")

        query_tokens = self._tokenize_for_overlap(translated_query)
        ranked_sentences = sorted(
            sentences,
            key=lambda sentence: len(query_tokens & self._tokenize_for_overlap(sentence)),
            reverse=True,
        )
        top_sentences = ranked_sentences[:2]
        snippet = " ".join(top_sentences)
        return snippet if snippet else record.get("summary", "")

    def _build_focused_candidate_text(self, record: dict[str, Any]) -> str:
        """Build a concise reranking representation for a candidate record."""
        title = record.get("title", "")
        summary = record.get("summary", "")
        citations = "; ".join(record.get("citations", [])[:6])
        return f"{title}. {summary}. Citations: {citations}".strip()

    @staticmethod
    def _cosine_similarity(query_vector: np.ndarray, candidate_vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity for normalized vectors."""
        if candidate_vectors.size == 0:
            return np.array([], dtype=np.float32)
        return candidate_vectors @ query_vector

    def _phrase_match_score(self, query_tokens: set[str], record: dict[str, Any]) -> float:
        """Boost candidates that hit high-signal query concepts in title or summary."""
        title_tokens = self._tokenize_for_overlap(record.get("title", ""))
        summary_tokens = self._tokenize_for_overlap(record.get("summary", ""))
        doc_tokens = title_tokens | summary_tokens
        if not query_tokens or not doc_tokens:
            return 0.0

        exact_overlap = len(query_tokens & doc_tokens) / len(query_tokens)
        prominent_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
        return (0.6 * exact_overlap) + (0.4 * prominent_overlap)

    def _domain_bonus_score(
        self,
        query_tokens: set[str],
        expanded_query: str,
        record: dict[str, Any],
    ) -> float:
        """Boost records that satisfy intent-specific legal concepts."""
        record_text = self._build_focused_candidate_text(record).lower()
        expanded_query = expanded_query.lower()
        best_score = 0.0

        for rule in self.DOMAIN_BONUS_RULES:
            triggered_terms = [
                term
                for term in rule["query_terms"]
                if term in query_tokens or term in expanded_query
            ]
            if not triggered_terms:
                continue

            matched_terms = [term for term in rule["record_terms"] if term in record_text]
            if not matched_terms:
                continue

            trigger_ratio = len(triggered_terms) / len(rule["query_terms"])
            match_ratio = len(matched_terms) / len(rule["record_terms"])
            best_score = max(best_score, (0.4 * trigger_ratio) + (0.6 * match_ratio))

        return best_score

    def _expand_query(self, translated_query: str) -> tuple[str, set[str]]:
        """Expand the translated query with legal-domain hints."""
        query_tokens = self._tokenize_for_overlap(translated_query)
        expansion_terms: list[str] = []
        citation_targets: set[str] = set()

        for rule in self.QUERY_EXPANSION_RULES:
            if query_tokens & rule["trigger_terms"]:
                expansion_terms.extend(rule["expansion_terms"])
                citation_targets.update(rule["citation_targets"])

        if not expansion_terms:
            return translated_query, citation_targets

        expanded_query = f"{translated_query}. {' '.join(expansion_terms)}"
        return expanded_query, citation_targets

    @staticmethod
    def _citation_match_score(citations: list[str], citation_targets: set[str]) -> float:
        """Score citation overlap between the query intent and a record."""
        if not citation_targets:
            return 0.0
        joined = " ".join(citations).upper()
        matches = sum(target.upper() in joined for target in citation_targets)
        return matches / len(citation_targets)

    def retrieve(
        self,
        query_text: str,
        language: str | None = None,
        top_k: int = 5,
        translate_results: bool = True,
    ) -> dict[str, Any]:
        """Run the retrieval pipeline for a query."""
        if not self.ready:
            self.prepare_artifacts(force_rebuild=False)

        language = normalize_language_code(language or self.translator.detect_language(query_text))
        translated_query = self.translator.translate_to_english(query_text, language)
        expanded_query, citation_targets = self._expand_query(translated_query)
        query_vector = self.embedder.encode_query(expanded_query)
        semantic_pool = max(top_k, self.settings.semantic_candidate_pool)
        bm25_pool = max(top_k, self.settings.bm25_candidate_pool)
        indices, scores = self.indexer.search(query_vector, top_k=semantic_pool)

        semantic_score_map = {int(index): float(score) for index, score in zip(indices, scores)}
        semantic_norm = self._normalize_scores(semantic_score_map)

        bm25_results = self.bm25.bm25_search(expanded_query, top_k=bm25_pool) if self.bm25 else []
        bm25_score_map = {
            self.case_id_to_index[result.case_id]: float(result.score)
            for result in bm25_results
            if result.case_id in self.case_id_to_index
        }
        bm25_norm = self._normalize_scores(bm25_score_map)

        query_tokens = self._tokenize_for_overlap(expanded_query)
        candidate_indices = set(semantic_score_map) | set(bm25_score_map)
        focused_pool = sorted(
            candidate_indices,
            key=lambda idx: semantic_norm.get(idx, 0.0) + bm25_norm.get(idx, 0.0),
            reverse=True,
        )[: self.settings.focused_rerank_pool]
        focused_semantic_norm: dict[int, float] = {}
        if focused_pool:
            focused_embeddings = self.focused_embeddings[focused_pool] if self.focused_embeddings is not None else np.zeros((0, 384), dtype=np.float32)
            focused_scores = self._cosine_similarity(query_vector, focused_embeddings)
            focused_semantic_norm = self._normalize_scores(
                {index: float(score) for index, score in zip(focused_pool, focused_scores)}
            )

        ranked_candidates: list[tuple[int, float]] = []
        combined_score_map: dict[int, float] = {}

        for index in candidate_indices:
            record = self.case_records[index]
            doc_tokens = self._tokenize_for_overlap(record.get("document_text", ""))
            title_tokens = self._tokenize_for_overlap(record.get("title", ""))
            overlap_score = (
                len(query_tokens & doc_tokens) / max(1, len(query_tokens))
                if query_tokens
                else 0.0
            )
            title_overlap_score = (
                len(query_tokens & title_tokens) / max(1, len(query_tokens))
                if query_tokens
                else 0.0
            )
            phrase_match_score = self._phrase_match_score(query_tokens, record)
            domain_bonus_score = self._domain_bonus_score(query_tokens, expanded_query, record)
            citation_score = self._citation_match_score(record.get("citations", []), citation_targets)
            combined_score = (
                self.settings.semantic_weight * semantic_norm.get(index, 0.0)
                + self.settings.bm25_weight * bm25_norm.get(index, 0.0)
                + self.settings.overlap_weight * overlap_score
                + self.settings.title_overlap_weight * title_overlap_score
                + self.settings.focused_semantic_weight * focused_semantic_norm.get(index, 0.0)
                + self.settings.phrase_match_weight * phrase_match_score
                + 0.18 * domain_bonus_score
                + self.settings.citation_weight * citation_score
            )
            if record.get("source_type") == "section":
                combined_score *= 0.92
            combined_score_map[index] = combined_score
            ranked_candidates.append((index, combined_score))

        ranked_candidates.sort(key=lambda item: item[1], reverse=True)
        combined_score_norm = self._normalize_scores(combined_score_map)
        selected_indices = [index for index, _ in ranked_candidates[:top_k]]
        selected_raw_scores = [score for _, score in ranked_candidates[:top_k]]
        selected_scores = [combined_score_norm.get(index, 0.0) for index in selected_indices]

        selected_records = [self.case_records[index] for index in selected_indices]
        titles = [self._pick_display_title(record) for record in selected_records]
        summaries = [self._pick_snippet(record, translated_query) for record in selected_records]

        if translate_results:
            translated_titles = self.translator.translate_texts_from_english(titles, language)
            translated_summaries = self.translator.translate_texts_from_english(summaries, language)
        else:
            translated_titles = titles
            translated_summaries = summaries

        results = []
        for rank, (record, title, summary, score, raw_score) in enumerate(
            zip(selected_records, translated_titles, translated_summaries, selected_scores, selected_raw_scores),
            start=1,
        ):
            results.append(
                {
                    "rank": rank,
                    "case_id": record["case_id"],
                    "title": title,
                    "summary": summary,
                    "score": round(float(score), 4),
                    "raw_score": round(float(raw_score), 4),
                    "semantic_similarity": round(float(semantic_score_map.get(self.case_id_to_index[record["case_id"]], 0.0)), 4),
                    "citations": record.get("citations", []),
                }
            )

        return {
            "query": query_text,
            "language": language,
            "translated_query": translated_query,
            "results": results,
        }
