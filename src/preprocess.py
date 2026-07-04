"""Text preprocessing utilities for legal documents."""

from __future__ import annotations

import re
from typing import Iterable


LEGAL_SYMBOLS_PATTERN = re.compile(r"[^\w\s\.,;:()\-\/\[\]#&]+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
PAGE_NOISE_PATTERN = re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str, lowercase: bool = False) -> str:
    """Clean noisy symbols while preserving common legal markers."""
    text = LEGAL_SYMBOLS_PATTERN.sub(" ", text)
    text = normalize_whitespace(text)
    return text.lower() if lowercase else text


def combine_sentences(sentences: Iterable[str]) -> str:
    """Join sentence lists into a single passage."""
    return normalize_whitespace(" ".join(sentence for sentence in sentences if sentence))


def select_content_sentences(sentences: Iterable[str], max_sentences: int = 4) -> list[str]:
    """Pick the most content-bearing early sentences from a legal text."""
    selected: list[str] = []
    for sentence in sentences:
        sentence = normalize_whitespace(sentence)
        if not sentence:
            continue
        if PAGE_NOISE_PATTERN.search(sentence):
            continue
        if len(sentence) < 40:
            continue
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    if not selected:
        fallback = [normalize_whitespace(sentence) for sentence in sentences if normalize_whitespace(sentence)]
        return fallback[:max_sentences]
    return selected


def truncate_words(text: str, max_words: int = 18) -> str:
    """Shorten text to a fixed number of words for titles/snippets."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def preprocess_case(case: dict, lowercase: bool = False) -> dict:
    """Create cleaned title and body text for a case record."""
    title = clean_text(case.get("title", f"Case {case.get('case_id', '')}"), lowercase)
    body = clean_text(combine_sentences(case.get("sentences", [])), lowercase)
    summary = clean_text(case.get("summary", body[:280]), lowercase)
    retrieval_text = clean_text(case.get("retrieval_text", summary), lowercase)
    citations = [clean_text(item, lowercase=False) for item in case.get("citations", [])]
    return {
        **case,
        "title": title,
        "body_text": body,
        "summary": summary,
        "retrieval_text": retrieval_text,
        "citations": citations,
        "document_text": normalize_whitespace(
            f"{title}. {retrieval_text}. Citations: {'; '.join(citations)}"
        ),
    }


def preprocess_lesicin_record(record: dict, lowercase: bool = False) -> dict:
    """Normalize an official LeSICiN record into the retrieval schema."""
    case_id = record.get("id", "")
    record_type = record.get("record_type", "fact")
    is_section = record_type == "section"
    sentences = record.get("text", []) or []
    citations = record.get("labels", []) or []
    content_sentences = select_content_sentences(sentences, max_sentences=5)

    if is_section:
        title = f"Statute Section {case_id}"
        summary = combine_sentences(content_sentences[:2]) or f"Statutory section {case_id}"
        retrieval_text = combine_sentences(content_sentences[:3]) or summary
    else:
        title = truncate_words(content_sentences[0], max_words=16) if content_sentences else f"Legal Fact {case_id}"
        summary = combine_sentences(content_sentences[:2]) or f"Legal fact pattern {case_id}"
        retrieval_text = combine_sentences(content_sentences[:4]) or summary

    normalized = {
        "case_id": case_id,
        "title": title,
        "sentences": sentences,
        "citations": citations,
        "summary": summary,
        "retrieval_text": retrieval_text,
        "source_type": record_type,
    }
    return preprocess_case(normalized, lowercase=lowercase)
