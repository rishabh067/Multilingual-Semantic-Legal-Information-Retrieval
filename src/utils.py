"""Utility helpers for logging, serialization, and runtime environment setup."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Create or reuse a configured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def save_json(data: Any, path: Path) -> None:
    """Save JSON to disk with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    """Load JSON from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division by zero in metric calculations."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def normalize_language_code(language: str | None) -> str:
    """Normalize language inputs into short lowercase codes."""
    mapping = {
        "english": "en",
        "hindi": "hi",
        "gujarati": "gu",
        "tamil": "ta",
        "telugu": "te",
        "marathi": "mr",
        "bengali": "bn",
        "punjabi": "pa",
        "kannada": "kn",
        "malayalam": "ml",
    }
    if not language:
        return "en"
    language = language.strip().lower()
    return mapping.get(language, language)


def ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def clear_dead_proxy_env() -> None:
    """Remove obviously broken proxy variables that block local downloads."""
    bad_proxy_values = {"http://127.0.0.1:9", "https://127.0.0.1:9"}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(key)
        if value in bad_proxy_values:
            os.environ.pop(key, None)
