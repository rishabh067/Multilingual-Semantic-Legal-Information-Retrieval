"""Translation helpers using IndicTrans2 with safe offline fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.utils import clear_dead_proxy_env, get_logger, normalize_language_code


LOGGER = get_logger(__name__)


UNICODE_LANGUAGE_RANGES = {
    "hi": ((0x0900, 0x097F),),
    "mr": ((0x0900, 0x097F),),
    "bn": ((0x0980, 0x09FF),),
    "pa": ((0x0A00, 0x0A7F),),
    "gu": ((0x0A80, 0x0AFF),),
    "or": ((0x0B00, 0x0B7F),),
    "ta": ((0x0B80, 0x0BFF),),
    "te": ((0x0C00, 0x0C7F),),
    "kn": ((0x0C80, 0x0CFF),),
    "ml": ((0x0D00, 0x0D7F),),
}

LANGUAGE_TO_FLORES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "gu": "guj_Gujr",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "or": "ory_Orya",
}

FALLBACK_TO_ENGLISH = {
    "gu": {
        "\u0a9b\u0ac7\u0aa4\u0ab0\u0aaa\u0abf\u0a82\u0aa1\u0ac0 \u0a95\u0ac7\u0ab8": "cheating forged receipts fake investment land allotment case",
        "\u0a9b\u0ac7\u0aa4\u0ab0\u0aaa\u0abf\u0a82\u0aa1\u0ac0 \u0aae\u0abe\u0a9f\u0ac7 \u0aa8\u0a95\u0ab2\u0ac0 \u0ab0\u0ab8\u0ac0\u0aa6\u0acb \u0ab5\u0aa1\u0ac7 \u0aaa\u0ac8\u0ab8\u0abe \u0ab5\u0ab8\u0ac2\u0ab2\u0ab5\u0abe\u0aa8\u0abe \u0a95\u0ac7\u0ab8": "case about cheating through fake receipts to collect money",
    },
    "hi": {
        "\u0926\u0939\u0947\u091c \u0914\u0930 \u0918\u0930\u0947\u0932\u0942 \u0915\u094d\u0930\u0942\u0930\u0924\u093e \u0915\u093e \u092e\u093e\u092e\u0932\u093e": "dowry and domestic cruelty case",
    },
    "ta": {
        "\u0b95\u0b9f\u0bcd\u0b9f\u0bbf\u0b9f \u0b92\u0baa\u0bcd\u0baa\u0ba8\u0bcd\u0ba4 \u0bae\u0bc0\u0bb1\u0bb2\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0b87\u0bb4\u0baa\u0bcd\u0baa\u0bc0\u0b9f\u0bc1": "construction contract breach and compensation",
    },
    "mr": {
        "\u0928\u094b\u0915\u0930\u0940\u0924 \u092c\u0947\u0915\u093e\u092f\u0926\u0947\u0936\u0940\u0930 \u092c\u0921\u0924\u0930\u094d\u092b\u0940 \u0906\u0923\u093f \u0928\u0948\u0938\u0930\u094d\u0917\u093f\u0915 \u0928\u094d\u092f\u093e\u092f": "illegal termination from employment and natural justice",
    },
    "bn": {
        "\u09aa\u09be\u09ac\u09b2\u09bf\u0995 \u09ac\u09be\u09b8\u09c7 \u09aa\u09cd\u09b0\u09a4\u09bf\u09ac\u09a8\u09cd\u09a7\u09c0 \u09aa\u09cd\u09b0\u09ac\u09c7\u09b6\u09be\u09a7\u09bf\u0995\u09be\u09b0 \u09b8\u0982\u0995\u09cd\u09b0\u09be\u09a8\u09cd\u09a4 \u09ae\u09be\u09ae\u09b2\u09be": "case about disability access in public buses",
    },
}

FALLBACK_FROM_ENGLISH = {
    "gu": {"cheating": "\u0a9b\u0ac7\u0aa4\u0ab0\u0aaa\u0abf\u0a82\u0aa1\u0ac0", "case": "\u0a95\u0ac7\u0ab8", "summary": "\u0ab8\u0abe\u0ab0\u0abe\u0a82\u0ab6"},
    "hi": {"dowry": "\u0926\u0939\u0947\u091c", "case": "\u092e\u093e\u092e\u0932\u093e", "summary": "\u0938\u093e\u0930\u093e\u0902\u0936"},
    "ta": {"contract": "\u0b92\u0baa\u0bcd\u0baa\u0ba8\u0bcd\u0ba4\u0bae\u0bcd", "case": "\u0bb5\u0bb4\u0b95\u0bcd\u0b95\u0bc1", "summary": "\u0b9a\u0bc1\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bae\u0bcd"},
}


@dataclass
class TranslatorBackend:
    """Wrap a translation backend."""

    tokenizer: Any | None = None
    model: Any | None = None
    processor: Any | None = None
    available: bool = False
    error: str | None = None
    model_dir: Path | None = None


class IndicTranslator:
    """Bidirectional translation with IndicTrans2 and fallbacks."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = LOGGER
        self.device = "cpu"
        self._indic_to_en = TranslatorBackend(model_dir=self.settings.indic_to_en_model_dir)
        self._en_to_indic = TranslatorBackend(model_dir=self.settings.en_to_indic_model_dir)
        self._translation_cache: dict[tuple[str, str, str], str] = {}
        if not self.settings.force_fallback_translation:
            self._load_models()

    def _load_single_backend(self, model_dir: Path) -> TranslatorBackend:
        backend = TranslatorBackend(model_dir=model_dir)
        if not model_dir.exists():
            backend.error = f"Model directory not found: {model_dir}"
            return backend

        try:
            clear_dead_proxy_env()
            import torch
            from IndicTransToolkit.processor import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            backend.tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                local_files_only=True,
            )
            model_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
            }
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            backend.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_dir),
                **model_kwargs,
            ).to(self.device)
            backend.model.eval()
            backend.processor = IndicProcessor(inference=True)
            backend.available = True
            return backend
        except Exception as exc:  # pragma: no cover
            backend.error = str(exc)
            self.logger.warning("Could not load IndicTrans2 model from %s: %s", model_dir, exc)
            return backend

    def _load_models(self) -> None:
        """Try loading both IndicTrans2 directions from local cache."""
        self._indic_to_en = self._load_single_backend(self.settings.indic_to_en_model_dir)
        self._en_to_indic = self._load_single_backend(self.settings.en_to_indic_model_dir)
        if self._indic_to_en.available and self._en_to_indic.available:
            self.logger.info("IndicTrans2 models loaded successfully from local cache")

    def detect_language(self, text: str) -> str:
        """Detect language via unicode script ranges with an English fallback."""
        for char in text:
            code_point = ord(char)
            for language, ranges in UNICODE_LANGUAGE_RANGES.items():
                if any(start <= code_point <= end for start, end in ranges):
                    return language
        return "en"

    def _model_translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        backend: TranslatorBackend,
    ) -> list[str]:
        """Run IndicTrans2 translation for a batch of texts."""
        if not backend.available or backend.processor is None or backend.model is None or backend.tokenizer is None:
            return texts

        import torch

        batch = backend.processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = backend.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        with torch.inference_mode():
            generated_tokens = backend.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=self.settings.translation_num_beams,
            )

        decoded = backend.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return backend.processor.postprocess_batch(decoded, lang=tgt_lang)

    def _translate_with_cache(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        backend: TranslatorBackend,
    ) -> list[str]:
        """Translate a batch, reusing cached results when available."""
        results = [""] * len(texts)
        missing: list[str] = []
        missing_indices: list[int] = []

        for index, text in enumerate(texts):
            cache_key = (src_lang, tgt_lang, text)
            if cache_key in self._translation_cache:
                results[index] = self._translation_cache[cache_key]
            else:
                missing.append(text)
                missing_indices.append(index)

        if missing:
            batch_size = self.settings.translation_batch_size
            translated_missing: list[str] = []
            for start in range(0, len(missing), batch_size):
                translated_missing.extend(
                    self._model_translate_batch(
                        missing[start : start + batch_size],
                        src_lang,
                        tgt_lang,
                        backend,
                    )
                )

            for index, original, translated in zip(missing_indices, missing, translated_missing):
                cache_key = (src_lang, tgt_lang, original)
                self._translation_cache[cache_key] = translated
                results[index] = translated

        return results

    def translate_to_english(self, text: str, source_lang: str | None = None) -> str:
        """Translate Indic-language text into English."""
        source_lang = normalize_language_code(source_lang or self.detect_language(text))
        if source_lang == "en":
            return text

        src = LANGUAGE_TO_FLORES.get(source_lang)
        tgt = LANGUAGE_TO_FLORES["en"]
        if src and self._indic_to_en.available:
            try:
                return self._translate_with_cache([text], src, tgt, self._indic_to_en)[0]
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Indic->EN translation failed, using fallback: %s", exc)

        return FALLBACK_TO_ENGLISH.get(source_lang, {}).get(text, text)

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate English output into a target Indic language."""
        target_lang = normalize_language_code(target_lang)
        if target_lang == "en":
            return text

        src = LANGUAGE_TO_FLORES["en"]
        tgt = LANGUAGE_TO_FLORES.get(target_lang)
        if tgt and self._en_to_indic.available:
            try:
                return self._translate_with_cache([text], src, tgt, self._en_to_indic)[0]
            except Exception as exc:  # pragma: no cover
                self.logger.warning("EN->Indic translation failed, using fallback: %s", exc)

        translated = text
        for source, target in FALLBACK_FROM_ENGLISH.get(target_lang, {}).items():
            translated = translated.replace(source, target).replace(source.title(), target)
        return translated

    def translate_texts_from_english(self, texts: list[str], target_lang: str) -> list[str]:
        """Translate a batch of English texts into the target language."""
        target_lang = normalize_language_code(target_lang)
        if target_lang == "en":
            return texts

        src = LANGUAGE_TO_FLORES["en"]
        tgt = LANGUAGE_TO_FLORES.get(target_lang)
        if tgt and self._en_to_indic.available:
            try:
                return self._translate_with_cache(texts, src, tgt, self._en_to_indic)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Batch EN->Indic translation failed, using fallback: %s", exc)

        return [self.translate_from_english(text, target_lang) for text in texts]
