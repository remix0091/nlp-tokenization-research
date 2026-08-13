from __future__ import annotations

import math
from typing import Any
import math


def safe_words_count(text: str) -> int:
    """
    Считает количество слов в тексте.

    Используется для метрики tokens_per_word.
    """

    return len([word for word in text.strip().split() if word])


def compression_metrics(
    tokenizer: Any,
    texts: list[str],
) -> dict[str, float]:
    """
    Считает метрики токенизации.

    Метрики:
    - tokens_per_char: сколько токенов приходится на один символ;
    - tokens_per_word: сколько токенов приходится на одно слово;
    - avg_token_len: средняя длина токена;
    - n_texts: количество текстовых фрагментов;
    - n_tokens: общее количество токенов.
    """

    total_chars = 0
    total_words = 0
    total_tokens = 0
    total_token_chars = 0

    for text in texts:
        text = text.strip()

        if not text:
            continue

        encoded = tokenizer.encode(text)
        tokens = encoded.tokens

        total_chars += len(text)
        total_words += safe_words_count(text)
        total_tokens += len(tokens)
        total_token_chars += sum(len(token) for token in tokens)

    if total_chars == 0:
        return {
            "tokens_per_char": math.nan,
            "tokens_per_word": math.nan,
            "avg_token_len": math.nan,
            "n_texts": 0,
            "n_tokens": 0,
        }

    return {
        "tokens_per_char": total_tokens / total_chars,
        "tokens_per_word": total_tokens / total_words if total_words > 0 else math.nan,
        "avg_token_len": total_token_chars / total_tokens if total_tokens > 0 else math.nan,
        "n_texts": len(texts),
        "n_tokens": total_tokens,
    }

def add_suffix_to_metrics(
    metrics: dict[str, float],
    suffix: str,
) -> dict[str, float]:
    """
    Добавляет суффикс к названиям метрик.

    Например:
        tokens_per_char -> tokens_per_char_en
    """

    return {
        f"{key}_{suffix}": value
        for key, value in metrics.items()
    }


def compression_metrics_by_language(
    tokenizer,
    all_texts: list[str],
    en_texts: list[str],
    target_texts: list[str],
) -> dict[str, float]:
    """
    Считает метрики сжатия:
    - по всему test;
    - отдельно по английским фрагментам;
    - отдельно по целевым фрагментам;
    - разницу сжатия между target и en.

    Основная разница:
        compression_gap_tokens_per_char =
            tokens_per_char_target - tokens_per_char_en
    """

    all_metrics = compression_metrics(
        tokenizer=tokenizer,
        texts=all_texts,
    )

    en_metrics = compression_metrics(
        tokenizer=tokenizer,
        texts=en_texts,
    )

    target_metrics = compression_metrics(
        tokenizer=tokenizer,
        texts=target_texts,
    )

    result = {
        **all_metrics,
        **add_suffix_to_metrics(en_metrics, "en"),
        **add_suffix_to_metrics(target_metrics, "target"),
    }

    tokens_per_char_en = result.get("tokens_per_char_en")
    tokens_per_char_target = result.get("tokens_per_char_target")

    tokens_per_char_all = result.get("tokens_per_char")

    compression_factor_char = safe_inverse(tokens_per_char_all)
    compression_factor_char_en = safe_inverse(tokens_per_char_en)
    compression_factor_char_target = safe_inverse(tokens_per_char_target)

    result["compression_factor_char"] = compression_factor_char
    result["compression_factor_char_en"] = compression_factor_char_en
    result["compression_factor_char_target"] = compression_factor_char_target

    if (
        not math.isnan(compression_factor_char_en)
        and not math.isnan(compression_factor_char_target)
    ):
        result["compression_penalty_char"] = (
            compression_factor_char_en - compression_factor_char_target
        )
    else:
        result["compression_penalty_char"] = math.nan

    tokens_per_word_en = result.get("tokens_per_word_en")
    tokens_per_word_target = result.get("tokens_per_word_target")

    if (
        tokens_per_char_en is not None
        and tokens_per_char_target is not None
        and not math.isnan(tokens_per_char_en)
        and not math.isnan(tokens_per_char_target)
    ):
        result["compression_gap_tokens_per_char"] = (
            tokens_per_char_target - tokens_per_char_en
        )
    else:
        result["compression_gap_tokens_per_char"] = math.nan

    if (
        tokens_per_word_en is not None
        and tokens_per_word_target is not None
        and not math.isnan(tokens_per_word_en)
        and not math.isnan(tokens_per_word_target)
    ):
        result["compression_gap_tokens_per_word"] = (
            tokens_per_word_target - tokens_per_word_en
        )
    else:
        result["compression_gap_tokens_per_word"] = math.nan

    return result

def safe_inverse(value: float) -> float:
    """
    Безопасно считает 1 / value.

    Используется для compression factor:
        compression_factor_char = 1 / tokens_per_char
    """

    if value is None:
        return math.nan

    if math.isnan(value):
        return math.nan

    if value == 0:
        return math.nan

    return 1 / value