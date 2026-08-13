from __future__ import annotations

import random
import re

import langid


def script_tag(text: str) -> str:
    """
    Грубое определение письменности текста.
    """

    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"

    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"

    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"

    if re.search(r"[\u0400-\u04ff]", text):
        return "cyr"

    if re.search(r"[A-Za-z]", text):
        return "lat"

    return "other"


def detect_english_or_target(text: str, target_lang: str) -> str:
    """
    Определяет, относится ли текст к английскому или целевому языку.

    Возвращает:
        "en"        — английский;
        target_lang — целевой язык;
        "unknown"   — не удалось уверенно определить.
    """

    text = text.strip()

    if not text:
        return "unknown"

    tag = script_tag(text)

    # Японский, китайский, корейский
    if target_lang in ("ja", "zh-cn", "ko"):
        if target_lang == "ja" and tag == "ja":
            return target_lang

        if target_lang == "zh-cn" and tag == "zh":
            return target_lang

        if target_lang == "ko" and tag == "ko":
            return target_lang

        if tag == "lat":
            detected_lang, _ = langid.classify(text)

            if detected_lang == "en":
                return "en"

        return "unknown"

    # Кириллические языки
    if target_lang in ("ru", "bg"):
        if tag == "cyr":
            return target_lang

        if tag == "lat":
            detected_lang, _ = langid.classify(text)

            if detected_lang == "en":
                return "en"

            return "unknown"

        return "unknown"

    # Латинские целевые языки: de, nl, da, cs и т.п.
    if tag == "lat":
        detected_lang, _ = langid.classify(text)

        if detected_lang == "en":
            return "en"

        if detected_lang == target_lang:
            return target_lang

        return "unknown"

    return "unknown"


def split_texts_by_language(
    texts: list[str],
    target_lang: str,
) -> tuple[list[str], list[str], list[str]]:
    """
    Делит тексты на:
    - english_texts;
    - target_texts;
    - unknown_texts.
    """

    english_texts = []
    target_texts = []
    unknown_texts = []

    for text in texts:
        detected = detect_english_or_target(
            text=text,
            target_lang=target_lang,
        )

        if detected == "en":
            english_texts.append(text)

        elif detected == target_lang:
            target_texts.append(text)

        else:
            unknown_texts.append(text)

    return english_texts, target_texts, unknown_texts

def stratified_train_test_split_by_language(
    texts: list[str],
    target_lang: str,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str], list[str], dict]:
    """
    Делит корпус на train/test с сохранением языковых групп.

    Возвращает:
    - train_texts;
    - test_texts;
    - test_en_texts;
    - test_target_texts;
    - test_unknown_texts;
    - split_stats.

    Это нужно для корректного расчёта:
        gap_loss = loss_target - loss_en
    """

    rng = random.Random(seed)

    english_texts, target_texts, unknown_texts = split_texts_by_language(
        texts=texts,
        target_lang=target_lang,
    )

    def split_group(items: list[str]) -> tuple[list[str], list[str]]:
        items = list(items)
        rng.shuffle(items)

        if len(items) == 0:
            return [], []

        # Если всего один элемент, его нельзя честно разделить на train и test.
        # Оставляем его в train.
        if len(items) == 1:
            return items, []

        n_test = round(len(items) * test_fraction)

        # Гарантируем хотя бы 1 test-элемент,
        # но не забираем всю группу в test.
        n_test = max(1, n_test)
        n_test = min(len(items) - 1, n_test)

        test_items = items[:n_test]
        train_items = items[n_test:]

        return train_items, test_items

    train_en, test_en = split_group(english_texts)
    train_target, test_target = split_group(target_texts)
    train_unknown, test_unknown = split_group(unknown_texts)

    train_texts = train_en + train_target + train_unknown
    test_texts = test_en + test_target + test_unknown

    rng.shuffle(train_texts)
    rng.shuffle(test_texts)

    split_stats = {
        "total_texts": len(texts),

        "total_en_texts": len(english_texts),
        "total_target_texts": len(target_texts),
        "total_unknown_texts": len(unknown_texts),

        "train_texts": len(train_texts),
        "test_texts": len(test_texts),

        "train_en_texts": len(train_en),
        "test_en_texts": len(test_en),

        "train_target_texts": len(train_target),
        "test_target_texts": len(test_target),

        "train_unknown_texts": len(train_unknown),
        "test_unknown_texts": len(test_unknown),
    }

    return (
        train_texts,
        test_texts,
        test_en,
        test_target,
        test_unknown,
        split_stats,
    )