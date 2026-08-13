from __future__ import annotations

import ast
import random
import re
from pathlib import Path
from typing import Iterable

from bs4 import UnicodeDammit
from ftfy import fix_text


def force_decode(text):
    """
    Приводит входной текст к нормальной строке.

    Эта функция нужна, потому что в корпусах иногда встречаются:
    - bytes;
    - строки вида b'...';
    - испорченные кодировки;
    - некорректные символы.

    Идея взята из твоего исследовательского кода.
    """

    try:
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="replace")

        if isinstance(text, str) and text.startswith("b'"):
            text = ast.literal_eval(text)

        return UnicodeDammit(text).unicode_markup

    except Exception:
        return str(text)


def clean_text(text: str, lang: str | None = None) -> str:
    """
    Очищает один текстовый фрагмент.

    Что делает:
    1. Исправляет кодировку.
    2. Исправляет Unicode-ошибки через ftfy.
    3. Удаляет мусорные escape-последовательности.
    4. Удаляет служебные маркеры Wiki40B.
    5. Для японского, китайского и корейского языков оставляет только допустимые символы.
    """

    text = force_decode(text)
    text = fix_text(text)

    # Удаляем мусорные escape-последовательности
    text = re.sub(r"\\x[0-9a-fA-F]{2,}", "", text)
    text = re.sub(r"\\[a-zA-Z0-9]", "", text)
    text = re.sub(r"\ufffd", "", text)

    # Удаляем маркеры Wiki40B
    markers = [
        "_START_ARTICLE_",
        "_START_SECTION_",
        "_START_PARAGRAPH_",
        "_NEWLINE_",
    ]

    for marker in markers:
        text = text.replace(marker, "")

    # Простая языковая фильтрация для CJK-языков
    if lang == "ko":
        text = re.sub(r"[^\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\s\w]", "", text)

    elif lang == "ja":
        text = re.sub(r"[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\s\w]", "", text)

    elif lang == "zh-cn":
        text = re.sub(r"[^\u4e00-\u9fff\s\w]", "", text)

    return text.strip()


def should_skip_text(text: str) -> bool:
    """
    Проверяет, нужно ли выбросить текстовый фрагмент.

    Фрагмент отбрасывается, если:
    - это не строка;
    - он слишком короткий;
    - в нём нет букв.
    """

    if not isinstance(text, str):
        return True

    stripped = text.strip()

    return len(stripped) < 10 or not any(char.isalpha() for char in stripped)


def read_paragraphs(path: str | Path) -> list[str]:
    """
    Читает текстовый файл и разбивает его на фрагменты.

    Если в файле есть пустые строки между абзацами,
    то разделяем по двойному переносу строки.

    Если пустых строк нет,
    то считаем отдельной единицей каждую строку.
    """

    path = Path(path)

    raw_text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if "\n\n" in raw_text:
        parts = raw_text.split("\n\n")
    else:
        parts = raw_text.splitlines()

    paragraphs = []

    for part in parts:
        stripped = part.strip()
        if stripped:
            paragraphs.append(stripped)

    return paragraphs


def clean_texts(texts: Iterable[str], lang: str | None = None) -> list[str]:
    """
    Очищает список текстовых фрагментов.

    На вход:
        список строк

    На выход:
        список очищенных строк
    """

    cleaned_texts = []

    for text in texts:
        cleaned = clean_text(text, lang=lang)

        if not should_skip_text(cleaned):
            cleaned_texts.append(cleaned)

    return cleaned_texts


def train_test_split(
    items: list[str],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Делит корпус на обучающую и тестовую выборки.

    seed нужен для воспроизводимости:
    при одинаковом seed разбиение будет одинаковым.
    """

    items = list(items)

    rng = random.Random(seed)
    rng.shuffle(items)

    cut_index = int(len(items) * (1 - test_fraction))

    train_items = items[:cut_index]
    test_items = items[cut_index:]

    return train_items, test_items