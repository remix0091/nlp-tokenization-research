from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from services.text_cleaning import clean_texts, read_paragraphs


LANGUAGE_GROUPS = {
    "slavic": ["ru", "bg", "cs"],
    "germanic": ["de", "nl", "da"],
    "isolated": ["ja", "zh-cn", "ko"],
}


RATIOS_UNDERREPRESENTED = {
    "99.9_0.1": (99.9, 0.1),
    "99_1": (99, 1),
    "98_2": (98, 2),
    "95_5": (95, 5),
}


RATIOS_TOKENIZER_TRAINING = {
    "90_10": (90, 10),
    "75_25": (75, 25),
    "50_50": (50, 50),
}


@dataclass
class MixtureConfig:
    """
    Конфигурация генерации смешанных корпусов.
    """

    source_dir: str
    output_dir: str
    selected_languages: list[str]
    selected_ratios: dict[str, tuple[float, float]]
    sample_size: int = 1000
    seed: int = 42


def get_language_group(language: str) -> str:
    """
    Возвращает группу языка: slavic, germanic, isolated.
    """

    for group_name, languages in LANGUAGE_GROUPS.items():
        if language in languages:
            return group_name

    return "custom"


def load_language_samples(
    source_dir: str | Path,
    language: str,
    sample_size: int,
) -> list[str]:
    """
    Загружает и очищает корпус конкретного языка.

    Ожидает файл:
        data/source_corpora/{language}.txt
    """

    source_dir = Path(source_dir)
    corpus_path = source_dir / f"{language}.txt"

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Не найден исходный корпус для языка {language}: {corpus_path}"
        )

    raw_texts = read_paragraphs(corpus_path)
    cleaned = clean_texts(raw_texts, lang=language)

    if len(cleaned) < 2:
        raise RuntimeError(
            f"После очистки для языка {language} осталось слишком мало фрагментов: {len(cleaned)}"
        )

    return cleaned[:sample_size]


def create_mixed_texts(
    english_samples: list[str],
    target_samples: list[str],
    english_percent: float,
    target_percent: float,
    total_size: int,
    seed: int,
) -> list[str]:
    """
    Создаёт одну смешанную выборку.

    Например:
        95% английский + 5% целевой язык.
    """

    if total_size <= 0:
        raise ValueError("total_size должен быть положительным.")

    rng = random.Random(seed)

    target_count = int(total_size * target_percent / 100)
    english_count = total_size - target_count

    if english_count > len(english_samples):
        raise RuntimeError(
            f"Недостаточно английских фрагментов: нужно {english_count}, есть {len(english_samples)}"
        )

    if target_count > len(target_samples):
        raise RuntimeError(
            f"Недостаточно целевых фрагментов: нужно {target_count}, есть {len(target_samples)}"
        )

    mixed = (
        rng.sample(english_samples, english_count)
        + rng.sample(target_samples, target_count)
    )

    rng.shuffle(mixed)

    return mixed


def save_mixed_corpus(
    texts: list[str],
    output_path: str | Path,
) -> None:
    """
    Сохраняет смешанный корпус в txt.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for text in texts:
            file.write(text.strip() + "\n\n")


def generate_mixtures(config: MixtureConfig) -> pd.DataFrame:
    """
    Генерирует набор смешанных корпусов.

    Возвращает DataFrame со списком созданных файлов.
    """

    source_dir = Path(config.source_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    english_samples = load_language_samples(
        source_dir=source_dir,
        language="en",
        sample_size=config.sample_size,
    )

    rows = []

    for language in config.selected_languages:
        group = get_language_group(language)

        target_samples = load_language_samples(
            source_dir=source_dir,
            language=language,
            sample_size=config.sample_size,
        )

        total_size = min(
            config.sample_size,
            len(english_samples),
            len(target_samples) * 100,
        )

        for ratio_name, (english_percent, target_percent) in config.selected_ratios.items():
            try:
                mixed_texts = create_mixed_texts(
                    english_samples=english_samples,
                    target_samples=target_samples,
                    english_percent=english_percent,
                    target_percent=target_percent,
                    total_size=total_size,
                    seed=config.seed,
                )

                filename = f"mix_{group}_{language}_{ratio_name}.txt"
                output_path = output_dir / filename

                save_mixed_corpus(
                    texts=mixed_texts,
                    output_path=output_path,
                )

                rows.append(
                    {
                        "file": filename,
                        "path": str(output_path),
                        "group": group,
                        "target_lang": language,
                        "ratio": ratio_name,
                        "english_percent": english_percent,
                        "target_percent": target_percent,
                        "n_texts": len(mixed_texts),
                        "status": "created",
                        "error": "",
                    }
                )

            except Exception as error:
                rows.append(
                    {
                        "file": "",
                        "path": "",
                        "group": group,
                        "target_lang": language,
                        "ratio": ratio_name,
                        "english_percent": english_percent,
                        "target_percent": target_percent,
                        "n_texts": 0,
                        "status": "failed",
                        "error": repr(error),
                    }
                )

    return pd.DataFrame(rows)