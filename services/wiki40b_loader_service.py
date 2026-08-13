from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from services.text_cleaning import clean_text, should_skip_text


WIKI40B_DATASET_NAME = "google/wiki40b"


SUPPORTED_WIKI40B_LANGUAGES = [
    "en",
    "ru",
    "bg",
    "cs",
    "de",
    "nl",
    "da",
    "ja",
    "zh-cn",
    "ko",
]


@dataclass
class Wiki40BLoadConfig:
    """
    Конфигурация загрузки одного языка из Wiki40B.
    """

    language: str
    output_dir: str
    split: str = "train"
    max_samples: int = 1000
    streaming: bool = True
    seed: int = 42


def extract_text_from_example(example: dict[str, Any]) -> str:
    """
    Достаёт текст из объекта Wiki40B.

    В разных сборках поле может называться по-разному.
    Самые вероятные варианты:
    - text;
    - content;
    - article.
    """

    for key in ["text", "content", "article"]:
        value = example.get(key)

        if isinstance(value, str) and value.strip():
            return value

    return ""


def load_wiki40b_language(config: Wiki40BLoadConfig) -> dict[str, Any]:
    """
    Загружает один язык из Wiki40B и сохраняет его как локальный txt-файл.

    Возвращает словарь со статусом загрузки.
    """

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{config.language}.txt"

    started_at = datetime.now().isoformat(timespec="seconds")

    row = {
        **asdict(config),
        "dataset": WIKI40B_DATASET_NAME,
        "output_path": str(output_path),
        "loaded_samples": 0,
        "status": "started",
        "error": "",
        "started_at": started_at,
        "finished_at": "",
    }

    try:
        dataset = load_dataset(
            WIKI40B_DATASET_NAME,
            config.language,
            split=config.split,
            streaming=config.streaming,
            trust_remote_code=True,
        )

        texts = []

        for example in itertools.islice(dataset, config.max_samples * 5):
            raw_text = extract_text_from_example(example)
            cleaned = clean_text(raw_text, lang=config.language)

            if should_skip_text(cleaned):
                continue

            texts.append(cleaned)

            if len(texts) >= config.max_samples:
                break

        if not texts:
            raise RuntimeError(
                f"Не удалось получить ни одного текстового фрагмента для языка {config.language}."
            )

        with output_path.open("w", encoding="utf-8") as file:
            for text in texts:
                file.write(text.strip() + "\n\n")

        row["loaded_samples"] = len(texts)
        row["status"] = "completed"

    except Exception as error:
        row["status"] = "failed"
        row["error"] = repr(error)

    row["finished_at"] = datetime.now().isoformat(timespec="seconds")

    return row


def load_wiki40b_languages(
    languages: list[str],
    output_dir: str,
    split: str = "train",
    max_samples: int = 1000,
    streaming: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Загружает несколько языков Wiki40B.

    Возвращает DataFrame-манифест.
    """

    rows = []

    for language in languages:
        config = Wiki40BLoadConfig(
            language=language,
            output_dir=output_dir,
            split=split,
            max_samples=max_samples,
            streaming=streaming,
            seed=seed,
        )

        row = load_wiki40b_language(config)
        rows.append(row)

    manifest = pd.DataFrame(rows)

    output_path = Path(output_dir) / "wiki40b_load_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return manifest