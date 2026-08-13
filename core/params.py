from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ExperimentParams:
    """
    ПараметрыЭксперимента.

    Этот класс хранит все настройки одного запуска:
    - язык;
    - путь к корпусу;
    - выбранные токенизаторы;
    - размеры словаря;
    - выбранные модели;
    - список метрик;
    - параметры разбиения train/test;
    - seed для воспроизводимости.
    """

    language: str
    corpus_path: str

    tokenizer_methods: list[str] = field(default_factory=lambda: ["bpe"])
    vocab_sizes: list[int] = field(default_factory=lambda: [8000])

    model_names: list[str] = field(default_factory=lambda: ["none"])
    metrics: list[str] = field(default_factory=lambda: ["compression"])

    test_fraction: float = 0.1
    min_texts: int = 10
    seed: int = 42
    split_strategy: str = "stratified_by_language"

    # Пока языковую модель не запускаем.
    # Это поле понадобится позже, когда добавим tiny GPT-2 и perplexity.
    run_language_model: bool = False
    max_steps: int = 30
    block_size: int = 128
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4

    def validate(self, catalog: Any | None = None) -> list[str]:
        """
        Проверяет корректность параметров.

        Возвращает список ошибок.
        Если список пустой, значит параметры корректны.
        """

        errors: list[str] = []

        if not self.language or not self.language.strip():
            errors.append("Не выбран язык эксперимента.")

        if not self.corpus_path or not Path(self.corpus_path).exists():
            errors.append("Файл корпуса не найден.")

        if not self.tokenizer_methods:
            errors.append("Нужно выбрать хотя бы один метод токенизации.")

        if not self.vocab_sizes:
            errors.append("Нужно выбрать хотя бы один размер словаря.")

        for size in self.vocab_sizes:
            if int(size) <= 0:
                errors.append(f"Размер словаря должен быть положительным: {size}")

        if not 0 < float(self.test_fraction) < 0.9:
            errors.append("Доля тестовой выборки должна быть в диапазоне от 0 до 0.9.")

        allowed_split_strategies = {
            "random",
            "stratified_by_language",
        }

        if self.split_strategy not in allowed_split_strategies:
            errors.append(
                "Некорректная стратегия разбиения. "
                "Допустимые значения: random, stratified_by_language."
            )

        if int(self.min_texts) < 2:
            errors.append("Минимальное количество текстов должно быть не меньше 2.")

        if self.run_language_model:
            if "tiny_gpt2" not in self.model_names:
                errors.append("Для расчёта perplexity выберите модель tiny_gpt2.")

            if int(self.max_steps) <= 0:
                errors.append("max_steps должен быть положительным.")

            if int(self.block_size) <= 1:
                errors.append("block_size должен быть больше 1.")

            if int(self.batch_size) <= 0:
                errors.append("batch_size должен быть положительным.")

            if int(self.gradient_accumulation_steps) <= 0:
                errors.append("gradient_accumulation_steps должен быть положительным.")

            if float(self.learning_rate) <= 0:
                errors.append("learning_rate должен быть положительным.")

        if catalog is not None:
            for method in self.tokenizer_methods:
                if not catalog.check_tokenizer(method):
                    errors.append(f"Метод токенизации недоступен: {method}")

            for model in self.model_names:
                if not catalog.check_model(model):
                    errors.append(f"Модель недоступна: {model}")

        return errors

    def to_dict(self) -> dict:
        """
        Преобразует параметры в словарь.

        Это нужно для сохранения config.json.
        """

        return asdict(self)