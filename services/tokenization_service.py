from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from tokenizers import SentencePieceUnigramTokenizer, Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer


SPECIAL_TOKENS = ["<unk>", "<pad>", "<s>", "</s>", "<mask>"]


def _write_temp_training_file(train_texts: list[str]) -> str:
    """
    Создаёт временный txt-файл для обучения токенизатора.

    Библиотека tokenizers обучает токенизаторы из файлов,
    поэтому список строк нужно сначала записать во временный файл.
    """

    temp_file = NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        suffix=".txt",
    )

    with temp_file as file:
        for text in train_texts:
            text = text.replace("\n", " ")
            file.write(text + "\n")

    return temp_file.name


def train_tokenizer(
    method: str,
    train_texts: list[str],
    vocab_size: int,
) -> Any:
    """
    Обучает токенизатор выбранного типа.

    method:
        bpe
        wordpiece
        unigram

    train_texts:
        обучающие текстовые фрагменты

    vocab_size:
        размер словаря токенизатора
    """

    if not train_texts:
        raise ValueError("Нельзя обучить токенизатор: обучающая выборка пуста.")

    temp_path = _write_temp_training_file(train_texts)

    if method == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
        )

        tokenizer.train([temp_path], trainer=trainer)
        return tokenizer

    if method == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordPieceTrainer(
            vocab_size=int(vocab_size),
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
        )

        tokenizer.train([temp_path], trainer=trainer)
        return tokenizer

    if method == "unigram":
        tokenizer = SentencePieceUnigramTokenizer()

        tokenizer.train(
            files=[temp_path],
            vocab_size=int(vocab_size),
            special_tokens=SPECIAL_TOKENS,
            unk_token="<unk>",
        )

        return tokenizer

    raise ValueError(f"Неизвестный метод токенизации: {method}")


def save_tokenizer(
    tokenizer: Any,
    save_dir: str | Path,
) -> str:
    """
    Сохраняет обученный токенизатор в tokenizer.json.

    Возвращает путь к сохранённому файлу.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    return str(tokenizer_path)