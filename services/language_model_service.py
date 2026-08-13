from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class TinyGpt2Params:
    """
    Параметры tiny GPT-2.
    """

    max_steps: int = 30
    block_size: int = 128
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    seed: int = 42

    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    max_pos: int = 256


class CausalTextDataset(Dataset):
    """
    Dataset для causal language modeling.
    """

    def __init__(
        self,
        tokenizer,
        texts: list[str],
        block_size: int = 128,
    ) -> None:
        self.examples = []

        for text in texts:
            text = (text or "").strip().replace("\n", " ")

            if not text:
                continue

            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            if len(ids) < 2:
                continue

            for index in range(0, len(ids) - 1, block_size):
                chunk = ids[index:index + block_size]

                if len(chunk) >= 2:
                    self.examples.append(
                        torch.tensor(chunk, dtype=torch.long)
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.examples[index],
        }


def _evaluate_dataset(
    trainer,
    dataset,
) -> dict[str, float]:
    """
    Считает loss и perplexity для конкретного набора данных.
    """

    if dataset is None or len(dataset) == 0:
        return {
            "loss": math.nan,
            "ppl": math.nan,
            "blocks": 0,
        }

    eval_metrics = trainer.evaluate(
        eval_dataset=dataset,
    )

    loss = float(
        eval_metrics.get("eval_loss", math.nan)
    )

    if math.isfinite(loss):
        ppl = float(math.exp(loss))
    else:
        ppl = math.nan

    return {
        "loss": loss,
        "ppl": ppl,
        "blocks": len(dataset),
    }


def train_and_eval_tiny_gpt2(
    tokenizer_json_path: str,
    train_texts: list[str],
    test_texts: list[str],
    output_dir: str,
    params: TinyGpt2Params,
    test_en_texts: list[str] | None = None,
    test_target_texts: list[str] | None = None,
) -> dict[str, Any]:
    """
    Обучает маленькую GPT-2-подобную модель и считает:
    - общий PPL;
    - общий eval_loss;
    - PPL/loss для английского;
    - PPL/loss для целевого языка;
    - gap_loss = loss_target - loss_en.
    """

    from transformers import (
        DataCollatorForLanguageModeling,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(params.seed)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,
    )

    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token = "<pad>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.mask_token = "<mask>"

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=params.max_pos,
        n_ctx=params.max_pos,
        n_embd=params.n_embd,
        n_layer=params.n_layer,
        n_head=params.n_head,
        bos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.bos_token),
        eos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
    )

    model = GPT2LMHeadModel(config)

    train_dataset = CausalTextDataset(
        tokenizer=tokenizer,
        texts=train_texts,
        block_size=params.block_size,
    )

    test_all_dataset = CausalTextDataset(
        tokenizer=tokenizer,
        texts=test_texts,
        block_size=params.block_size,
    )

    test_en_dataset = None
    test_target_dataset = None

    if test_en_texts:
        test_en_dataset = CausalTextDataset(
            tokenizer=tokenizer,
            texts=test_en_texts,
            block_size=params.block_size,
        )

    if test_target_texts:
        test_target_dataset = CausalTextDataset(
            tokenizer=tokenizer,
            texts=test_target_texts,
            block_size=params.block_size,
        )

    if len(train_dataset) == 0:
        raise RuntimeError("После токенизации обучающая выборка для LM пуста.")

    if len(test_all_dataset) == 0:
        raise RuntimeError("После токенизации тестовая выборка для LM пуста.")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args_signature = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    args_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": params.batch_size,
        "gradient_accumulation_steps": params.gradient_accumulation_steps,
        "learning_rate": params.learning_rate,
        "max_steps": params.max_steps,
        "logging_steps": max(1, min(10, params.max_steps)),
    }

    if "overwrite_output_dir" in training_args_signature:
        args_kwargs["overwrite_output_dir"] = True

    if "report_to" in training_args_signature:
        args_kwargs["report_to"] = "none"

    if "save_strategy" in training_args_signature:
        args_kwargs["save_strategy"] = "no"
    elif "save_steps" in training_args_signature:
        args_kwargs["save_steps"] = 10**9

    if "evaluation_strategy" in training_args_signature:
        args_kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in training_args_signature:
        args_kwargs["eval_strategy"] = "no"

    if "fp16" in training_args_signature:
        args_kwargs["fp16"] = torch.cuda.is_available()

    training_args = TrainingArguments(**args_kwargs)

    trainer_signature = inspect.signature(
        Trainer.__init__
    ).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
    }

    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_output = trainer.train()

    all_result = _evaluate_dataset(
        trainer=trainer,
        dataset=test_all_dataset,
    )

    en_result = _evaluate_dataset(
        trainer=trainer,
        dataset=test_en_dataset,
    )

    target_result = _evaluate_dataset(
        trainer=trainer,
        dataset=test_target_dataset,
    )

    loss_en = en_result["loss"]
    loss_target = target_result["loss"]

    if math.isfinite(loss_en) and math.isfinite(loss_target):
        gap_loss = loss_target - loss_en
    else:
        gap_loss = math.nan

    return {
        "ppl": all_result["ppl"],
        "eval_loss": all_result["loss"],
        "train_loss": float(getattr(train_output, "training_loss", math.nan)),

        "ppl_en": en_result["ppl"],
        "loss_en": en_result["loss"],
        "test_en_blocks": en_result["blocks"],

        "ppl_target": target_result["ppl"],
        "loss_target": target_result["loss"],
        "test_target_blocks": target_result["blocks"],

        "gap_loss": gap_loss,

        "train_blocks": len(train_dataset),
        "test_blocks": all_result["blocks"],
    }