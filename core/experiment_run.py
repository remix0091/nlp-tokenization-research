from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.params import ExperimentParams
from core.result_package import ResultPackage
from core.storage_manager import StorageManager

from services.metrics_service import compression_metrics_by_language
from services.plotting_service import make_compression_plots
from services.text_cleaning import (
    clean_texts,
    read_paragraphs,
    train_test_split,
)
from services.tokenization_service import (
    save_tokenizer,
    train_tokenizer,
)

from services.language_model_service import (
    TinyGpt2Params,
    train_and_eval_tiny_gpt2,
)
from services.language_detection_service import (
    split_texts_by_language,
    stratified_train_test_split_by_language,
)


class ExperimentRun:
    """
    ЗапускЭксперимента.

    Этот класс отвечает за полный цикл одного запуска:
    1. создать папку запуска;
    2. сохранить конфигурацию;
    3. скопировать корпус;
    4. очистить корпус;
    5. разделить на train/test;
    6. обучить токенизаторы;
    7. рассчитать метрики;
    8. сохранить таблицы, логи и отчёт;
    9. сформировать ПакетРезультатов.
    """

    def __init__(
        self,
        params: ExperimentParams,
        storage: StorageManager,
    ) -> None:
        self.params = params
        self.storage = storage

        self.run_dir: Path | None = None
        self.status: str = "created"

    def execute(self) -> ResultPackage:
        """
        Выполняет эксперимент.

        Возвращает объект ResultPackage.
        """

        self.run_dir = self.storage.create_run_dir()
        logger = self.storage.setup_logger(self.run_dir)

        self.storage.update_status(
            self.run_dir,
            status="created",
            message="Запуск создан.",
        )

        self.storage.save_json(
            self.run_dir / "config.json",
            self.params.to_dict(),
        )

        logger.info("Создан запуск эксперимента.")
        logger.info("Папка запуска: %s", self.run_dir)

        try:
            self.storage.update_status(
                self.run_dir,
                status="running",
                message="Эксперимент выполняется.",
            )

            logger.info("Старт подготовки корпуса.")

            copied_corpus_path = self.storage.copy_corpus(
                self.params.corpus_path,
                self.run_dir,
            )

            logger.info("Корпус скопирован: %s", copied_corpus_path)

            raw_texts = read_paragraphs(copied_corpus_path)
            cleaned_texts = clean_texts(raw_texts, lang=self.params.language)

            logger.info("Фрагментов до очистки: %s", len(raw_texts))
            logger.info("Фрагментов после очистки: %s", len(cleaned_texts))

            if len(cleaned_texts) < self.params.min_texts:
                raise RuntimeError(
                    "Недостаточно текстовых фрагментов после очистки: "
                    f"{len(cleaned_texts)}. Минимум: {self.params.min_texts}."
                )

            if self.params.split_strategy == "stratified_by_language":
                (
                    train_texts,
                    test_texts,
                    test_en_texts,
                    test_target_texts,
                    test_unknown_texts,
                    split_stats,
                ) = stratified_train_test_split_by_language(
                    texts=cleaned_texts,
                    target_lang=self.params.language,
                    test_fraction=self.params.test_fraction,
                    seed=self.params.seed,
                )

                logger.info(
                    "Используется стратифицированное разбиение по языкам."
                )

                logger.info("Статистика разбиения: %s", split_stats)

            else:
                train_texts, test_texts = train_test_split(
                    cleaned_texts,
                    test_fraction=self.params.test_fraction,
                    seed=self.params.seed,
                )

                test_en_texts, test_target_texts, test_unknown_texts = split_texts_by_language(
                    texts=test_texts,
                    target_lang=self.params.language,
                )

                split_stats = {
                    "total_texts": len(cleaned_texts),
                    "train_texts": len(train_texts),
                    "test_texts": len(test_texts),
                    "test_en_texts": len(test_en_texts),
                    "test_target_texts": len(test_target_texts),
                    "test_unknown_texts": len(test_unknown_texts),
                }

                logger.info("Используется случайное разбиение.")

            logger.info("Train фрагментов: %s", len(train_texts))
            logger.info("Test фрагментов: %s", len(test_texts))

            logger.info("Test EN фрагментов: %s", len(test_en_texts))
            logger.info("Test target фрагментов: %s", len(test_target_texts))
            logger.info("Test unknown фрагментов: %s", len(test_unknown_texts))

            rows = []
            errors = []

            for method in self.params.tokenizer_methods:
                for vocab_size in self.params.vocab_sizes:
                    stage_name = f"{method}/vocab_{vocab_size}"

                    logger.info("Начата комбинация: %s", stage_name)

                    try:
                        tokenizer = train_tokenizer(
                            method=method,
                            train_texts=train_texts,
                            vocab_size=vocab_size,
                        )

                        tokenizer_save_dir = (
                            self.run_dir
                            / "tokenizers"
                            / method
                            / f"vocab_{vocab_size}"
                        )

                        tokenizer_path = save_tokenizer(
                            tokenizer=tokenizer,
                            save_dir=tokenizer_save_dir,
                        )

                        logger.info("Токенизатор сохранён: %s", tokenizer_path)

                        metrics = compression_metrics_by_language(
                            tokenizer=tokenizer,
                            all_texts=test_texts,
                            en_texts=test_en_texts,
                            target_texts=test_target_texts,
                        )

                        logger.info("Метрики рассчитаны для %s", stage_name)

                        row = {
                            "method": method,
                            "vocab_size": vocab_size,
                            "tokenizer_path": tokenizer_path,
                            **metrics,
                        }

                        if (
                            self.params.run_language_model
                            and "tiny_gpt2" in self.params.model_names
                        ):
                            logger.info(
                                "Запущено обучение tiny_gpt2 для %s",
                                stage_name,
                            )

                            lm_output_dir = (
                                self.run_dir
                                / "lm"
                                / method
                                / f"vocab_{vocab_size}"
                            )

                            lm_params = TinyGpt2Params(
                                max_steps=self.params.max_steps,
                                block_size=self.params.block_size,
                                batch_size=self.params.batch_size,
                                gradient_accumulation_steps=(
                                    self.params.gradient_accumulation_steps
                                ),
                                learning_rate=self.params.learning_rate,
                                seed=self.params.seed,
                            )

                            lm_metrics = train_and_eval_tiny_gpt2(
                                tokenizer_json_path=tokenizer_path,
                                train_texts=train_texts,
                                test_texts=test_texts,
                                output_dir=str(lm_output_dir),
                                params=lm_params,
                                test_en_texts=test_en_texts,
                                test_target_texts=test_target_texts,
                            )

                            row.update(lm_metrics)

                            logger.info(
                                "Perplexity для %s: %s",
                                stage_name,
                                lm_metrics.get("ppl"),
                            )

                        rows.append(row)

                    except Exception as error:
                        error_info = {
                            "stage": stage_name,
                            "error": repr(error),
                        }

                        errors.append(error_info)

                        logger.exception(
                            "Ошибка при выполнении комбинации %s",
                            stage_name,
                        )

            result_columns = [
                "method",
                "vocab_size",
                "tokenizer_path",

                "tokens_per_char",
                "tokens_per_word",
                "avg_token_len",
                "n_texts",
                "n_tokens",

                "tokens_per_char_en",
                "tokens_per_word_en",
                "avg_token_len_en",
                "n_texts_en",
                "n_tokens_en",

                "tokens_per_char_target",
                "tokens_per_word_target",
                "avg_token_len_target",
                "n_texts_target",
                "n_tokens_target",

                "compression_gap_tokens_per_char",
                "compression_gap_tokens_per_word",

                "compression_factor_char",
                "compression_factor_char_en",
                "compression_factor_char_target",
                "compression_penalty_char",
            ]

            if self.params.run_language_model:
                result_columns.extend(
                    [
                        "ppl",
                        "eval_loss",
                        "train_loss",
                        "ppl_en",
                        "loss_en",
                        "test_en_blocks",
                        "ppl_target",
                        "loss_target",
                        "test_target_blocks",
                        "gap_loss",
                        "train_blocks",
                        "test_blocks",
                    ]
                )

            results_df = pd.DataFrame(rows, columns=result_columns)

            results_csv_path = self.run_dir / "tables" / "experiment_results.csv"

            self.storage.save_dataframe(
                results_csv_path,
                results_df,
            )

            logger.info("Таблица результатов сохранена: %s", results_csv_path)

            plots = make_compression_plots(
                df=results_df,
                output_dir=self.run_dir / "figures",
            )

            logger.info("Построено графиков: %s", len(plots))

            if errors:
                errors_csv_path = self.run_dir / "tables" / "errors.csv"

                self.storage.save_dataframe(
                    errors_csv_path,
                    pd.DataFrame(errors),
                )

                logger.info("Таблица ошибок сохранена: %s", errors_csv_path)

            metrics_summary = self._build_metrics_summary(
                raw_texts=raw_texts,
                cleaned_texts=cleaned_texts,
                train_texts=train_texts,
                test_texts=test_texts,
                test_en_texts=test_en_texts,
                test_target_texts=test_target_texts,
                test_unknown_texts=test_unknown_texts,
                split_stats=split_stats,
                rows=rows,
                errors=errors,
                results_df=results_df,
            )

            self.storage.save_json(
                self.run_dir / "metrics.json",
                metrics_summary,
            )

            logger.info("Сводные метрики сохранены.")

            report_text = self._build_report_text(
                metrics_summary=metrics_summary,
                rows_count=len(rows),
                errors_count=len(errors),
            )

            report_path = self.run_dir / "report.md"

            self.storage.save_text(
                report_path,
                report_text,
            )

            logger.info("Отчёт сохранён: %s", report_path)

            if rows and not errors:
                final_status = "completed"
            elif rows and errors:
                final_status = "partially_completed"
            else:
                final_status = "failed"

            self.storage.update_status(
                self.run_dir,
                status=final_status,
                message="Эксперимент завершён.",
            )

            self.status = final_status

            logger.info("Эксперимент завершён со статусом: %s", final_status)

            self._flush_logger(logger)

            logs_path = self.run_dir / "logs.txt"

            logs_text = logs_path.read_text(
                encoding="utf-8",
            )

            result_package = ResultPackage(
                metrics=metrics_summary,
                tables={
                    "experiment_results": str(results_csv_path),
                },
                plots=plots,
                report=report_text,
                logs=logs_text,
                save_path=str(self.run_dir),
                errors=errors,
            )

            self.storage.save_json(
                self.run_dir / "result_package.json",
                result_package.to_dict(),
            )

            logger.info("Пакет результатов сохранён.")
            self._flush_logger(logger)

            return result_package

        except Exception as error:
            self.status = "failed"

            self.storage.update_status(
                self.run_dir,
                status="failed",
                message=repr(error),
            )

            logger.exception("Эксперимент завершён с ошибкой.")
            self._flush_logger(logger)

            raise error

    def _build_metrics_summary(
        self,
        raw_texts: list[str],
        cleaned_texts: list[str],
        train_texts: list[str],
        test_texts: list[str],
        test_en_texts: list[str],
        test_target_texts: list[str],
        test_unknown_texts: list[str],
        split_stats: dict,
        rows: list[dict],
        errors: list[dict],
        results_df: pd.DataFrame,
    ) -> dict:
        """
        Формирует краткую сводку метрик.
        """

        metrics_summary = {
            "raw_texts": len(raw_texts),
            "cleaned_texts": len(cleaned_texts),
            "train_texts": len(train_texts),
            "test_texts": len(test_texts),
            "completed_combinations": len(rows),
            "failed_combinations": len(errors),
            "best_tokens_per_char": None,
            "best_ppl": None,
            "test_en_texts": len(test_en_texts),
            "test_target_texts": len(test_target_texts),
            "test_unknown_texts": len(test_unknown_texts),
            "best_gap_loss": None,
            "split_strategy": self.params.split_strategy,
            "split_stats": split_stats,
            "best_target_tokens_per_char": None,
            "best_compression_gap_tokens_per_char": None,
            "best_target_compression_factor": None,
            "best_compression_penalty_char": None,
        }

        if not results_df.empty:
            best_row = (
                results_df
                .sort_values("tokens_per_char")
                .head(1)
                .to_dict(orient="records")[0]
            )

            metrics_summary["best_tokens_per_char"] = best_row

        if "compression_factor_char_target" in results_df.columns:
            factor_df = results_df.dropna(
                subset=["compression_factor_char_target"]
            )

            if not factor_df.empty:
                best_factor_row = (
                    factor_df
                    .sort_values("compression_factor_char_target", ascending=False)
                    .head(1)
                    .to_dict(orient="records")[0]
                )

                metrics_summary["best_target_compression_factor"] = best_factor_row

        if "compression_penalty_char" in results_df.columns:
            penalty_df = results_df.dropna(
                subset=["compression_penalty_char"]
            ).copy()

            if not penalty_df.empty:
                penalty_df["compression_penalty_abs"] = (
                    penalty_df["compression_penalty_char"].abs()
                )

                best_penalty_row = (
                    penalty_df
                    .sort_values("compression_penalty_abs")
                    .head(1)
                    .to_dict(orient="records")[0]
                )

                metrics_summary["best_compression_penalty_char"] = best_penalty_row

        if "tokens_per_char_target" in results_df.columns:
            target_df = results_df.dropna(subset=["tokens_per_char_target"])

            if not target_df.empty:
                best_target_row = (
                    target_df
                    .sort_values("tokens_per_char_target")
                    .head(1)
                    .to_dict(orient="records")[0]
                )

                metrics_summary["best_target_tokens_per_char"] = best_target_row

        if "compression_gap_tokens_per_char" in results_df.columns:
            compression_gap_df = results_df.dropna(
                subset=["compression_gap_tokens_per_char"]
            ).copy()

            if not compression_gap_df.empty:
                compression_gap_df["compression_gap_abs"] = (
                    compression_gap_df["compression_gap_tokens_per_char"].abs()
                )

                best_compression_gap_row = (
                    compression_gap_df
                    .sort_values("compression_gap_abs")
                    .head(1)
                    .to_dict(orient="records")[0]
                )

                metrics_summary["best_compression_gap_tokens_per_char"] = (
                    best_compression_gap_row
                )
        
        if "ppl" in results_df.columns:
            ppl_df = results_df.dropna(subset=["ppl"])

            if not ppl_df.empty:
                best_ppl_row = (
                    ppl_df
                    .sort_values("ppl")
                    .head(1)
                    .to_dict(orient="records")[0]
                )

                metrics_summary["best_ppl"] = best_ppl_row

        if "gap_loss" in results_df.columns:
            gap_df = results_df.dropna(subset=["gap_loss"])

            if not gap_df.empty:
                best_gap_row = (
                    gap_df
                    .sort_values("gap_loss")
                    .head(1)
                    .to_dict(orient="records")[0]
                )

                metrics_summary["best_gap_loss"] = best_gap_row

        return metrics_summary

    def _build_report_text(
        self,
        metrics_summary: dict,
        rows_count: int,
        errors_count: int,
    ) -> str:
        """
        Формирует текстовый отчёт.
        """

        best_result = metrics_summary.get("best_tokens_per_char")

        if best_result is None:
            best_result_block = "Нет успешных комбинаций."
        else:
            best_result_block = "\n".join(
                [
                    f"- Метод: `{best_result['method']}`",
                    f"- Размер словаря: `{best_result['vocab_size']}`",
                    f"- Tokens per char: `{best_result['tokens_per_char']:.4f}`",
                    f"- Tokens per word: `{best_result['tokens_per_word']:.4f}`",
                    f"- Средняя длина токена: `{best_result['avg_token_len']:.4f}`",
                    f"- Количество текстов: `{best_result['n_texts']}`",
                    f"- Количество токенов: `{best_result['n_tokens']}`",
                    f"- Путь к токенизатору: `{best_result['tokenizer_path']}`",
                ]
            )

        best_target_compression = metrics_summary.get("best_target_tokens_per_char")

        if best_target_compression is None:
            best_target_compression_block = "Метрики целевого языка не рассчитаны."
        else:
            best_target_compression_block = "\n".join(
                [
                    f"- Метод: `{best_target_compression['method']}`",
                    f"- Размер словаря: `{best_target_compression['vocab_size']}`",
                    f"- Target tokens per char: `{best_target_compression['tokens_per_char_target']:.4f}`",
                    f"- Target tokens per word: `{best_target_compression['tokens_per_word_target']:.4f}`",
                    f"- Target avg token len: `{best_target_compression['avg_token_len_target']:.4f}`",
                ]
            )

        best_compression_gap = metrics_summary.get("best_compression_gap_tokens_per_char")

        if best_compression_gap is None:
            best_compression_gap_block = "Разрыв сжатия между target и EN не рассчитан."
        else:
            best_compression_gap_block = "\n".join(
                [
                    f"- Метод: `{best_compression_gap['method']}`",
                    f"- Размер словаря: `{best_compression_gap['vocab_size']}`",
                    f"- Compression gap tokens/char: `{best_compression_gap['compression_gap_tokens_per_char']:.4f}`",
                    f"- EN tokens/char: `{best_compression_gap['tokens_per_char_en']:.4f}`",
                    f"- Target tokens/char: `{best_compression_gap['tokens_per_char_target']:.4f}`",
                ]
            )

        best_ppl = metrics_summary.get("best_ppl")

        if best_ppl is None:
            best_ppl_block = "Perplexity не рассчитывалась."
        else:
            best_ppl_block = "\n".join(
                [
                    f"- Метод: `{best_ppl['method']}`",
                    f"- Размер словаря: `{best_ppl['vocab_size']}`",
                    f"- PPL: `{best_ppl['ppl']:.4f}`",
                    f"- Eval loss: `{best_ppl['eval_loss']:.4f}`",
                    f"- Train loss: `{best_ppl['train_loss']:.4f}`",
                    f"- Train blocks: `{best_ppl['train_blocks']}`",
                    f"- Test blocks: `{best_ppl['test_blocks']}`",
                ]
            )

        best_gap_loss = metrics_summary.get("best_gap_loss")

        if best_gap_loss is None:
            best_gap_loss_block = "Gap loss не рассчитывался."
        else:
            best_gap_loss_block = "\n".join(
                [
                    f"- Метод: `{best_gap_loss['method']}`",
                    f"- Размер словаря: `{best_gap_loss['vocab_size']}`",
                    f"- Gap loss: `{best_gap_loss['gap_loss']:.4f}`",
                    f"- Loss target: `{best_gap_loss['loss_target']:.4f}`",
                    f"- Loss EN: `{best_gap_loss['loss_en']:.4f}`",
                    f"- PPL target: `{best_gap_loss['ppl_target']:.4f}`",
                    f"- PPL EN: `{best_gap_loss['ppl_en']:.4f}`",
                ]
            )

        lines = [
            "# Отчёт по эксперименту",
            "",
            "## Конфигурация",
            "",
            f"- Язык: `{self.params.language}`",
            f"- Корпус: `{self.params.corpus_path}`",
            f"- Методы токенизации: `{', '.join(self.params.tokenizer_methods)}`",
            f"- Размеры словаря: `{', '.join(map(str, self.params.vocab_sizes))}`",
            f"- Доля тестовой выборки: `{self.params.test_fraction}`",
            f"- Seed: `{self.params.seed}`",
            "",
            "## Подготовка корпуса",
            "",
            f"- Фрагментов до очистки: `{metrics_summary['raw_texts']}`",
            f"- Фрагментов после очистки: `{metrics_summary['cleaned_texts']}`",
            f"- Train: `{metrics_summary['train_texts']}`",
            f"- Test: `{metrics_summary['test_texts']}`",
            f"- Стратегия разбиения: `{metrics_summary.get('split_strategy')}`",
            f"- Test EN: `{metrics_summary.get('test_en_texts')}`",
            f"- Test target: `{metrics_summary.get('test_target_texts')}`",
            f"- Test unknown: `{metrics_summary.get('test_unknown_texts')}`",
            "",
            "## Выполнение",
            "",
            f"- Успешных комбинаций: `{rows_count}`",
            f"- Ошибочных комбинаций: `{errors_count}`",
            "",
            "## Лучший результат по tokens_per_char",
            "",
            best_result_block,
            "",
            "## Лучшее сжатие целевого языка",
            "",
            best_target_compression_block,
            "",
            "## Минимальный разрыв сжатия target и EN",
            "",
            best_compression_gap_block,
            "## Лучший результат по perplexity",
            "",
            best_ppl_block,
            "",
            "## Лучший результат по gap_loss",
            "",
            best_gap_loss_block,
            "",
            "## Визуализации",
            "",
            "Графики по метрикам сохраняются в папке `figures` внутри папки запуска.",
        ]

        return "\n".join(lines)
    
    def _flush_logger(self, logger) -> None:
        """
        Принудительно записывает все накопленные логи в файл.
        """

        for handler in logger.handlers:
            handler.flush()