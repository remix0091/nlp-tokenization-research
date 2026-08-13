from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_metric_line_plot(
    df: pd.DataFrame,
    metric_name: str,
    y_label: str,
    title: str,
    output_path: str | Path,
) -> str | None:
    """
    Строит линейный график метрики по размеру словаря.

    По оси X:
        vocab_size

    По оси Y:
        выбранная метрика

    Отдельная линия:
        каждый метод токенизации
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return None

    required_columns = {"method", "vocab_size", metric_name}

    if not required_columns.issubset(df.columns):
        return None

    plot_df = df.dropna(subset=[metric_name]).copy()

    if plot_df.empty:
        return None

    aggregated = (
        plot_df
        .groupby(["method", "vocab_size"], as_index=False)[metric_name]
        .mean()
        .sort_values(["method", "vocab_size"])
    )

    plt.figure(figsize=(6, 3.5))

    for method in sorted(aggregated["method"].unique()):
        method_df = aggregated[aggregated["method"] == method].sort_values("vocab_size")

        plt.plot(
            method_df["vocab_size"],
            method_df[metric_name],
            marker="o",
            label=method,
        )

    plt.xlabel("Размер словаря")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    return str(output_path)


def make_compression_plots(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[str]:
    """
    Создаёт набор графиков по результатам эксперимента.

    Возвращает список путей к созданным PNG-файлам.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: list[str] = []

    plot_specs = [
        {
            "metric_name": "compression_factor_char_target",
            "y_label": "Target chars per token",
            "title": "Множитель сжатия целевого языка",
            "filename": "compression_factor_char_target_by_vocab.png",
        },
        {
            "metric_name": "compression_penalty_char",
            "y_label": "EN factor - Target factor",
            "title": "Штраф сжатия целевого языка относительно EN",
            "filename": "compression_penalty_char_by_vocab.png",
        },
        {
            "metric_name": "ppl_target",
            "y_label": "Target PPL",
            "title": "Perplexity целевого языка",
            "filename": "ppl_target_by_vocab.png",
        },
        {
            "metric_name": "gap_loss",
            "y_label": "Gap loss",
            "title": "Разница ошибки модели между target и EN",
            "filename": "gap_loss_by_vocab.png",
        },
    ]

    for spec in plot_specs:
        plot_path = make_metric_line_plot(
            df=df,
            metric_name=spec["metric_name"],
            y_label=spec["y_label"],
            title=spec["title"],
            output_path=output_dir / spec["filename"],
        )

        if plot_path is not None:
            plots.append(plot_path)

    return plots