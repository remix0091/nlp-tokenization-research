from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError


def read_csv_safely(path: str | Path) -> pd.DataFrame | None:
    """
    Безопасно читает CSV.

    Возвращает None, если файл отсутствует, пустой или повреждён.
    """

    path = Path(path)

    if not path.exists():
        return None

    if path.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(path)

    except EmptyDataError:
        return None


def ratio_to_target_percent(ratio: str) -> float | None:
    """
    Преобразует ratio вида:
        90_10
        75_25
        50_50
        99.9_0.1
        99_1
        98_2
        95_5

    в долю целевого языка:
        10
        25
        50
        0.1
        1
        2
        5
    """

    if not isinstance(ratio, str) or "_" not in ratio:
        return None

    try:
        parts = ratio.split("_")
        target_part = parts[-1]
        return float(target_part)

    except ValueError:
        return None


def collect_batch_all_results(
    batch_summary_path: str | Path,
) -> pd.DataFrame:
    """
    Собирает полную таблицу результатов пакетного запуска.

    На вход:
        results/batch_.../batch_summary.csv

    На выход:
        DataFrame, где каждая строка — это конкретная комбинация:
        corpus + tokenizer + vocab_size.
    """

    batch_summary_path = Path(batch_summary_path)
    batch_dir = batch_summary_path.parent

    summary_df = read_csv_safely(batch_summary_path)

    if summary_df is None:
        raise RuntimeError(f"Не удалось прочитать batch_summary.csv: {batch_summary_path}")

    all_rows = []

    for _, summary_row in summary_df.iterrows():
        run_path = summary_row.get("run_path", "")

        if not isinstance(run_path, str) or not run_path:
            continue

        run_dir = Path(run_path)
        results_csv_path = run_dir / "tables" / "experiment_results.csv"

        result_df = read_csv_safely(results_csv_path)

        if result_df is None or result_df.empty:
            continue

        for _, result_row in result_df.iterrows():
            row = {
                "batch_dir": str(batch_dir),
                "run_path": str(run_dir),
                "corpus_path": summary_row.get("corpus_path", ""),
                "group": summary_row.get("group", ""),
                "target_lang": summary_row.get("target_lang", ""),
                "ratio": summary_row.get("ratio", ""),
                "target_percent": ratio_to_target_percent(summary_row.get("ratio", "")),
                "batch_status": summary_row.get("status", ""),
            }

            for column in result_df.columns:
                row[column] = result_row.get(column)

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def save_batch_all_results(
    batch_summary_path: str | Path,
) -> Path:
    """
    Собирает batch_all_results.csv и сохраняет его рядом с batch_summary.csv.
    """

    batch_summary_path = Path(batch_summary_path)
    batch_dir = batch_summary_path.parent

    all_results_df = collect_batch_all_results(batch_summary_path)

    output_path = batch_dir / "batch_all_results.csv"

    all_results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


def make_metric_by_target_percent_plot(
    df: pd.DataFrame,
    metric_name: str,
    y_label: str,
    title: str,
    output_path: str | Path,
) -> str | None:
    """
    Строит график:
        X: target_percent
        Y: metric_name
        линии: method

    Значения усредняются по языкам и корпусам.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"target_percent", "method", metric_name}

    if df.empty or not required_columns.issubset(df.columns):
        return None

    plot_df = df.dropna(subset=["target_percent", metric_name]).copy()

    if plot_df.empty:
        return None

    aggregated = (
        plot_df
        .groupby(["method", "target_percent"], as_index=False)[metric_name]
        .mean()
        .sort_values(["method", "target_percent"])
    )

    if aggregated.empty:
        return None

    plt.figure(figsize=(6, 3.5))

    for method in sorted(aggregated["method"].unique()):
        method_df = aggregated[aggregated["method"] == method].sort_values("target_percent")

        plt.plot(
            method_df["target_percent"],
            method_df[metric_name],
            marker="o",
            label=method,
        )

    plt.xlabel("Доля целевого языка, %")
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


def make_metric_by_language_plot(
    df: pd.DataFrame,
    metric_name: str,
    y_label: str,
    title: str,
    output_path: str | Path,
) -> str | None:
    """
    Строит график:
        X: target_percent
        Y: metric_name
        линии: target_lang

    Значения усредняются по методам и размерам словаря.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"target_percent", "target_lang", metric_name}

    if df.empty or not required_columns.issubset(df.columns):
        return None

    plot_df = df.dropna(subset=["target_percent", metric_name]).copy()

    if plot_df.empty:
        return None

    aggregated = (
        plot_df
        .groupby(["target_lang", "target_percent"], as_index=False)[metric_name]
        .mean()
        .sort_values(["target_lang", "target_percent"])
    )

    if aggregated.empty:
        return None

    plt.figure(figsize=(6, 3.5))

    for language in sorted(aggregated["target_lang"].dropna().unique()):
        language_df = aggregated[aggregated["target_lang"] == language].sort_values("target_percent")

        plt.plot(
            language_df["target_percent"],
            language_df[metric_name],
            marker="o",
            label=language,
        )

    plt.xlabel("Доля целевого языка, %")
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


def generate_batch_analysis_plots(
    batch_all_results_path: str | Path,
) -> list[str]:
    """
    Строит набор агрегированных графиков для batch_all_results.csv.
    """

    batch_all_results_path = Path(batch_all_results_path)
    batch_dir = batch_all_results_path.parent
    figures_dir = batch_dir / "analysis_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_safely(batch_all_results_path)

    if df is None or df.empty:
        return []

    plots: list[str] = []

    metric_specs = [
        {
            "metric_name": "compression_factor_char_target",
            "y_label": "Target chars per token",
            "title_by_method": "Множитель сжатия target от доли языка по методам",
            "title_by_language": "Множитель сжатия target от доли языка по языкам",
            "filename_by_method": "compression_factor_target_by_method.png",
            "filename_by_language": "compression_factor_target_by_language.png",
        },
        {
            "metric_name": "compression_penalty_char",
            "y_label": "EN factor - Target factor",
            "title_by_method": "Штраф сжатия target относительно EN по методам",
            "title_by_language": "Штраф сжатия target относительно EN по языкам",
            "filename_by_method": "compression_penalty_by_method.png",
            "filename_by_language": "compression_penalty_by_language.png",
        },
        {
            "metric_name": "ppl_target",
            "y_label": "Target PPL",
            "title_by_method": "PPL target от доли языка по методам",
            "title_by_language": "PPL target от доли языка по языкам",
            "filename_by_method": "ppl_target_by_method.png",
            "filename_by_language": "ppl_target_by_language.png",
        },
        {
            "metric_name": "gap_loss",
            "y_label": "Gap loss",
            "title_by_method": "gap_loss от доли языка по методам",
            "title_by_language": "gap_loss от доли языка по языкам",
            "filename_by_method": "gap_loss_by_method.png",
            "filename_by_language": "gap_loss_by_language.png",
        },
    ]

    for spec in metric_specs:
        metric_name = spec["metric_name"]

        if metric_name not in df.columns:
            continue

        by_method = make_metric_by_target_percent_plot(
            df=df,
            metric_name=metric_name,
            y_label=spec["y_label"],
            title=spec["title_by_method"],
            output_path=figures_dir / spec["filename_by_method"],
        )

        if by_method is not None:
            plots.append(by_method)

        by_language = make_metric_by_language_plot(
            df=df,
            metric_name=metric_name,
            y_label=spec["y_label"],
            title=spec["title_by_language"],
            output_path=figures_dir / spec["filename_by_language"],
        )

        if by_language is not None:
            plots.append(by_language)

    return plots

def make_metric_by_vocab_size_plot(
    df: pd.DataFrame,
    metric_name: str,
    y_label: str,
    title: str,
    output_path: str | Path,
    line_column: str = "target_percent",
) -> str | None:
    """
    Строит график влияния размера словаря.

    По оси X:
        vocab_size

    По оси Y:
        выбранная метрика

    Отдельные линии:
        значения line_column, например target_percent.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"vocab_size", line_column, metric_name}

    if df.empty or not required_columns.issubset(df.columns):
        return None

    plot_df = df.dropna(subset=["vocab_size", line_column, metric_name]).copy()

    if plot_df.empty:
        return None

    plot_df["vocab_size"] = pd.to_numeric(
        plot_df["vocab_size"],
        errors="coerce",
    )

    plot_df[metric_name] = pd.to_numeric(
        plot_df[metric_name],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["vocab_size", metric_name])

    if plot_df.empty:
        return None

    aggregated = (
        plot_df
        .groupby([line_column, "vocab_size"], as_index=False)[metric_name]
        .mean()
        .sort_values([line_column, "vocab_size"])
    )

    if aggregated.empty:
        return None

    plt.figure(figsize=(6, 3.5))

    for line_value in sorted(aggregated[line_column].dropna().unique()):
        line_df = aggregated[aggregated[line_column] == line_value].sort_values(
            "vocab_size"
        )

        plt.plot(
            line_df["vocab_size"],
            line_df[metric_name],
            marker="o",
            label=str(line_value),
        )

    plt.xlabel("Размер словаря")
    plt.ylabel(y_label)
    plt.title(title)

    if line_column == "target_percent":
        plt.legend(title="Доля target, %")
    else:
        plt.legend(title=line_column)

    plt.grid(True, alpha=0.3)

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    return str(output_path)


def generate_vocab_size_analysis_plots(
    batch_all_results_path: str | Path,
) -> list[str]:
    """
    Строит графики влияния размера словаря.

    Использует:
        batch_all_results.csv

    Создаёт графики:
    - compression_factor_char_target vs vocab_size;
    - compression_penalty_char vs vocab_size;
    - ppl_target vs vocab_size;
    - gap_loss vs vocab_size.

    Линии на графиках:
        target_percent.
    """

    batch_all_results_path = Path(batch_all_results_path)
    batch_dir = batch_all_results_path.parent
    figures_dir = batch_dir / "analysis_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_safely(batch_all_results_path)

    if df is None or df.empty:
        return []

    plots: list[str] = []

    metric_specs = [
        {
            "metric_name": "compression_factor_char_target",
            "y_label": "Target chars per token",
            "title": "Влияние размера словаря на сжатие target",
            "filename": "vocab_compression_factor_target_by_ratio.png",
        },
        {
            "metric_name": "compression_penalty_char",
            "y_label": "EN factor - Target factor",
            "title": "Влияние размера словаря на штраф сжатия target",
            "filename": "vocab_compression_penalty_by_ratio.png",
        },
        {
            "metric_name": "ppl_target",
            "y_label": "Target PPL",
            "title": "Влияние размера словаря на PPL target",
            "filename": "vocab_ppl_target_by_ratio.png",
        },
        {
            "metric_name": "gap_loss",
            "y_label": "Gap loss",
            "title": "Влияние размера словаря на gap_loss",
            "filename": "vocab_gap_loss_by_ratio.png",
        },
    ]

    for spec in metric_specs:
        metric_name = spec["metric_name"]

        if metric_name not in df.columns:
            continue

        plot_path = make_metric_by_vocab_size_plot(
            df=df,
            metric_name=metric_name,
            y_label=spec["y_label"],
            title=spec["title"],
            output_path=figures_dir / spec["filename"],
            line_column="target_percent",
        )

        if plot_path is not None:
            plots.append(plot_path)

    return plots