from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError

from core.component_catalog import ComponentCatalog
from core.experiment_controller import ExperimentController
from core.experiment_system import ExperimentSystem
from core.params import ExperimentParams
from core.storage_manager import StorageManager

from services.text_cleaning import (
    clean_texts,
    read_paragraphs,
    train_test_split,
)

from services.mixture_service import (
    LANGUAGE_GROUPS,
    RATIOS_TOKENIZER_TRAINING,
    RATIOS_UNDERREPRESENTED,
    MixtureConfig,
    generate_mixtures,
)

from services.wiki40b_loader_service import (
    SUPPORTED_WIKI40B_LANGUAGES,
    load_wiki40b_languages,
)


from services.batch_analysis_service import (
    generate_batch_analysis_plots,
    generate_vocab_size_analysis_plots,
    save_batch_all_results,
)


DATA_DIR = Path("data") / "corpora"
SOURCE_CORPORA_DIR = Path("data") / "source_corpora"
GENERATED_MIXTURES_DIR = Path("data") / "generated_mixtures"
RESULTS_DIR = Path("results")

DATA_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_CORPORA_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_MIXTURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def show_compact_results_table(df: pd.DataFrame) -> None:
    """
    Показывает компактную исследовательскую таблицу результатов.

    Полная техническая таблица доступна в expander.
    """

    preferred_columns = [
        "method",
        "vocab_size",

        "tokens_per_char_en",
        "tokens_per_char_target",

        "compression_factor_char_en",
        "compression_factor_char_target",
        "compression_penalty_char",

        "ppl_en",
        "ppl_target",
        "gap_loss",
    ]

    visible_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    if visible_columns:
        st.dataframe(
            df[visible_columns],
            use_container_width=True,
        )
    else:
        st.dataframe(
            df,
            use_container_width=True,
        )

    with st.expander("Показать полную техническую таблицу"):
        st.dataframe(
            df,
            use_container_width=True,
        )

def save_uploaded_corpus(uploaded_file) -> str:
    """
    Сохраняет загруженный пользователем корпус в data/corpora.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(uploaded_file.name).name.replace(" ", "_")

    save_path = DATA_DIR / f"{timestamp}_{original_name}"

    save_path.write_bytes(uploaded_file.getbuffer())

    return str(save_path)


def list_available_corpora() -> list[str]:
    """
    Возвращает список доступных .txt корпусов.

    Смотрит в:
    - data/corpora;
    - data/generated_mixtures.
    """

    files = []

    for folder in [DATA_DIR, GENERATED_MIXTURES_DIR]:
        if folder.exists():
            files.extend(sorted(folder.glob("*.txt")))

    return [str(path) for path in files]


def list_generated_mixtures() -> list[str]:
    """
    Возвращает список сгенерированных mix_*.txt файлов.
    """

    if not GENERATED_MIXTURES_DIR.exists():
        return []

    mixture_files = sorted(GENERATED_MIXTURES_DIR.glob("mix_*.txt"))

    return [str(path) for path in mixture_files]


def parse_mixture_filename(path: str | Path) -> dict:
    """
    Извлекает метаданные из имени файла смеси.

    Ожидаемый формат:
        mix_{group}_{language}_{ratio}.txt

    Примеры:
        mix_slavic_ru_90_10.txt
        mix_germanic_de_75_25.txt
        mix_isolated_zh-cn_95_5.txt
    """

    path = Path(path)
    stem = path.stem

    parts = stem.split("_")

    if len(parts) < 4 or parts[0] != "mix":
        return {
            "group": "unknown",
            "target_lang": "custom",
            "ratio": "unknown",
        }

    group = parts[1]
    target_lang = parts[2]
    ratio = "_".join(parts[3:])

    return {
        "group": group,
        "target_lang": target_lang,
        "ratio": ratio,
    }


def make_batch_summary_row(
    corpus_path: str,
    params: ExperimentParams,
    result_package=None,
    status: str = "completed",
    error: str = "",
) -> dict:
    """
    Формирует одну строку сводки пакетного запуска.
    """

    mixture_info = parse_mixture_filename(corpus_path)

    row = {
        "corpus_path": corpus_path,
        "group": mixture_info["group"],
        "target_lang": mixture_info["target_lang"],
        "ratio": mixture_info["ratio"],
        "status": status,
        "error": error,
        "run_path": "",
        "completed_combinations": 0,
        "failed_combinations": 0,
        "best_tokens_per_char_method": "",
        "best_tokens_per_char_vocab": "",
        "best_tokens_per_char": "",
        "best_ppl_method": "",
        "best_ppl_vocab": "",
        "best_ppl": "",
    }

    if result_package is None:
        return row

    row["run_path"] = result_package.save_path
    row["completed_combinations"] = result_package.metrics.get("completed_combinations", 0)
    row["failed_combinations"] = result_package.metrics.get("failed_combinations", 0)

    best_tpc = result_package.metrics.get("best_tokens_per_char")

    if best_tpc:
        row["best_tokens_per_char_method"] = best_tpc.get("method", "")
        row["best_tokens_per_char_vocab"] = best_tpc.get("vocab_size", "")
        row["best_tokens_per_char"] = best_tpc.get("tokens_per_char", "")

    best_ppl = result_package.metrics.get("best_ppl")

    if best_ppl:
        row["best_ppl_method"] = best_ppl.get("method", "")
        row["best_ppl_vocab"] = best_ppl.get("vocab_size", "")
        row["best_ppl"] = best_ppl.get("ppl", "")

    return row


def get_selected_ratios(ratio_mode: str) -> dict[str, tuple[float, float]]:
    """
    Возвращает набор долей для генерации смешанных корпусов.
    """

    if ratio_mode == "Недопредставленные языки: 99.9/0.1, 99/1, 98/2, 95/5":
        return RATIOS_UNDERREPRESENTED

    if ratio_mode == "Обучение токенизатора: 90/10, 75/25, 50/50":
        return RATIOS_TOKENIZER_TRAINING

    if ratio_mode == "Все наборы долей":
        return {
            **RATIOS_UNDERREPRESENTED,
            **RATIOS_TOKENIZER_TRAINING,
        }

    return RATIOS_TOKENIZER_TRAINING


def list_batch_summary_files() -> list[str]:
    """
    Возвращает список всех batch_summary.csv.
    """

    batch_files = sorted(
        RESULTS_DIR.glob("batch_*/batch_summary.csv"),
        reverse=True,
    )

    return [str(path) for path in batch_files]


def show_compact_batch_results_table(df: pd.DataFrame) -> None:
    """
    Показывает компактную таблицу результатов пакетного запуска.

    Полная таблица остаётся доступной в раскрывающемся блоке.
    """

    preferred_columns = [
        "group",
        "target_lang",
        "ratio",
        "target_percent",
        "method",
        "vocab_size",

        "compression_factor_char_en",
        "compression_factor_char_target",
        "compression_penalty_char",

        "ppl_en",
        "ppl_target",
        "gap_loss",
    ]

    visible_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    if visible_columns:
        st.dataframe(
            df[visible_columns],
            use_container_width=True,
        )
    else:
        st.dataframe(
            df,
            use_container_width=True,
        )

    with st.expander("Показать полную техническую таблицу"):
        st.dataframe(
            df,
            use_container_width=True,
        )


def show_analysis_figures(plot_paths: list[str]) -> None:
    """
    Показывает графики анализа компактно по два в ряд.
    """

    if not plot_paths:
        st.info("Графики анализа пока не созданы.")
        return

    st.subheader("Графики анализа")

    paths = [Path(path) for path in plot_paths]
    paths = [path for path in paths if path.exists()]

    for index in range(0, len(paths), 2):
        cols = st.columns(2)

        for col, plot_path in zip(cols, paths[index:index + 2]):
            with col:
                st.image(
                    str(plot_path),
                    caption=plot_path.name,
                    width=450,
                )


def read_csv_safely(csv_path: str | Path) -> pd.DataFrame | None:
    """
    Безопасно читает CSV.

    Возвращает:
    - DataFrame, если файл читается нормально;
    - None, если файл отсутствует, пустой или повреждён.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        return None

    if csv_path.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(csv_path)

    except EmptyDataError:
        return None


def show_best_result(best_result: dict | None) -> None:
    """
    Красиво показывает лучший результат по tokens_per_char.
    """

    if not best_result:
        st.info("Лучший результат пока отсутствует.")
        return

    st.subheader("Лучший результат по tokens_per_char")

    col1, col2, col3 = st.columns(3)
    

    with col1:
        st.metric("Метод", best_result["method"])
        st.metric("Размер словаря", best_result["vocab_size"])
        st.metric("Количество текстов", best_result["n_texts"])

    with col2:
        st.metric("Tokens per char", f"{best_result['tokens_per_char']:.4f}")
        st.metric("Tokens per word", f"{best_result['tokens_per_word']:.4f}")
        st.metric("Количество токенов", best_result["n_tokens"])

    with col3:
        st.metric("Средняя длина токена", f"{best_result['avg_token_len']:.4f}")

    with st.expander("Подробнее о лучшем результате"):
        st.write(f"**Путь к токенизатору:** `{best_result['tokenizer_path']}`")


def show_best_target_compression(best_target: dict | None) -> None:
    """
    Показывает лучшее сжатие целевого языка.
    """

    if not best_target:
        return

    st.subheader("Лучшее сжатие целевого языка")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Метод", best_target["method"])
        st.metric("Размер словаря", best_target["vocab_size"])

    with col2:
        st.metric(
            "Target tokens per char",
            f"{best_target['tokens_per_char_target']:.4f}",
        )
        st.metric(
            "Target tokens per word",
            f"{best_target['tokens_per_word_target']:.4f}",
        )

    with col3:
        st.metric(
            "Target avg token len",
            f"{best_target['avg_token_len_target']:.4f}",
        )
        st.metric(
            "Target tokens",
            best_target["n_tokens_target"],
        )


def show_best_compression_gap(best_gap: dict | None) -> None:
    """
    Показывает минимальный разрыв сжатия между target и EN.
    """

    if not best_gap:
        return

    st.subheader("Минимальный разрыв сжатия target и EN")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Метод", best_gap["method"])
        st.metric("Размер словаря", best_gap["vocab_size"])

    with col2:
        st.metric(
            "Compression gap",
            f"{best_gap['compression_gap_tokens_per_char']:.4f}",
        )
        st.metric(
            "EN tokens per char",
            f"{best_gap['tokens_per_char_en']:.4f}",
        )

    with col3:
        st.metric(
            "Target tokens per char",
            f"{best_gap['tokens_per_char_target']:.4f}",
        )

def show_best_ppl(best_ppl: dict | None) -> None:
    """
    Красиво показывает лучший результат по perplexity.
    """

    if not best_ppl:
        return

    st.subheader("Лучший результат по perplexity")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Метод", best_ppl["method"])
        st.metric("Размер словаря", best_ppl["vocab_size"])

    with col2:
        st.metric("PPL", f"{best_ppl['ppl']:.4f}")
        st.metric("Eval loss", f"{best_ppl['eval_loss']:.4f}")

    with col3:
        st.metric("Train loss", f"{best_ppl['train_loss']:.4f}")
        st.metric("Train/Test blocks", f"{best_ppl['train_blocks']} / {best_ppl['test_blocks']}")


def show_best_gap_loss(best_gap_loss: dict | None) -> None:
    """
    Красиво показывает лучший результат по gap_loss.
    """

    if not best_gap_loss:
        return

    st.subheader("Лучший результат по gap_loss")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Метод", best_gap_loss["method"])
        st.metric("Размер словаря", best_gap_loss["vocab_size"])

    with col2:
        st.metric("Gap loss", f"{best_gap_loss['gap_loss']:.4f}")
        st.metric("Loss target", f"{best_gap_loss['loss_target']:.4f}")

    with col3:
        st.metric("Loss EN", f"{best_gap_loss['loss_en']:.4f}")
        st.metric(
            "PPL target / EN",
            f"{best_gap_loss['ppl_target']:.2f} / {best_gap_loss['ppl_en']:.2f}",
        )


def show_result_package(result_package, storage: StorageManager) -> None:
    """
    Отображает пакет результатов после выполнения эксперимента.
    """

    run_dir = Path(result_package.save_path)

    st.success(f"Эксперимент завершён. Папка запуска: `{run_dir}`")

    st.subheader("Сводные метрики")

    summary_metrics = {
        "Фрагментов до очистки": result_package.metrics.get("raw_texts"),
        "Фрагментов после очистки": result_package.metrics.get("cleaned_texts"),
        "Train": result_package.metrics.get("train_texts"),
        "Test": result_package.metrics.get("test_texts"),
        "Test EN": result_package.metrics.get("test_en_texts"),
        "Test target": result_package.metrics.get("test_target_texts"),
        "Test unknown": result_package.metrics.get("test_unknown_texts"),
        "Стратегия разбиения": result_package.metrics.get("split_strategy"),
        "Успешных комбинаций": result_package.metrics.get("completed_combinations"),
        "Ошибочных комбинаций": result_package.metrics.get("failed_combinations"),
    }

    st.json(summary_metrics)

    show_best_result(result_package.metrics.get("best_tokens_per_char"))
    show_best_target_compression(
        result_package.metrics.get("best_target_tokens_per_char")
    )

    show_best_compression_gap(
        result_package.metrics.get("best_compression_gap_tokens_per_char")
    )
    show_best_ppl(result_package.metrics.get("best_ppl"))
    show_best_gap_loss(result_package.metrics.get("best_gap_loss"))

    st.subheader("Таблица результатов")

    results_csv_path = result_package.tables.get("experiment_results")

    if results_csv_path:
        results_df = read_csv_safely(results_csv_path)

        if results_df is not None:
            show_compact_results_table(results_df)
        else:
            st.warning(
                "Таблица результатов пуста. Вероятно, все комбинации эксперимента завершились с ошибкой."
            )
    else:
        st.warning("Файл таблицы результатов не найден.")

    if result_package.plots:
        st.subheader("Графики")

        for plot_path in result_package.plots:
            plot_path = Path(plot_path)

            if plot_path.exists():
                st.image(
                    str(plot_path),
                    caption=plot_path.name,
                    width=650,
                )

    st.subheader("Отчёт")
    st.markdown(result_package.report)

    with st.expander("Логи выполнения"):
        st.code(result_package.logs)

    if result_package.errors:
        with st.expander("Ошибки"):
            st.json(result_package.errors)

    zip_path = storage.zip_run(run_dir)

    st.download_button(
        "Скачать пакет результатов (.zip)",
        data=zip_path.read_bytes(),
        file_name=zip_path.name,
        mime="application/zip",
        key=f"download_result_package_{run_dir.name}",
    )


def show_saved_run(run_id: str, storage: StorageManager) -> None:
    """
    Показывает ранее сохранённый запуск.
    """

    run_dir = RESULTS_DIR / run_id

    if not run_dir.exists():
        st.error("Папка выбранного запуска не найдена.")
        return

    st.subheader(f"Просмотр запуска `{run_id}`")

    status_path = run_dir / "status.json"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.md"
    results_csv_path = run_dir / "tables" / "experiment_results.csv"
    logs_path = run_dir / "logs.txt"
    errors_csv_path = run_dir / "tables" / "errors.csv"

    col1, col2 = st.columns(2)

    with col1:
        if status_path.exists():
            st.write("**Статус:**")
            st.json(storage.load_json(status_path))

    with col2:
        if config_path.exists():
            st.write("**Конфигурация:**")
            st.json(storage.load_json(config_path))

    if errors_csv_path.exists():
        errors_df = read_csv_safely(errors_csv_path)

        if errors_df is not None:
            st.write("**Ошибки выполнения:**")
            show_compact_results_table(results_df)

    if metrics_path.exists():
        st.write("**Метрики:**")
        metrics = storage.load_json(metrics_path)

        summary_metrics = {
            "Фрагментов до очистки": metrics.get("raw_texts"),
            "Фрагментов после очистки": metrics.get("cleaned_texts"),
            "Train": metrics.get("train_texts"),
            "Test": metrics.get("test_texts"),
            "Test EN": metrics.get("test_en_texts"),
            "Test target": metrics.get("test_target_texts"),
            "Test unknown": metrics.get("test_unknown_texts"),
            "Стратегия разбиения": metrics.get("split_strategy"),
            "Успешных комбинаций": metrics.get("completed_combinations"),
            "Ошибочных комбинаций": metrics.get("failed_combinations"),
        }

        st.json(summary_metrics)

        show_best_result(
            metrics.get("best_tokens_per_char"))
        
        show_best_target_compression(
            metrics.get("best_target_tokens_per_char")
        )

        show_best_compression_gap(
            metrics.get("best_compression_penalty_char")
        )
        show_best_ppl(
            metrics.get("best_ppl"))
        
        show_best_gap_loss(
            metrics.get("best_gap_loss"))

    if results_csv_path.exists():
        st.write("**Таблица результатов:**")

        results_df = read_csv_safely(results_csv_path)

        if results_df is not None:
            st.dataframe(
                results_df,
                use_container_width=True,
            )
        else:
            st.warning(
                "Таблица результатов пуста. Вероятно, этот запуск завершился ошибкой до получения успешных результатов."
            )

    figures_dir = run_dir / "figures"
    figure_paths = sorted(figures_dir.glob("*.png"))

    if figure_paths:
        st.write("**Графики:**")

        for figure_path in figure_paths:
            st.image(
                str(figure_path),
                caption=figure_path.name,
                width=650,
            )

    if report_path.exists():
        st.write("**Отчёт:**")
        st.markdown(report_path.read_text(encoding="utf-8"))

    if logs_path.exists():
        with st.expander("Логи"):
            st.code(logs_path.read_text(encoding="utf-8"))

    zip_path = storage.zip_run(run_dir)

    st.download_button(
        "Скачать выбранный запуск (.zip)",
        data=zip_path.read_bytes(),
        file_name=zip_path.name,
        mime="application/zip",
        key=f"download_saved_run_{run_id}",
    )


st.set_page_config(
    page_title="Система исследования методов токенизации",
    layout="wide",
)

st.title("Система исследования методов токенизации")
st.write(
    "Интерфейс теперь позволяет загружать корпус, выбирать параметры "
    "и запускать эксперимент без изменения кода."
)


catalog = ComponentCatalog()
storage = StorageManager(root_dir=RESULTS_DIR)

system = ExperimentSystem(
    storage=storage,
    catalog=catalog,
)

controller = ExperimentController(
    system=system,
)


tab_wiki40b, tab_mixtures, tab_batch, tab_summary, tab_new, tab_history, tab_catalog = st.tabs(
    [
        "Загрузка Wiki40B",
        "Генерация выборок",
        "Пакетный запуск",
        "Сводные результаты",
        "Новый эксперимент",
        "Прошлые результаты",
        "Каталог компонентов",
    ]
)


with tab_wiki40b:
    st.header("Загрузка исходных корпусов из Wiki40B")

    st.write(
        "Этот раздел загружает тексты из Wiki40B и сохраняет их локально "
        "в `data/source_corpora/{lang}.txt`."
    )

    st.warning(
        "Не загружайте сразу все языки на большом sample_size. "
        "Сначала проверьте 1–2 языка и 20–100 фрагментов."
    )

    selected_wiki40b_languages = st.multiselect(
        "Языки для загрузки",
        options=SUPPORTED_WIKI40B_LANGUAGES,
        default=["en", "ru"],
    )

    wiki40b_output_dir = st.text_input(
        "Папка сохранения",
        value=str(SOURCE_CORPORA_DIR),
    )

    wiki40b_split = st.selectbox(
        "Split",
        options=["train", "validation", "test"],
        index=0,
    )

    wiki40b_max_samples = st.number_input(
        "Количество фрагментов на язык",
        min_value=10,
        value=100,
        step=10,
    )

    wiki40b_streaming = st.checkbox(
        "Использовать streaming",
        value=True,
        help="Streaming позволяет не скачивать весь датасет целиком.",
    )

    wiki40b_seed = st.number_input(
        "Seed",
        min_value=0,
        value=42,
        step=1,
        key="wiki40b_seed",
    )

    if st.button(
        "Загрузить выбранные языки",
        type="primary",
        disabled=not selected_wiki40b_languages,
    ):
        with st.spinner("Загрузка Wiki40B..."):
            manifest_df = load_wiki40b_languages(
                languages=selected_wiki40b_languages,
                output_dir=wiki40b_output_dir,
                split=wiki40b_split,
                max_samples=int(wiki40b_max_samples),
                streaming=bool(wiki40b_streaming),
                seed=int(wiki40b_seed),
            )

        completed_count = int((manifest_df["status"] == "completed").sum())
        failed_count = int((manifest_df["status"] == "failed").sum())

        st.success(
            f"Загрузка завершена. Успешно: {completed_count}. Ошибок: {failed_count}."
        )

        st.dataframe(
            manifest_df,
            use_container_width=True,
        )

    st.subheader("Локальные исходные корпуса")

    source_files = sorted(SOURCE_CORPORA_DIR.glob("*.txt"))

    if not source_files:
        st.info("В `data/source_corpora` пока нет txt-файлов.")
    else:
        source_rows = []

        for path in source_files:
            source_rows.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "size_kb": round(path.stat().st_size / 1024, 2),
                }
            )

        st.dataframe(
            pd.DataFrame(source_rows),
            use_container_width=True,
        )


with tab_mixtures:
    st.header("Генерация смешанных корпусов")

    st.write(
        "Этот раздел формирует выборки вида: английский язык + целевой язык "
        "с заданным процентным соотношением."
    )

    st.info(
        "Ожидается, что исходные корпуса лежат в папке "
        "`data/source_corpora/` и называются `en.txt`, `ru.txt`, `de.txt` и т.д."
    )

    st.subheader("1. Папки данных")

    source_dir = st.text_input(
        "Папка исходных корпусов",
        value=str(SOURCE_CORPORA_DIR),
        help="Здесь должны лежать en.txt, ru.txt, bg.txt, cs.txt и другие исходные корпуса.",
    )

    output_dir = st.text_input(
        "Папка для сохранения смешанных корпусов",
        value=str(GENERATED_MIXTURES_DIR),
    )

    st.subheader("2. Выбор языков")

    all_target_languages = [
        language
        for languages in LANGUAGE_GROUPS.values()
        for language in languages
    ]

    selected_languages = st.multiselect(
        "Целевые языки",
        options=all_target_languages,
        default=["ru"] if "ru" in all_target_languages else [],
    )

    with st.expander("Группы языков"):
        st.json(LANGUAGE_GROUPS)

    st.subheader("3. Выбор процентных соотношений")

    ratio_mode = st.radio(
        "Набор долей",
        options=[
            "Недопредставленные языки: 99.9/0.1, 99/1, 98/2, 95/5",
            "Обучение токенизатора: 90/10, 75/25, 50/50",
            "Все наборы долей",
        ],
        index=1,
    )

    selected_ratios = get_selected_ratios(ratio_mode)

    st.write("Будут использованы доли:")

    ratios_preview = pd.DataFrame(
        [
            {
                "ratio": ratio_name,
                "english_percent": values[0],
                "target_percent": values[1],
            }
            for ratio_name, values in selected_ratios.items()
        ]
    )

    st.dataframe(ratios_preview, use_container_width=True)

    st.subheader("4. Параметры генерации")

    col_mix_1, col_mix_2 = st.columns(2)

    with col_mix_1:
        sample_size = st.number_input(
            "sample_size",
            min_value=10,
            value=1000,
            step=10,
            help=(
                "Сколько фрагментов использовать для формирования каждой смеси. "
                "Для доли 0.1% нужно минимум около 1000 фрагментов, иначе целевых текстов будет 0."
            ),
        )

    with col_mix_2:
        mixture_seed = st.number_input(
            "Seed генерации",
            min_value=0,
            value=42,
            step=1,
        )

    st.subheader("5. Проверка исходных файлов")

    source_path = Path(source_dir)

    expected_files = ["en.txt"] + [f"{language}.txt" for language in selected_languages]

    file_check_rows = []

    for filename in expected_files:
        path = source_path / filename

        file_check_rows.append(
            {
                "file": filename,
                "path": str(path),
                "exists": path.exists(),
            }
        )

    file_check_df = pd.DataFrame(file_check_rows)

    st.dataframe(file_check_df, use_container_width=True)

    missing_files = [
        row["file"]
        for row in file_check_rows
        if not row["exists"]
    ]

    if missing_files:
        st.warning(
            "Не найдены исходные файлы: "
            + ", ".join(missing_files)
        )
    else:
        st.success("Все необходимые исходные файлы найдены.")

    st.subheader("6. Запуск генерации")

    if st.button(
        "Сформировать смешанные выборки",
        type="primary",
        disabled=bool(missing_files) or not selected_languages,
    ):
        try:
            config = MixtureConfig(
                source_dir=source_dir,
                output_dir=output_dir,
                selected_languages=selected_languages,
                selected_ratios=selected_ratios,
                sample_size=int(sample_size),
                seed=int(mixture_seed),
            )

            mixtures_df = generate_mixtures(config)

            manifest_path = Path(output_dir) / "generation_manifest.csv"
            mixtures_df.to_csv(
                manifest_path,
                index=False,
                encoding="utf-8",
            )

            created_count = int((mixtures_df["status"] == "created").sum())
            failed_count = int((mixtures_df["status"] == "failed").sum())

            st.success(
                f"Генерация завершена. Создано файлов: {created_count}. "
                f"Ошибок: {failed_count}."
            )

            st.write(f"Манифест сохранён: `{manifest_path}`")

            st.subheader("Созданные выборки")
            st.dataframe(mixtures_df, use_container_width=True)

            if failed_count > 0:
                st.warning("Некоторые выборки не были созданы. Проверьте колонку `error`.")

        except Exception as error:
            st.error(f"Ошибка генерации выборок: {error}")


with tab_batch:
    st.header("Пакетный запуск экспериментов")

    st.write(
        "Этот раздел позволяет запустить эксперименты сразу по нескольким "
        "сгенерированным смешанным корпусам."
    )

    st.info(
        "Ожидается, что файлы лежат в `data/generated_mixtures/` "
        "и имеют имена вида `mix_slavic_ru_90_10.txt`."
    )

    mixture_files = list_generated_mixtures()

    if not mixture_files:
        st.warning(
            "Сгенерированные смеси не найдены. "
            "Сначала создайте их во вкладке `Генерация выборок`."
        )

    else:
        st.subheader("1. Выбор корпусов")

        selected_mixture_files = st.multiselect(
            "Выберите mix-файлы для пакетного запуска",
            options=mixture_files,
            default=mixture_files[: min(3, len(mixture_files))],
        )

        if selected_mixture_files:
            preview_rows = []

            for file_path in selected_mixture_files:
                info = parse_mixture_filename(file_path)

                preview_rows.append(
                    {
                        "path": file_path,
                        "group": info["group"],
                        "target_lang": info["target_lang"],
                        "ratio": info["ratio"],
                    }
                )

            st.write("Выбранные файлы:")
            st.dataframe(
                pd.DataFrame(preview_rows),
                use_container_width=True,
            )

        st.subheader("2. Общие параметры экспериментов")

        col_batch_1, col_batch_2, col_batch_3 = st.columns(3)

        with col_batch_1:
            batch_tokenizer_methods = st.multiselect(
                "Методы токенизации",
                options=catalog.get_tokenizers(),
                default=["bpe", "wordpiece"],
                key="batch_tokenizer_methods",
            )

        with col_batch_2:
            batch_vocab_sizes = st.multiselect(
                "Размеры словаря",
                options=[500, 1000, 2000, 4000, 8000, 16000, 32000, 64000],
                default=[1000, 2000],
                key="batch_vocab_sizes",
            )

        with col_batch_3:
            batch_test_fraction = st.slider(
                "Доля тестовой выборки",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="batch_test_fraction",
            )

            batch_seed = st.number_input(
                "Seed",
                min_value=0,
                value=42,
                step=1,
                key="batch_seed",
            )

            batch_min_texts = st.number_input(
                "Минимум фрагментов после очистки",
                min_value=2,
                value=10,
                step=1,
                key="batch_min_texts",
            )

            batch_split_strategy_label = st.selectbox(
                "Стратегия разбиения train/test",
                options=[
                    "Стратифицированное по языкам",
                    "Случайное",
                ],
                index=0,
                key="batch_split_strategy_label",
                help=(
                    "Для gap_loss рекомендуется стратифицированное разбиение."
                ),
            )

            batch_split_strategy = (
                "stratified_by_language"
                if batch_split_strategy_label == "Стратифицированное по языкам"
                else "random"
            )

        st.subheader("3. Языковая модель")

        batch_run_language_model = st.checkbox(
            "Запустить tiny GPT-2 для расчёта perplexity",
            value=False,
            help=(
                "Для пакетного запуска этот режим может быть долгим. "
                "Сначала лучше проверить без tiny GPT-2."
            ),
            key="batch_run_language_model",
        )

        with st.expander("Настройки tiny GPT-2 для пакетного запуска"):
            lm_b1, lm_b2, lm_b3, lm_b4 = st.columns(4)

            with lm_b1:
                batch_max_steps = st.number_input(
                    "max_steps",
                    min_value=1,
                    value=5,
                    step=5,
                    key="batch_max_steps",
                )

            with lm_b2:
                batch_block_size = st.number_input(
                    "block_size",
                    min_value=8,
                    value=32,
                    step=8,
                    key="batch_block_size",
                )

            with lm_b3:
                batch_batch_size = st.number_input(
                    "batch_size",
                    min_value=1,
                    value=1,
                    step=1,
                    key="batch_batch_size",
                )

            with lm_b4:
                batch_learning_rate = st.number_input(
                    "learning_rate",
                    min_value=0.000001,
                    value=0.0005,
                    format="%.6f",
                    key="batch_learning_rate",
                )

        st.subheader("4. Проверка параметров пакетного запуска")

        batch_validation_errors = []

        if not selected_mixture_files:
            batch_validation_errors.append("Нужно выбрать хотя бы один mix-файл.")

        if not batch_tokenizer_methods:
            batch_validation_errors.append("Нужно выбрать хотя бы один метод токенизации.")

        if not batch_vocab_sizes:
            batch_validation_errors.append("Нужно выбрать хотя бы один размер словаря.")

        if batch_run_language_model and len(selected_mixture_files) > 3:
            st.warning(
                "Вы выбрали больше 3 корпусов и включили tiny GPT-2. "
                "Пакетный запуск может выполняться долго."
            )

        if batch_validation_errors:
            st.warning("Параметры пакетного запуска пока некорректны:")
            for error in batch_validation_errors:
                st.write(f"- {error}")
        else:
            st.success("Параметры пакетного запуска корректны.")

        st.subheader("5. Запуск пакета")

        if st.button(
            "Запустить пакет экспериментов",
            type="primary",
            disabled=bool(batch_validation_errors),
        ):
            batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = RESULTS_DIR / f"batch_{batch_timestamp}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            batch_rows = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_files = len(selected_mixture_files)

            for index, corpus_file in enumerate(selected_mixture_files, start=1):
                mixture_info = parse_mixture_filename(corpus_file)
                target_lang = mixture_info["target_lang"]

                status_text.write(
                    f"Выполняется {index}/{total_files}: `{Path(corpus_file).name}`"
                )

                params = ExperimentParams(
                    language=target_lang,
                    corpus_path=corpus_file,
                    tokenizer_methods=batch_tokenizer_methods,
                    vocab_sizes=[int(size) for size in batch_vocab_sizes],
                    model_names=["tiny_gpt2"] if batch_run_language_model else ["none"],
                    metrics=(
                        ["compression", "perplexity"]
                        if batch_run_language_model
                        else ["compression"]
                    ),
                    test_fraction=float(batch_test_fraction),
                    min_texts=int(batch_min_texts),
                    seed=int(batch_seed),
                    run_language_model=bool(batch_run_language_model),
                    max_steps=int(batch_max_steps),
                    block_size=int(batch_block_size),
                    batch_size=int(batch_batch_size),
                    learning_rate=float(batch_learning_rate),
                    split_strategy=batch_split_strategy,
                )

                validation_errors = controller.validate(params)

                if validation_errors:
                    error_message = "; ".join(validation_errors)

                    batch_rows.append(
                        make_batch_summary_row(
                            corpus_path=corpus_file,
                            params=params,
                            result_package=None,
                            status="validation_failed",
                            error=error_message,
                        )
                    )

                    progress_bar.progress(index / total_files)
                    continue

                try:
                    result_package = controller.create_and_execute(params)

                    run_status = "completed"

                    if result_package.errors:
                        run_status = "partially_completed"

                    batch_rows.append(
                        make_batch_summary_row(
                            corpus_path=corpus_file,
                            params=params,
                            result_package=result_package,
                            status=run_status,
                            error="",
                        )
                    )

                except Exception as error:
                    batch_rows.append(
                        make_batch_summary_row(
                            corpus_path=corpus_file,
                            params=params,
                            result_package=None,
                            status="failed",
                            error=repr(error),
                        )
                    )

                progress_bar.progress(index / total_files)

            batch_summary_df = pd.DataFrame(batch_rows)

            batch_summary_path = batch_dir / "batch_summary.csv"

            batch_summary_df.to_csv(
                batch_summary_path,
                index=False,
                encoding="utf-8",
            )

            status_text.write("Пакетный запуск завершён.")

            st.success(
                f"Пакетный запуск завершён. Сводка сохранена: `{batch_summary_path}`"
            )

            st.subheader("Сводка пакетного запуска")
            st.dataframe(batch_summary_df, use_container_width=True)

            st.download_button(
                "Скачать batch_summary.csv",
                data=batch_summary_path.read_bytes(),
                file_name=batch_summary_path.name,
                mime="text/csv",
                key=f"download_batch_summary_after_run_{batch_dir.name}",
            )

with tab_summary:
    st.header("Сводные результаты пакетного запуска")

    st.write(
        "Этот раздел собирает результаты пакетных экспериментов в единую таблицу "
        "и отображает базовые графики. Интерпретацию результатов выполняет пользователь."
    )

    batch_summary_files = list_batch_summary_files()

    if not batch_summary_files:
        st.warning(
            "Файлы batch_summary.csv не найдены. "
            "Сначала выполните пакетный запуск."
        )

    else:
        selected_batch_summary = st.selectbox(
            "Выберите batch_summary.csv",
            options=batch_summary_files,
        )

        batch_summary_path = Path(selected_batch_summary)
        batch_dir = batch_summary_path.parent
        batch_all_results_path = batch_dir / "batch_all_results.csv"

        st.subheader("1. Сводка пакетного запуска")

        summary_df = read_csv_safely(batch_summary_path)

        if summary_df is not None:
            st.dataframe(
                summary_df,
                use_container_width=True,
            )

            st.download_button(
                "Скачать batch_summary.csv",
                data=batch_summary_path.read_bytes(),
                file_name=batch_summary_path.name,
                mime="text/csv",
                key=f"download_batch_summary_{batch_dir.name}",
            )
        else:
            st.warning("Не удалось прочитать batch_summary.csv.")

        st.subheader("2. Полная таблица результатов")

        if st.button("Собрать batch_all_results.csv", type="primary"):
            try:
                output_path = save_batch_all_results(batch_summary_path)

                st.success(f"Файл создан: `{output_path}`")

            except Exception as error:
                st.error(f"Не удалось собрать batch_all_results.csv: {error}")

        if batch_all_results_path.exists():
            st.write(f"Найдена полная таблица: `{batch_all_results_path}`")

            all_results_df = read_csv_safely(batch_all_results_path)

            if all_results_df is not None:
                show_compact_batch_results_table(all_results_df)

                st.download_button(
                    "Скачать batch_all_results.csv",
                    data=batch_all_results_path.read_bytes(),
                    file_name=batch_all_results_path.name,
                    mime="text/csv",
                    key=f"download_batch_all_results_{batch_dir.name}",
                )

            else:
                st.warning("batch_all_results.csv пустой или повреждён.")

        else:
            st.info(
                "Полная таблица ещё не создана. "
                "Нажмите кнопку `Собрать batch_all_results.csv`."
            )

        st.subheader("3. Сводные графики")

        st.write(
            "Графики показывают основные метрики из плана работы: "
            "сжатие целевого языка, разрыв сжатия, PPL и gap_loss."
        )

        if st.button(
            "Построить сводные графики",
            disabled=not batch_all_results_path.exists(),
        ):
            try:
                plot_paths = generate_batch_analysis_plots(batch_all_results_path)

                st.success(f"Построено графиков: {len(plot_paths)}")

            except Exception as error:
                st.error(f"Не удалось построить графики: {error}")

        existing_plot_paths = sorted((batch_dir / "analysis_figures").glob("*.png"))

        if existing_plot_paths:
            show_analysis_figures([str(path) for path in existing_plot_paths])


with tab_new:
    st.header("Новый эксперимент")

    st.subheader("1. Корпус текстов")

    uploaded_file = st.file_uploader(
    "Загрузите текстовый корпус (.txt)",
    type=["txt"],
    )

    if "uploaded_corpus_path" not in st.session_state:
        st.session_state.uploaded_corpus_path = ""

    if uploaded_file is not None:
        saved_path = save_uploaded_corpus(uploaded_file)
        st.session_state.uploaded_corpus_path = saved_path
        st.success(f"Корпус сохранён: `{saved_path}`")

    available_corpora = list_available_corpora()

    selected_existing_corpus = st.selectbox(
        "Или выберите уже существующий корпус",
        options=[""] + available_corpora,
        format_func=lambda value: "Не выбрано" if value == "" else value,
    )

    default_corpus_path = (
        st.session_state.uploaded_corpus_path
        or selected_existing_corpus
        or "data/corpora/demo.txt"
    )

    corpus_path = st.text_input(
        "Путь к корпусу",
        value=default_corpus_path,
        help="Можно загрузить файл выше, выбрать сгенерированный корпус или указать путь вручную.",
    )

    st.subheader("2. Параметры эксперимента")

    col1, col2, col3 = st.columns(3)

    with col1:
        language = st.selectbox(
            "Целевой язык",
            options=[
                "ru",
                "bg",
                "cs",
                "de",
                "nl",
                "da",
                "ja",
                "zh-cn",
                "ko",
                "custom",
            ],
            index=0,
        )

        tokenizer_methods = st.multiselect(
            "Методы токенизации",
            options=catalog.get_tokenizers(),
            default=["bpe", "wordpiece"],
        )

    with col2:
        vocab_sizes = st.multiselect(
            "Размеры словаря",
            options=[500, 1000, 2000, 4000, 8000, 16000, 32000, 64000],
            default=[1000, 2000],
        )

        test_fraction = st.slider(
            "Доля тестовой выборки",
            min_value=0.05,
            max_value=0.5,
            value=0.2,
            step=0.05,
        )

    with col3:
        seed = st.number_input(
            "Seed",
            min_value=0,
            value=42,
            step=1,
        )

        min_texts = st.number_input(
            "Минимум фрагментов после очистки",
            min_value=2,
            value=10,
            step=1,
        )

        split_strategy_label = st.selectbox(
            "Стратегия разбиения train/test",
            options=[
                "Стратифицированное по языкам",
                "Случайное",
            ],
            index=0,
            help=(
                "Для расчёта gap_loss лучше использовать стратифицированное "
                "разбиение, чтобы в test попадали и английские, и целевые тексты."
            ),
        )

        split_strategy = (
            "stratified_by_language"
            if split_strategy_label == "Стратифицированное по языкам"
            else "random"
        )

    st.subheader("3. Параметры языковой модели")

    run_language_model = st.checkbox(
        "Запустить tiny GPT-2 для расчёта perplexity",
        value=False,
        help=(
            "Если включить, эксперимент будет выполняться дольше, "
            "но появится метрика perplexity."
        ),
    )

    with st.expander("Настройки tiny GPT-2"):
        lm_col1, lm_col2, lm_col3, lm_col4 = st.columns(4)

        with lm_col1:
            max_steps = st.number_input(
                "max_steps",
                min_value=1,
                value=30,
                step=10,
            )

        with lm_col2:
            block_size = st.number_input(
                "block_size",
                min_value=8,
                value=128,
                step=8,
            )

        with lm_col3:
            batch_size = st.number_input(
                "batch_size",
                min_value=1,
                value=4,
                step=1,
            )

        with lm_col4:
            learning_rate = st.number_input(
                "learning_rate",
                min_value=0.000001,
                value=0.0005,
                format="%.6f",
            )

    params = ExperimentParams(
        language=language,
        corpus_path=corpus_path,
        tokenizer_methods=tokenizer_methods,
        vocab_sizes=[int(size) for size in vocab_sizes],
        model_names=["tiny_gpt2"] if run_language_model else ["none"],
        metrics=["compression", "perplexity"] if run_language_model else ["compression"],
        test_fraction=float(test_fraction),
        min_texts=int(min_texts),
        seed=int(seed),
        run_language_model=bool(run_language_model),
        max_steps=int(max_steps),
        block_size=int(block_size),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        split_strategy=split_strategy,
    )

    st.subheader("3. Проверка параметров")

    validation_errors = controller.validate(params)

    if validation_errors:
        st.warning("Параметры пока некорректны:")
        for error in validation_errors:
            st.write(f"- {error}")
    else:
        st.success("Параметры корректны. Эксперимент можно запускать.")

    with st.expander("Показать текущую конфигурацию"):
        st.json(params.to_dict())

    st.subheader("4. Предпросмотр корпуса")

    if not validation_errors:
        try:
            raw_texts = read_paragraphs(params.corpus_path)
            cleaned_texts = clean_texts(raw_texts, lang=params.language)

            train_texts, test_texts = train_test_split(
                cleaned_texts,
                test_fraction=params.test_fraction,
                seed=params.seed,
            )

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("До очистки", len(raw_texts))

            with col_b:
                st.metric("После очистки", len(cleaned_texts))

            with col_c:
                st.metric("Train/Test", f"{len(train_texts)} / {len(test_texts)}")

            with st.expander("Примеры очищенных фрагментов"):
                for index, text in enumerate(cleaned_texts[:5], start=1):
                    st.markdown(f"**Фрагмент {index}:**")
                    st.write(text)

        except Exception as error:
            st.error(f"Не удалось выполнить предпросмотр корпуса: {error}")

    else:
        st.info("Предпросмотр появится после исправления параметров.")

    st.subheader("5. Запуск")

    if st.button(
        "Запустить эксперимент",
        type="primary",
        disabled=bool(validation_errors),
    ):
        with st.spinner("Эксперимент выполняется..."):
            try:
                result_package = controller.create_and_execute(params)
                show_result_package(result_package, storage)

            except Exception as error:
                st.error(f"Эксперимент завершился с ошибкой: {error}")


with tab_history:
    st.header("Прошлые результаты")

    runs = controller.get_saved_runs()

    if not runs:
        st.info("Сохранённых запусков пока нет.")
    else:
        runs_df = pd.DataFrame(runs)

        st.dataframe(
            runs_df,
            use_container_width=True,
        )

        selected_run_id = st.selectbox(
            "Выберите запуск для просмотра",
            options=[run["run_id"] for run in runs],
        )

        if selected_run_id:
            show_saved_run(selected_run_id, storage)


with tab_catalog:
    st.header("Каталог компонентов")

    st.subheader("Токенизаторы")
    st.json(catalog.tokenizers)

    st.subheader("Модели")
    st.json(catalog.models)