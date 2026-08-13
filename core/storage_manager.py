from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


class StorageManager:
    """
    МенеджерХранилища.

    Отвечает за:
    - создание папки запуска;
    - сохранение конфигурации;
    - сохранение статуса;
    - сохранение таблиц;
    - сохранение пакета результатов;
    - сохранение логов;
    - упаковку результатов в zip.
    """

    def __init__(self, root_dir: str | Path = "results") -> None:
        """
        root_dir — корневая папка для всех результатов.
        По умолчанию это папка results.
        """

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self) -> Path:
        """
        Создаёт отдельную папку для нового запуска эксперимента.

        Пример:
            results/run_20260501_153000_a1b2c3d4
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid4().hex[:8]

        run_dir = self.root_dir / f"run_{timestamp}_{short_id}"

        # Основные подпапки запуска
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)
        (run_dir / "tokenizers").mkdir(parents=True, exist_ok=True)
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)

        return run_dir

    def save_json(self, path: str | Path, data: dict[str, Any]) -> None:
        """
        Сохраняет словарь в JSON-файл.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_json(self, path: str | Path) -> dict[str, Any]:
        """
        Загружает JSON-файл и возвращает словарь.
        """

        path = Path(path)

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def save_dataframe(self, path: str | Path, df: pd.DataFrame) -> None:
        """
        Сохраняет pandas DataFrame в CSV-файл.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            path,
            index=False,
            encoding="utf-8",
        )

    def save_text(self, path: str | Path, text: str) -> None:
        """
        Сохраняет обычный текстовый файл.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            text,
            encoding="utf-8",
        )

    def copy_corpus(self, corpus_path: str | Path, run_dir: str | Path) -> Path:
        """
        Копирует исходный корпус в папку запуска.

        Это нужно для воспроизводимости:
        внутри папки запуска будет лежать не только результат,
        но и копия исходных данных.
        """

        corpus_path = Path(corpus_path)
        run_dir = Path(run_dir)

        destination = run_dir / "corpus.txt"

        shutil.copyfile(
            corpus_path,
            destination,
        )

        return destination

    def update_status(
        self,
        run_dir: str | Path,
        status: str,
        message: str = "",
    ) -> None:
        """
        Обновляет файл status.json.

        Возможные статусы:
            created
            running
            completed
            partially_completed
            failed
        """

        run_dir = Path(run_dir)

        status_data = {
            "run_id": run_dir.name,
            "status": status,
            "message": message,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

        self.save_json(
            run_dir / "status.json",
            status_data,
        )

    def setup_logger(self, run_dir: str | Path) -> logging.Logger:
        """
        Создаёт logger для конкретного запуска.

        Все сообщения будут записываться в:
            logs.txt
        """

        run_dir = Path(run_dir)

        logger_name = f"experiment.{run_dir.name}"
        logger = logging.getLogger(logger_name)

        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        log_path = run_dir / "logs.txt"

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def list_runs(self) -> list[dict[str, Any]]:
        """
        Возвращает список сохранённых запусков.

        Используется потом во вкладке:
            Прошлые результаты
        """

        runs = []

        for run_dir in sorted(self.root_dir.glob("run_*"), reverse=True):
            status_path = run_dir / "status.json"
            config_path = run_dir / "config.json"

            status_data = {}
            config_data = {}

            if status_path.exists():
                try:
                    status_data = self.load_json(status_path)
                except Exception:
                    status_data = {}

            if config_path.exists():
                try:
                    config_data = self.load_json(config_path)
                except Exception:
                    config_data = {}

            runs.append(
                {
                    "run_id": run_dir.name,
                    "path": str(run_dir),
                    "status": status_data.get("status", "unknown"),
                    "message": status_data.get("message", ""),
                    "language": config_data.get("language", ""),
                    "updated_at": status_data.get("updated_at", ""),
                }
            )

        return runs

    def zip_run(self, run_dir: str | Path) -> Path:
        """
        Упаковывает папку запуска в zip.

        Возвращает путь к zip-файлу.
        """

        run_dir = Path(run_dir)

        zip_path = run_dir.with_suffix(".zip")

        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as archive:
            for file_path in run_dir.rglob("*"):
                if file_path.is_file():
                    archive.write(
                        file_path,
                        file_path.relative_to(run_dir.parent),
                    )

        return zip_path