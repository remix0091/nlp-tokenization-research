from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ResultPackage:
    """
    ПакетРезультатов.

    Этот класс хранит итог одного эксперимента:
    - метрики;
    - таблицы;
    - графики;
    - отчёт;
    - логи;
    - путь к папке сохранения;
    - ошибки.
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)

    report: str = ""
    logs: str = ""
    save_path: str = ""

    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Преобразует пакет результатов в словарь.

        Потом этот словарь будем сохранять в result_package.json.
        """

        return asdict(self)