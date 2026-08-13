from __future__ import annotations

from core.experiment_system import ExperimentSystem
from core.params import ExperimentParams
from core.result_package import ResultPackage


class ExperimentController:
    """
    КонтроллерЭксперимента.

    Этот класс является прослойкой между интерфейсом и системой экспериментов.

    Интерфейс не должен напрямую обращаться к ExperimentRun.
    Он обращается к контроллеру.
    """

    def __init__(
        self,
        system: ExperimentSystem,
    ) -> None:
        self.system = system

    def validate(
        self,
        params: ExperimentParams,
    ) -> list[str]:
        """
        Проверяет параметры эксперимента.
        """

        return self.system.validate_params(params)

    def create_and_execute(
        self,
        params: ExperimentParams,
    ) -> ResultPackage:
        """
        Создаёт запуск эксперимента и выполняет его.

        Возвращает ПакетРезультатов.
        """

        return self.system.run_experiment(params)

    def get_saved_runs(self) -> list[dict]:
        """
        Возвращает список сохранённых запусков.
        """

        return self.system.list_saved_runs()