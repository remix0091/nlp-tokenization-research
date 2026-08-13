from __future__ import annotations

from core.component_catalog import ComponentCatalog
from core.experiment_run import ExperimentRun
from core.params import ExperimentParams
from core.result_package import ResultPackage
from core.storage_manager import StorageManager


class ExperimentSystem:
    """
    СистемаЭкспериментов.

    Этот класс отвечает за:
    - проверку корректности параметров;
    - создание запуска эксперимента;
    - запуск выполнения;
    - возврат пакета результатов.
    """

    def __init__(
        self,
        storage: StorageManager,
        catalog: ComponentCatalog,
    ) -> None:
        self.storage = storage
        self.catalog = catalog

    def validate_params(
        self,
        params: ExperimentParams,
    ) -> list[str]:
        """
        Проверяет корректность параметров эксперимента.

        Возвращает список ошибок.
        Если список пустой, параметры корректны.
        """

        return params.validate(self.catalog)

    def create_run(
        self,
        params: ExperimentParams,
    ) -> ExperimentRun:
        """
        Создаёт объект запуска эксперимента.

        Если параметры некорректны, выбрасывает ValueError.
        """

        validation_errors = self.validate_params(params)

        if validation_errors:
            message = "; ".join(validation_errors)
            raise ValueError(message)

        experiment_run = ExperimentRun(
            params=params,
            storage=self.storage,
        )

        return experiment_run

    def execute_run(
        self,
        experiment_run: ExperimentRun,
    ) -> ResultPackage:
        """
        Выполняет созданный запуск эксперимента.
        """

        return experiment_run.execute()

    def run_experiment(
        self,
        params: ExperimentParams,
    ) -> ResultPackage:
        """
        Создаёт и сразу выполняет эксперимент.

        Это удобный метод для интерфейса.
        """

        experiment_run = self.create_run(params)

        result_package = self.execute_run(experiment_run)

        return result_package

    def list_saved_runs(self) -> list[dict]:
        """
        Возвращает список сохранённых запусков из хранилища.
        """

        return self.storage.list_runs()