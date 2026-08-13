from __future__ import annotations


class ComponentCatalog:
    """
    КаталогКомпонентов.

    Хранит список доступных токенизаторов и моделей.
    В будущем сюда можно добавить версии компонентов, описания, пути к файлам и настройки.
    """

    def __init__(self) -> None:
        self.tokenizers = {
            "bpe": {
                "title": "BPE",
                "description": "Byte Pair Encoding — токенизация на основе частотных пар символов.",
                "versions": ["default"],
            },
            "wordpiece": {
                "title": "WordPiece",
                "description": "WordPiece — подсловная токенизация, используемая в BERT-подобных моделях.",
                "versions": ["default"],
            },
            "unigram": {
                "title": "Unigram",
                "description": "Unigram — вероятностная модель подсловной токенизации.",
                "versions": ["default"],
            },
        }

        self.models = {
            "none": {
                "title": "Без языковой модели",
                "description": "Быстрый режим: только обучение токенизаторов и расчёт метрик сжатия.",
                "versions": ["default"],
            },
            "tiny_gpt2": {
                "title": "Tiny GPT-2",
                "description": "Маленькая GPT-2-подобная модель для расчёта perplexity. Добавим позже.",
                "versions": ["default"],
            },
        }

    def get_tokenizers(self) -> list[str]:
        """
        Возвращает список доступных методов токенизации.
        """

        return list(self.tokenizers.keys())

    def get_models(self) -> list[str]:
        """
        Возвращает список доступных моделей.
        """

        return list(self.models.keys())

    def check_tokenizer(self, name: str) -> bool:
        """
        Проверяет, доступен ли токенизатор.
        """

        return name in self.tokenizers

    def check_model(self, name: str) -> bool:
        """
        Проверяет, доступна ли модель.
        """

        return name in self.models

    def get_tokenizer_info(self, name: str) -> dict:
        """
        Возвращает описание токенизатора.
        """

        return self.tokenizers[name]

    def get_model_info(self, name: str) -> dict:
        """
        Возвращает описание модели.
        """

        return self.models[name]