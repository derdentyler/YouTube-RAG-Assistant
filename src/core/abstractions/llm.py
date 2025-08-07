from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Абстракция для LLM-модели.
    Определяет базовый интерфейс для генерации ответов и стриминга.
    """

    @abstractmethod
    def generate(self, prompt: str, max_length: int = 200) -> str:
        """
        Сгенерировать полный ответ на основе prompt.

        :param prompt: входной текстовый промпт
        :param max_length: максимальное количество токенов в ответе
        :return: сгенерированный текст
        """
        pass

    @abstractmethod
    def stream_generate(self, prompt: str, max_length: int = 200):
        """
        Построчный или по-токенный стриминг генерации.

        :param prompt: входной текстовый промпт
        :param max_length: ограничение на длину
        :yield: части ответа (строки или токены)
        """
        pass
