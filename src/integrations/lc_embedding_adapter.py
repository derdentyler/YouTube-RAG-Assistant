"""
LCEmbeddingAdapter
------------------

Адаптер для использования нативного Embedder в LangChain через Runnable API.

Тип: адаптер
Назначение: превращает любой Embedder с методом .encode() в объект, который LangChain может вызывать через .invoke() или .apply().
"""

from typing import List, Union
from langchain_core.runnables import RunnablePassthrough
from src.core.abstractions.embeddings import Embedder


class LCEmbeddingAdapter(RunnablePassthrough):
    """
    Адаптер для совместимости нативного Embedder с LangChain Runnable API.
    """

    def __init__(self, embedder: Embedder):
        """
        Инициализация адаптера.

        :param embedder: объект, реализующий протокол Embedder с методом .encode()
        """
        super().__init__()  # обязательно вызываем конструктор базового класса
        self.embedder = embedder

    def invoke(self, input_text: Union[str, List[str]], *args, **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Основной метод Runnable. LangChain вызывает его автоматически.

        :param input_text: текст или список текстов для получения эмбеддингов
        :param args: дополнительные аргументы (игнорируются)
        :param kwargs: дополнительные именованные аргументы, например convert_to_tensor
        :return: embedding или список embedding
        """
        convert_to_tensor = kwargs.get("convert_to_tensor", False)
        return self.embedder.encode(input_text, convert_to_tensor=convert_to_tensor)
