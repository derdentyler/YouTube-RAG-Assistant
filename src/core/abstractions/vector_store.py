from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    """
    Абстракция для векторного хранилища.
    Позволяет добавлять документы и выполнять поиск по embedding.
    """

    @abstractmethod
    def add(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Добавить список документов в векторное хранилище.

        :param texts: список текстов документов
        :param metadatas: список метаданных для каждого документа
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Найти top-k документов по семантическому сходству к запросу.

        :param query: поисковый запрос
        :param k: число возвращаемых документов
        :return: список словарей вида {"page_content": str, "score": float, ...}
        """
        pass
