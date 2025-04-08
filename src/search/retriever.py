from typing import List, Tuple
from src.utils.db_connector import DBConnector
from sentence_transformers import SentenceTransformer
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader

class Retriever:
    def __init__(self, db_pool: DBConnector):
        """Инициализация Retriever с подключением к БД через пул соединений и модели эмбеддинга."""
        self.db_connector = db_pool  # Используем пул соединений
        self.embedder = SentenceTransformer(ConfigLoader.get_config()["embedding_model"])
        self.logger = LoggerLoader.get_logger()
        self.logger.info("Retriever инициализирован.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Ищет похожие субтитры по текстовому запросу.

        Args:
            query (str): Запрос, по которому ищутся похожие субтитры.
            top_k (int): Количество возвращаемых результатов (по умолчанию 5).

        Returns:
            List[Tuple[str, float]]: Список из кортежей (текст субтитров, схожесть).
        """
        try:
            # Преобразуем запрос в эмбеддинг
            query_embedding = self.embedder.encode(query).tolist()

            # Получаем похожие эмбеддинги из БД
            results = self.db_connector.search_similar_embeddings(query_embedding, top_k)

            self.logger.info(f"Найдено {len(results)} похожих субтитров для запроса: {query}")
            return results

        except Exception as error:
            self.logger.error(f"Ошибка при извлечении похожих субтитров для запроса '{query}': {error}")
            return []

    def close(self):
        """Закрыть соединение с базой данных."""
        self.db_connector.close()
