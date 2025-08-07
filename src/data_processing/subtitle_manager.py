from src.utils.db_connector import DBConnector
from src.utils.config_loader import ConfigLoader
from typing import Optional, List, Dict, Union
from src.core.abstractions.embeddings import Embedder

class SubtitleManager:
    def __init__(self, db_pool: DBConnector, embedding_model: Embedder):
        """
        Инициализация менеджера субтитров с подключением к базе данных через пул соединений.
        """
        self.db_connector = db_pool  # Используем пул соединений
        # Загружаем конфигурацию
        self.config = ConfigLoader.get_config()
        # Инициализация модели для получения эмбеддингов
        self.embedding_model = embedding_model
        # self.embedding_model = SentenceTransformer(self.config["embedding_model"])

    def add_subtitles(self, video_id: str, subtitles: List[Dict[str, Union[str, float]]]):
        """Добавление субтитров в базу данных."""
        for subtitle in subtitles:
            text = subtitle["text"]
            start_time = subtitle["start"]
            end_time = start_time + subtitle["duration"]

            # Получаем эмбеддинг для текста субтитра
            embedding = self.get_embedding(text)

            # Добавляем субтитры в базу
            self.db_connector.insert_subtitle(video_id, start_time, end_time, text, embedding)

    def get_subtitles(self, video_id: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Получение субтитров по video_id."""
        return self.db_connector.fetch_subtitles(video_id)

    def clear_subtitles(self):
        """Очистка таблицы субтитров."""
        self.db_connector.clear_table()

    def close(self):
        """Закрыть соединение с базой данных."""
        self.db_connector.close()

    def get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддинга для текста."""
        # Преобразуем текст в эмбеддинг с помощью модели
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()  # Преобразуем в список для сохранения в базе
