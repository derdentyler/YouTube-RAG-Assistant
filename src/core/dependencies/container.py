from typing import Optional

from src.utils.db_connector import DBConnector
from src.answer_generator.rag_model import RAGModel
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from sentence_transformers import SentenceTransformer
from src.core.abstractions.llm import BaseLLM
from src.answer_generator.model_factory import model_factory
from src.core.adapters.storage_factory import StorageFactory
from src.core.abstractions.storage import StorageBackend


class DependencyContainer:
    """
    Контейнер зависимостей для приложения.
    Хранит синглтон-объекты и управляет их жизненным циклом.
    """
    
    def __init__(self):
        self.logger = LoggerLoader.get_logger()
        self.config = ConfigLoader.get_config()
        
        # Тяжелые объекты - будут инициализированы один раз
        self._db_connector: Optional[DBConnector] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._llm: Optional[BaseLLM] = None
        self._rag_model: Optional[RAGModel] = None
        self._storage: Optional[StorageBackend] = None
    
    def get_db_connector(self) -> DBConnector:
        """Возвращает единственный экземпляр DBConnector."""
        if self._db_connector is None:
            self.logger.info("Initializing DBConnector...")
            self._db_connector = DBConnector(
                embedding_dimension=self.config.embedding_dimension
            )
        return self._db_connector
    
    def get_embedding_model(self) -> SentenceTransformer:
        """Возвращает единственный экземпляр embedding модели."""
        if self._embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
        return self._embedding_model
    
    def get_llm(self) -> BaseLLM:
        """Возвращает единственный экземпляр LLM."""
        if self._llm is None:
            self.logger.info(f"Loading LLM for language: {self.config.language}")
            self._llm = model_factory(self.config)
        return self._llm
    
    def get_rag_model(self) -> RAGModel:
        """Возвращает единственный экземпляр RAGModel."""
        if self._rag_model is None:
            self.logger.info("Initializing RAGModel with injected dependencies...")
            db = self.get_db_connector()
            embeddings = self.get_embedding_model()
            llm = self.get_llm()
            storage = self.get_storage()
            
            # Передаем уже загруженные модели
            self._rag_model = RAGModel(
                db_connector=db,
                embedding_model=embeddings,
                llm=llm,
                config=self.config,
                storage=storage
            )
        return self._rag_model

    def get_storage(self) -> StorageBackend:
        """Return shared storage backend for subtitles or artifacts."""
        if self._storage is None:
            self._storage = StorageFactory.create_storage()
        return self._storage
    
    def cleanup(self) -> None:
        """Освобождает ресурсы при завершении работы."""
        self.logger.info("Cleaning up dependencies...")
        if self._db_connector is not None:
            self._db_connector.close()
            self.logger.info("DBConnector closed")
        
        # Очистка памяти от моделей
        if self._llm is not None:
            del self._llm
            self.logger.info("LLM released")
        
        if self._embedding_model is not None:
            del self._embedding_model
            self.logger.info("Embedding model released")
        
        if self._rag_model is not None:
            del self._rag_model
            self.logger.info("RAGModel released")
        
        if self._storage is not None:
            del self._storage
            self._storage = None
            self.logger.info("Storage backend released")


# Глобальный экземпляр контейнера (создается один раз)
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Возвращает глобальный контейнер зависимостей."""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


def reset_container() -> None:
    """Сбрасывает контейнер (для тестов)."""
    global _container
    if _container is not None:
        _container.cleanup()
    _container = None

