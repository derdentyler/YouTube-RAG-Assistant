from typing import Annotated
from fastapi import Depends
from src.utils.db_connector import DBConnector
from src.answer_generator.rag_model import RAGModel
from src.core.config.models import AppConfig
from src.utils.config_loader import ConfigLoader
from src.core.dependencies.container import get_container, DependencyContainer


def get_config() -> AppConfig:
    """Провайдер конфигурации."""
    return ConfigLoader.get_config()


def get_dependency_container() -> DependencyContainer:
    """Провайдер контейнера зависимостей."""
    return get_container()


def get_db_connector(
    container: Annotated[DependencyContainer, Depends(get_dependency_container)]
) -> DBConnector:
    """Провайдер подключения к БД."""
    return container.get_db_connector()


def get_rag_model(
    container: Annotated[DependencyContainer, Depends(get_dependency_container)]
) -> RAGModel:
    """Провайдер RAG модели."""
    return container.get_rag_model()


# Type aliases для удобства использования
ConfigDep = Annotated[AppConfig, Depends(get_config)]
DBConnectorDep = Annotated[DBConnector, Depends(get_db_connector)]
RAGModelDep = Annotated[RAGModel, Depends(get_rag_model)]

