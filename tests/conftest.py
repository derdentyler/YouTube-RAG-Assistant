import pytest
from unittest.mock import MagicMock
from src.core.dependencies.container import DependencyContainer, reset_container
from src.utils.config_loader import ConfigLoader
from src.core.config.models import AppConfig, ModelConfigLlamaCpp


@pytest.fixture(autouse=True)
def cleanup_container():
    """Автоматическая очистка контейнера после каждого теста."""
    yield
    reset_container()


@pytest.fixture
def mock_container():
    """Мок контейнера зависимостей для тестов."""
    container = DependencyContainer()
    
    # Мокаем тяжелые объекты
    container._db_connector = MagicMock()
    container._embedding_model = MagicMock()
    container._llm = MagicMock()
    container._rag_model = MagicMock()
    
    return container


@pytest.fixture
def test_config(tmp_path):
    """Конфигурация для тестов."""
    model_file = tmp_path / "test_model.gguf"
    model_file.write_text("fake")
    
    return AppConfig(
        language='ru',
        models={
            'ru': ModelConfigLlamaCpp(
                backend='llama.cpp',
                model_path=str(model_file),
                n_ctx=1024
            )
        },
        embedding_model='test-model',
        embedding_dimension=768
    )

