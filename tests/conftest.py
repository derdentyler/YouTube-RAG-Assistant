import pytest
import os
from unittest.mock import MagicMock, patch
from src.core.dependencies.container import DependencyContainer, reset_container
from src.utils.config_loader import ConfigLoader
from src.core.config.models import AppConfig, ModelConfigLlamaCpp


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Настройка тестового окружения перед каждым тестом."""
    # Устанавливаем переменную окружения для пропуска проверки файлов моделей
    os.environ["SKIP_MODEL_FILE_CHECK"] = "true"
    yield
    # Очищаем после теста
    os.environ.pop("SKIP_MODEL_FILE_CHECK", None)


@pytest.fixture(autouse=True)
def cleanup_container():
    """Автоматическая очистка контейнера после каждого теста."""
    # Сбрасываем синглтон ConfigLoader перед каждым тестом
    ConfigLoader._instance = None
    yield
    reset_container()
    # Сбрасываем синглтон ConfigLoader после каждого теста
    ConfigLoader._instance = None


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

