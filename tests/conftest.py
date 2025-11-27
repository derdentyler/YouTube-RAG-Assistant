import pytest
import os
from unittest.mock import MagicMock, patch
from src.core.dependencies.container import DependencyContainer, reset_container
from src.utils.config_loader import ConfigLoader
from src.core.config.models import AppConfig, ModelConfigLlamaCpp


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Настройка тестового окружения перед каждым тестом."""
    # Устанавливаем переменные окружения для пропуска проверки файлов моделей
    # и предотвращения загрузки моделей
    os.environ["SKIP_MODEL_FILE_CHECK"] = "true"
    os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/sentence_transformers_cache"
    yield
    # Очищаем после теста
    for key in ["SKIP_MODEL_FILE_CHECK", "HF_HOME", "TRANSFORMERS_CACHE", 
                "SENTENCE_TRANSFORMERS_HOME"]:
        os.environ.pop(key, None)


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


@pytest.fixture(autouse=True)
def mock_s3(monkeypatch):
    """Mock boto3 S3 client to avoid real AWS calls."""
    class MockS3:
        def put_object(self, **kwargs):
            return {}

        def get_object(self, **kwargs):
            class Body:
                def read(self):
                    return b""

            return {'Body': Body()}

        def head_object(self, **kwargs):
            return {}

        def delete_object(self, **kwargs):
            return {}

    monkeypatch.setattr('boto3.client', lambda service: MockS3())
    yield

