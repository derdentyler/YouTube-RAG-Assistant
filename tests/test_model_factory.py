import pytest
import os
from pathlib import Path

from src.answer_generator.model_factory import model_factory, TransformersLLM, LlamaCppLLM
from src.utils.logger_loader import LoggerLoader
from src.core.config.models import AppConfig, ModelConfigLlamaCpp, ModelConfigTransformers

# Заглушка для генерации текстов
class DummyLLM:
    def __init__(self, *args, **kwargs):
        pass
    def generate(self, prompt, max_length=None):
        # Всегда возвращает эту строку для теста
        return "generated"

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Подменяем класс TransformersLLM внутри фабрики
    monkeypatch.setattr(
        'src.answer_generator.model_factory.TransformersLLM',
        lambda model_name: DummyLLM()
    )
    # Подменяем класс LlamaCppLLM внутри фабрики
    monkeypatch.setattr(
        'src.answer_generator.model_factory.LlamaCppLLM',
        lambda model_path, n_ctx=2048: DummyLLM()
    )
    # Подменяем LoggerLoader, чтобы не писать в лог
    monkeypatch.setattr(LoggerLoader, 'get_logger', lambda: None)


def test_model_factory_llama_cpp(tmp_path):
    # Создаем временный файл модели
    model_file = tmp_path / "test_model.gguf"
    model_file.write_text("fake")
    
    # Конфиг для backend llama.cpp
    config = AppConfig(
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
    # Вызываем фабрику
    model = model_factory(config)
    # Проверяем, что возвращён объект имеет метод generate
    assert hasattr(model, 'generate'), "Модель должна иметь метод generate"
    # Проверяем работу метода generate
    assert model.generate("test prompt") == "generated"


def test_model_factory_transformers():
    # Конфиг для backend transformers
    config = AppConfig(
        language='en',
        models={
            'en': ModelConfigTransformers(
                backend='transformers',
                model_name='some/transformer'
            )
        },
        embedding_model='test-model',
        embedding_dimension=768
    )
    # Получение модели из фабрики
    model = model_factory(config)
    # Проверяем, что у неё есть метод generate
    assert hasattr(model, 'generate'), "Модель должна иметь метод generate"
    # И что она возвращает ожидаемое значение
    assert model.generate("another prompt", max_length=10) == "generated"


def test_model_factory_unknown_backend_raises(tmp_path):
    # Создаем временный файл модели
    model_file = tmp_path / "test_model.gguf"
    model_file.write_text("fake")
    
    # Конфиг с невалидным backend (Pydantic не позволит создать такой конфиг)
    # Но можно попробовать создать с валидным backend и затем изменить его
    config = AppConfig(
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
    # Меняем backend на невалидный (это вызовет ошибку в фабрике)
    config.models['ru'].backend = "unknown_backend"  # type: ignore
    # Ожидаем ValueError при неизвестном backend
    with pytest.raises(ValueError, match="Unknown backend"):
        model_factory(config)