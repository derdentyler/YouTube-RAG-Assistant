import pytest

from src.answer_generator.model_factory import model_factory, TransformersLLM, LlamaCppLLM
from src.utils.logger_loader import LoggerLoader

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


def test_model_factory_llama_cpp():
    # Конфиг для backend llama.cpp
    config = {
        'language': 'ru',  # выбираем русскую модель
        'models': {
            'ru': {
                'backend': 'llama.cpp',            # тип бэкенда
                'model_path': 'path/to/model',     # путь к модели
                'n_ctx': 1024                       # контекст
            }
        }
    }
    # Вызываем фабрику
    model = model_factory(config)
    # Проверяем, что возвращён объект имеет метод generate
    assert hasattr(model, 'generate'), "Модель должна иметь метод generate"
    # Проверяем работу метода generate
    assert model.generate("test prompt") == "generated"


def test_model_factory_transformers():
    # Конфиг для backend transformers
    config = {
        'language': 'en',  # английская модель
        'models': {
            'en': {
                'backend': 'transformers',        # тип бэкенда
                'model_name': 'some/transformer'  # имя модели в HF
            }
        }
    }
    # Получение модели из фабрики
    model = model_factory(config)
    # Проверяем, что у неё есть метод generate
    assert hasattr(model, 'generate'), "Модель должна иметь метод generate"
    # И что она возвращает ожидаемое значение
    assert model.generate("another prompt", max_length=10) == "generated"


def test_model_factory_unknown_backend_raises():
    # Конфиг без корректного backend
    config = {
        'language': 'ru',
        'models': {
            'ru': {
                # backend не указан или не соответствует ожидаемым
            }
        }
    }
    # Ожидаем ValueError при неизвестном backend
    with pytest.raises(Exception):
        model_factory(config)