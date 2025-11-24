import pytest
import os
from pathlib import Path
from src.utils.config_loader import ConfigLoader

def write_config(tmp_path, model_paths):
    """
    Генерируем конфиг YAML с безопасными путями для Windows.
    """
    ru_path = str(model_paths['ru']).replace("\\", "/")
    reranker_path = str(model_paths['reranker']).replace("\\", "/")

    content = f"""
    language: 'ru'
    
    models:
      ru:
        backend: 'llama.cpp'
        model_path: '{ru_path}'
        n_ctx: 8192
    
    embedding_model: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    embedding_dimension: 768
    
    retriever:
      top_k: 6
      similarity_metric: 'cosine'
    
    reranker:
      use_reranker: true
      top_k: 3
      model_path: '{reranker_path}'
    
    subtitle_block_duration: 60
    subtitle_block_overlap: 10
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(content, encoding="utf-8")
    return cfg_file

def test_get_config_returns_pydantic_model(tmp_path, monkeypatch):
    """Проверка что ConfigLoader возвращает Pydantic модель."""
    model_ru = tmp_path / "dummy_ru_model.gguf"
    reranker = tmp_path / "dummy_reranker.pkl"
    model_ru.write_text("fake")
    reranker.write_text("fake")

    cfg_file = write_config(tmp_path, {'ru': model_ru, 'reranker': reranker})
    monkeypatch.setenv('CONFIG_PATH', str(cfg_file))

    ConfigLoader._instance = None
    cfg = ConfigLoader.get_config()

    # Проверка что это Pydantic модель
    from src.core.config.models import AppConfig
    assert isinstance(cfg, AppConfig)
    
    # Проверка доступа к полям
    assert cfg.language == "ru"
    assert cfg.retriever.top_k == 6
    assert cfg.reranker.use_reranker == True
    assert cfg.embedding_dimension == 768

def test_model_paths_exist(tmp_path, monkeypatch):
    model_ru = tmp_path / "saiga_llama3_8b-q4_k_m.gguf"
    reranker = tmp_path / "logreg_reranker.pkl"
    model_ru.write_text("fake")
    reranker.write_text("fake")

    cfg_file = write_config(tmp_path, {'ru': model_ru, 'reranker': reranker})
    monkeypatch.setenv('CONFIG_PATH', str(cfg_file))

    ConfigLoader._instance = None
    cfg = ConfigLoader.get_config()

    assert os.path.isfile(cfg.models['ru'].model_path), "Russian model file not found"
    assert os.path.isfile(cfg.reranker.model_path), "Reranker model file not found"

def test_get_config_file_not_found(monkeypatch):
    ConfigLoader._instance = None
    monkeypatch.setenv('CONFIG_PATH', 'nonexistent.yaml')
    with pytest.raises(FileNotFoundError):
        ConfigLoader.get_config()


def test_invalid_config_raises_validation_error(tmp_path, monkeypatch):
    """Проверка что невалидная конфигурация вызывает ошибку."""
    content = """
    language: 'ru'
    models:
      ru:
        backend: 'llama.cpp'
        # model_path отсутствует - должна быть ошибка
    embedding_model: 'test-model'
    embedding_dimension: 768
    """
    cfg_file = tmp_path / "invalid_config.yaml"
    cfg_file.write_text(content, encoding="utf-8")
    monkeypatch.setenv('CONFIG_PATH', str(cfg_file))

    ConfigLoader._instance = None
    with pytest.raises(ValueError, match="Invalid configuration"):
        ConfigLoader.get_config()


def test_overlap_greater_than_duration_raises_error(tmp_path, monkeypatch):
    """Проверка валидации overlap < duration."""
    model_ru = tmp_path / "dummy_ru_model.gguf"
    model_ru.write_text("fake")
    
    content = f"""
    language: 'ru'
    models:
      ru:
        backend: 'llama.cpp'
        model_path: '{str(model_ru).replace(chr(92), '/')}'
    embedding_model: 'test-model'
    embedding_dimension: 768
    subtitle_block_duration: 60
    subtitle_block_overlap: 70  # Больше чем duration!
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(content, encoding="utf-8")
    monkeypatch.setenv('CONFIG_PATH', str(cfg_file))

    ConfigLoader._instance = None
    with pytest.raises(ValueError, match="overlap.*must be less than.*duration"):
        ConfigLoader.get_config()


def test_missing_language_model_raises_error(tmp_path, monkeypatch):
    """Проверка валидации наличия модели для выбранного языка."""
    model_ru = tmp_path / "dummy_ru_model.gguf"
    model_ru.write_text("fake")
    
    content = f"""
    language: 'en'  # Выбран английский, но модели для него нет!
    models:
      ru:
        backend: 'llama.cpp'
        model_path: '{str(model_ru).replace(chr(92), '/')}'
    embedding_model: 'test-model'
    embedding_dimension: 768
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(content, encoding="utf-8")
    monkeypatch.setenv('CONFIG_PATH', str(cfg_file))

    ConfigLoader._instance = None
    with pytest.raises(ValueError, match="No model configured for language"):
        ConfigLoader.get_config()
