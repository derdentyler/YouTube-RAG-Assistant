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

def test_get_config_returns_expected_keys(tmp_path, monkeypatch):
    model_ru = tmp_path / "dummy_ru_model.gguf"
    reranker = tmp_path / "dummy_reranker.pkl"
    model_ru.write_text("fake")
    reranker.write_text("fake")

    cfg_file = write_config(tmp_path, {'ru': model_ru, 'reranker': reranker})
    monkeypatch.setenv('CONFIG_PATH', str(cfg_file))

    ConfigLoader._instance = None
    cfg = ConfigLoader.get_config()

    expected_keys = [
        'language', 'models', 'embedding_model', 'retriever',
        'reranker', 'subtitle_block_duration', 'subtitle_block_overlap'
    ]
    for key in expected_keys:
        assert key in cfg

def test_model_paths_exist(tmp_path, monkeypatch):
    model_ru = tmp_path / "saiga_llama3_8b-q4_k_m.gguf"
    reranker = tmp_path / "logreg_reranker.pkl"
    model_ru.write_text("fake")
    reranker.write_text("fake")

    cfg_file = write_config(tmp_path, {'ru': model_ru, 'reranker': reranker})
    monkeypatch.setenv('CONFIG_PATH', str(cfg_file))

    ConfigLoader._instance = None
    cfg = ConfigLoader.get_config()

    assert os.path.isfile(cfg['models']['ru']['model_path']), "Russian model file not found"
    assert os.path.isfile(cfg['reranker']['model_path']), "Reranker model file not found"

def test_get_config_file_not_found(monkeypatch):
    ConfigLoader._instance = None
    monkeypatch.setenv('CONFIG_PATH', 'nonexistent.yaml')
    with pytest.raises(FileNotFoundError):
        ConfigLoader.get_config()
