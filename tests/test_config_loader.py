import pytest
import os
from src.utils.config_loader import ConfigLoader

def write_config(tmp_path, model_paths):
    # Формируем YAML со всеми нужными путями
    content = f"""
    language: "ru"
    
    models:
      ru:
        backend: "llama.cpp"
        model_path: "{model_paths['ru']}"
        n_ctx: 8192
      en:
        backend: "transformers"
        model_name: "mistralai/Mistral-7B-Instruct-v0.1"
    
    embedding_model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    retriever:
      top_k: 6
      similarity_metric: "cosine"
    
    reranker:
      use_reranker: true
      top_k: 3
      model_path: "{model_paths['reranker']}"
    
    subtitle_block_duration: 60
    subtitle_block_overlap: 10
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(content)
    return cfg_file

def test_get_config_returns_expected_keys(tmp_path, monkeypatch):
    # Создаём минимальный конфиг без проверки путей
    cfg_file = write_config(tmp_path, {
        'ru': '/tmp/nonexistent1.gguf',
        'reranker': '/tmp/nonexistent2.pkl'
    })
    monkeypatch.setenv('CONFIG_PATH', str(cfg_file))
    # Сброс синглетона
    ConfigLoader._instance = None
    cfg = ConfigLoader.get_config()
    # Проверяем ключи
    for key in ['language','models','embedding_model','retriever','reranker','subtitle_block_duration','subtitle_block_overlap']:
        assert key in cfg

def test_model_paths_exist(tmp_path, monkeypatch):
    # создаём фиктивные файлы-модели
    model_ru = tmp_path / "saiga_llama3_8b-q4_k_m.gguf"
    reranker = tmp_path / "logreg_reranker.pkl"
    model_ru.write_text("fake")
    reranker.write_text("fake")
    # пишем конфиг с абсолютными путями
    cfg_file = write_config(tmp_path, {
        'ru': str(model_ru),
        'reranker': str(reranker)
    })
    monkeypatch.setenv('CONFIG_PATH', str(cfg_file))
    ConfigLoader._instance = None
    cfg = ConfigLoader.get_config()
    # Проверяем, что файлы действительно есть
    assert os.path.isfile(cfg['models']['ru']['model_path']), "Russian model file not found"
    assert os.path.isfile(cfg['reranker']['model_path']), "Reranker model file not found"

def test_get_config_file_not_found(monkeypatch):
    ConfigLoader._instance = None
    monkeypatch.setenv('CONFIG_PATH', 'nonexistent.yaml')
    with pytest.raises(FileNotFoundError):
        ConfigLoader.get_config()
