import pytest
import os
from src.utils.config_loader import ConfigLoader


def test_get_config_returns_expected_keys(tmp_path, monkeypatch):
    # Создаем временный yaml-файл с минимальным конфигом
    config_content = """
    language: 'ru'
    embedding_model: 'test-model'
    cross_encoder_teacher: 'teacher-model'
    bi_encoder_fast: 'fast-model'
    bi_encoder_distilled: 'distilled-model'
    top_k_fast: 50
    top_k_distill: 10
    """

    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Мокаем путь к config.yaml, чтобы ConfigLoader читал наш тестовый
    monkeypatch.setenv('CONFIG_PATH', str(config_file))

    # Получаем конфиг
    cfg = ConfigLoader.get_config()

    # Проверяем, что ключи присутствуют и имеют ожидаемые значения
    expected_keys = [
        'language', 'embedding_model', 'cross_encoder_teacher',
        'bi_encoder_fast', 'bi_encoder_distilled', 'top_k_fast', 'top_k_distill'
    ]
    for key in expected_keys:
        assert key in cfg, f"Key '{key}' not found in config"
    assert cfg['language'] == 'ru'
    assert cfg['embedding_model'] == 'test-model'
    assert cfg['cross_encoder_teacher'] == 'teacher-model'
    assert cfg['bi_encoder_fast'] == 'fast-model'
    assert cfg['bi_encoder_distilled'] == 'distilled-model'
    assert cfg['top_k_fast'] == 50
    assert cfg['top_k_distill'] == 10


def test_get_config_file_not_found(monkeypatch):
    from src.utils.config_loader import ConfigLoader
    # Сброс singleton, чтобы _initialize вызвался заново
    ConfigLoader._instance = None
    # Мокаем путь к несуществующему файлу
    monkeypatch.setenv('CONFIG_PATH', 'nonexistent.yaml')
    # Ожидаем FileNotFoundError при загрузке
    with pytest.raises(FileNotFoundError):
        ConfigLoader.get_config()
