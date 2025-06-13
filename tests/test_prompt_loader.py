import pytest
from pathlib import Path
from src.utils.prompt_loader import PromptLoader

def test_prompt_loader_load_existing(tmp_path, monkeypatch):
    """
    Проверяем, что PromptLoader.load возвращает содержимое файла с шаблоном,
    и что в шаблоне присутствуют плейсхолдеры {query} и {context}.
    """
    # 1) Создаём фейковую папку prompts/ru с файлом prompt.txt
    prompts_dir = tmp_path / "prompts" / "ru"
    prompts_dir.mkdir(parents=True)
    prompt_file = prompts_dir / "prompt.txt"
    prompt_file.write_text("Вопрос: {query}\nКонтекст: {context}")

    # 2) Подменяем константу пути в модуле prompt_loader
    monkeypatch.setenv("PROMPTS_PATH", str(tmp_path / "prompts"))

    loader = PromptLoader()
    template = loader.load("ru")

    # 3) Убедимся, что шаблон прочитан полностью и содержит нужные плейсхолдеры
    assert isinstance(template, str)
    assert "{query}" in template
    assert "{context}" in template

def test_prompt_loader_missing_language(tmp_path, monkeypatch):
    """
    Проверяем, что при отсутствии папки/файла для языка бросается FileNotFoundError.
    """
    # Указываем пустую папку prompts
    empty_prompts = tmp_path / "empty_prompts"
    empty_prompts.mkdir()
    monkeypatch.setenv("PROMPTS_PATH", str(empty_prompts))

    loader = PromptLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("xx")  # языка xx нет в пустой папке
