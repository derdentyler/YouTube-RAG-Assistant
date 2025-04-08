import os
from typing import Dict


class PromptLoader:
    def __init__(self, prompts_dir: str = "src/prompts"):
        self.prompts_dir = prompts_dir
        self.prompts_cache: Dict[str, str] = {}

    def load(self, language_code: str) -> str:
        """
        Загружает промпт-файл в зависимости от языка (например, 'ru', 'en').

        Args:
            language_code (str): Язык (например, 'ru' или 'en').

        Returns:
            str: Строка промпта с плейсхолдерами {query} и {context}.
        """
        if language_code in self.prompts_cache:
            return self.prompts_cache[language_code]

        filename = f"prompt_{language_code}.txt"
        path = os.path.join(self.prompts_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Промпт-файл не найден: {path}")

        with open(path, "r", encoding="utf-8") as file:
            prompt = file.read()

        self.prompts_cache[language_code] = prompt
        return prompt
