import yaml
import os

class ConfigLoader:
    """
    Синглтон для загрузки конфигурации из YAML-файла.
    """
    _instance = None  # Храним единственный экземпляр

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialize(config_path)
        return cls._instance

    def _initialize(self, config_path: str):
        """Загружает конфиг при первом создании экземпляра."""
        if not config_path:
            config_path = os.getenv("CONFIG_PATH", "config/config.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    @classmethod
    def get_config(cls):
        """Возвращает загруженный конфиг."""
        if cls._instance is None:
            cls()
        return cls._instance.config

if __name__ == "__main__":
    config = ConfigLoader.get_config()
    print(config)  # Выведет конфиг
