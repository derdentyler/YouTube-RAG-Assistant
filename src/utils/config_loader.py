import yaml
import os
from src.core.config.models import AppConfig


class ConfigLoader:
    """
    Синглтон для загрузки и валидации конфигурации из YAML-файла.
    """
    _instance = None
    
    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialize(config_path)
        return cls._instance
    
    def _initialize(self, config_path: str):
        """Загружает и валидирует конфиг при первом создании экземпляра."""
        if not config_path:
            config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
        
        # Валидация через Pydantic
        try:
            self.config = AppConfig(**raw_config)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}") from e
    
    @classmethod
    def get_config(cls) -> AppConfig:
        """Возвращает загруженный и валидированный конфиг."""
        if cls._instance is None:
            cls()
        return cls._instance.config

if __name__ == "__main__":
    config = ConfigLoader.get_config()
    print(config)  # Выведет конфиг
