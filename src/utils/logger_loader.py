import logging
import os
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

class LoggerLoader:
    """
    Синглтон для настройки логирования.
    """
    _instance = None  # Храним единственный экземпляр

    def __new__(cls, log_level: str = "INFO"):
        if cls._instance is None:
            cls._instance = super(LoggerLoader, cls).__new__(cls)
            cls._instance._initialize(log_level)
        return cls._instance

    def _initialize(self, log_level: str):
        """Настраивает логгер при первом создании."""
        self.logger = logging.getLogger()

        if not self.logger.hasHandlers():  # Чтобы не добавлять хендлеры повторно
            self.logger.setLevel(self._get_log_level(log_level))

            # Формат логов
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # Лог в консоль
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Лог в файл (если указан)
            log_file = os.getenv("LOG_FILE", "logs/app.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    @staticmethod
    def _get_log_level(level: str):
        """Конвертирует строковый уровень логирования в logging.LEVEL."""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return levels.get(level.upper(), logging.INFO)

    @classmethod
    def get_logger(cls):
        """Возвращает синглтон-логгер."""
        if cls._instance is None:
            cls()
        return cls._instance.logger

if __name__ == "__main__":
    logger = LoggerLoader.get_logger()
    logger.info("Logger initialized successfully!")
