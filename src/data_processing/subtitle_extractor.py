import os
import re
from dotenv import load_dotenv
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, Transcript, TranscriptList, FetchedTranscript
from typing import Optional, List, Dict, Union


load_dotenv()  # Загружаем переменные окружения


class SubtitleExtractor:
    """Класс для работы с субтитрами YouTube."""

    def __init__(self) -> None:
        """Инициализирует конфиг, логгер и пути."""
        self.config = ConfigLoader.get_config()
        self.language: str = self.config.get("language", "en")  # Язык субтитров
        self.download_path: str = os.getenv("SUBTITLE_DOWNLOAD_PATH", "downloads/subtitles")  # Путь сохранения субтитров
        self.logger = LoggerLoader.get_logger()
        self.api = YouTubeTranscriptApi()

        os.makedirs(self.download_path, exist_ok=True)  # Создаем папку, если её нет
        self.logger.info("SubtitleExtractor инициализирован.")

    def extract_video_id(self, url: str) -> Optional[str]:
        """Извлекает video_id из ссылки на YouTube.

        Args:
            url (str): Ссылка на видео.

        Returns:
            Optional[str]: video_id или None, если не удалось извлечь.
        """
        self.logger.info("Извлекаем video_id из ссылки: %s", url)

        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"  # Регулярка для извлечения video_id
        match: Optional[re.Match] = re.search(pattern, url)  # Аннотация Optional[re.Match]

        if match:
            video_id: str = match.group(1)
            self.logger.info("Найден video_id: %s", video_id)
            return video_id

        self.logger.error("Ошибка: не удалось извлечь video_id из ссылки.")
        return None

    def get_subtitles(self, video_id: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Загружает и объединяет субтитры в блоки нужной длительности.

        Args:
            video_id (str): ID YouTube-видео.

        Returns:
            Optional[List[Dict[str, Union[str, float]]]]: Список объединенных субтитров.
        """
        self.logger.info("Загружаем субтитры для video_id: %s", video_id)
        try:
            transcripts: TranscriptList = self.api.list(video_id)
            transcript: Transcript = transcripts.find_transcript(['ru'])
            fetched: FetchedTranscript = transcript.fetch()

            raw_subs = [
                {"text": entry.text, "start": entry.start, "duration": entry.duration}
                for entry in fetched
            ]

            block_duration = int(self.config.get("subtitle_block_duration", 60))
            self.logger.info(f"Будем объединять субтитры в блоки по {block_duration} секунд.")

            merged_subs: List[Dict[str, Union[str, float]]] = []
            current_block = {"text": "", "start": None, "duration": 0.0}

            for entry in raw_subs:
                start, duration, text = entry["start"], entry["duration"], entry["text"]

                if current_block["start"] is None:
                    current_block["start"] = start

                current_block["text"] += " " + text
                current_block["duration"] += duration

                # Сохраняем блок, если достигли лимита по времени
                if current_block["duration"] >= block_duration:
                    merged_subs.append(current_block.copy())
                    current_block = {"text": "", "start": None, "duration": 0.0}

            # Добавим остатки, если есть
            if current_block["text"]:
                merged_subs.append(current_block)

            self.logger.info(f"Всего блоков субтитров после объединения: {len(merged_subs)}")
            return merged_subs

        except TranscriptsDisabled:
            self.logger.error("Субтитры отключены для этого видео.")
        except NoTranscriptFound:
            self.logger.error("Субтитры не найдены для видео ID: %s", video_id)
        except Exception as e:
            self.logger.error("Ошибка при загрузке субтитров: %s", str(e))

        return None
