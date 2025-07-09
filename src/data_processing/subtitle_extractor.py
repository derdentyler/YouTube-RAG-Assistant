import os
import re
from typing import Optional, List, Dict, Union
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    Transcript,
    TranscriptList,
    FetchedTranscript
)
from transformers import AutoTokenizer

from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader

load_dotenv()


class SubtitleExtractor:
    """Извлечение и чанкинг субтитров из YouTube."""

    def __init__(self) -> None:
        self.config = ConfigLoader.get_config()
        self.logger = LoggerLoader.get_logger()
        self.api = YouTubeTranscriptApi()

        # Настройки пути и модели
        self.download_path = os.getenv("SUBTITLES_DIR", "downloads/subtitles")
        os.makedirs(self.download_path, exist_ok=True)

        model_name = self.config["embedding_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Настройки чанкинга
        self.chunk_size = int(self.config.get("chunk_size_tokens", 200))
        self.chunk_overlap = int(self.config.get("chunk_overlap_tokens", 50))

        self.logger.info("SubtitleExtractor инициализирован.")

    def extract_video_id(self, url: str) -> Optional[str]:
        """Извлекает video_id из URL."""
        self.logger.info("Извлекаем video_id из ссылки: %s", url)
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
        match = re.search(pattern, url)

        if match:
            video_id = match.group(1)
            self.logger.info("Найден video_id: %s", video_id)
            return video_id

        self.logger.error("Не удалось извлечь video_id из ссылки.")
        return None

    def get_subtitles(self, video_id: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Возвращает чанки субтитров по токенам с оверлапом."""
        self.logger.info("Загружаем субтитры для video_id: %s", video_id)
        try:
            transcripts: TranscriptList = self.api.list(video_id)
            transcript: Transcript = transcripts.find_transcript([self.config.get("language", "ru")])
            fetched: FetchedTranscript = transcript.fetch()

            # Собираем полный текст и соответствие символов к таймкодам
            full_text = ""
            char_times = []  # Один таймкод на каждый символ
            for entry in fetched:
                text = entry.text.strip()
                start = entry.start
                full_text += text + " "
                char_times.extend([start] * (len(text) + 1))

            # Токенизируем с оффсетами
            encoding = self.tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding.offset_mapping

            # Разбиваем на чанки
            chunks = []
            i = 0
            while i < len(offsets):
                end_i = min(i + self.chunk_size, len(offsets))
                char_start = offsets[i][0]
                char_end = offsets[end_i - 1][1]

                text_chunk = full_text[char_start:char_end].strip()
                start_time = char_times[char_start]
                end_time = char_times[char_end - 1]
                duration = max(end_time - start_time, 0.0)

                chunks.append({
                    "text": text_chunk,
                    "start": start_time,
                    "duration": duration
                })

                i += self.chunk_size - self.chunk_overlap

            self.logger.info(f"Чанков получено: {len(chunks)}")
            return chunks

        except TranscriptsDisabled:
            self.logger.error("Субтитры отключены для видео.")
        except NoTranscriptFound:
            self.logger.error("Субтитры не найдены для видео ID: %s", video_id)
        except Exception as e:
            self.logger.error("Ошибка при загрузке субтитров: %s", str(e))

        return None
