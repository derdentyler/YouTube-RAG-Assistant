import os
import re
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Union

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, TranscriptList
from transformers import AutoTokenizer
from yt_dlp import YoutubeDL

from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader


def _parse_vtt(path: str) -> List[Dict[str, Union[str, float]]]:
    """Парсит VTT-файл и возвращает список чанков с текстом, временем начала и длительностью."""
    from datetime import datetime

    def parse_timestamp(ts: str) -> float:
        # Обрезаем всё после первого пробела (параметры) и парсим Только HH:MM:SS.mmm
        ts_clean = ts.split()[0]
        dt = datetime.strptime(ts_clean, "%H:%M:%S.%f")
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

    chunks: List[Dict[str, Union[str, float]]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    buffer_text: List[str] = []
    start_time = 0.0
    end_time = 0.0

    for line in lines:
        if "-->" in line:
            # сохраняем предыдущий блок
            if buffer_text:
                text = " ".join(buffer_text).strip()
                duration = end_time - start_time
                chunks.append({"text": text, "start": start_time, "duration": duration})
                buffer_text = []
            # парсим времена
            parts = line.strip().split(" --> ")
            start_time = parse_timestamp(parts[0])
            end_time = parse_timestamp(parts[1])
        elif line.strip():
            buffer_text.append(line.strip())

    # последний блок
    if buffer_text:
        text = " ".join(buffer_text).strip()
        duration = end_time - start_time
        chunks.append({"text": text, "start": start_time, "duration": duration})

    return chunks


class SubtitleExtractor:
    """Извлечение и чанкинг субтитров из YouTube с надежным fallback-ом."""

    def __init__(self) -> None:
        self.config = ConfigLoader.get_config()
        self.logger = LoggerLoader.get_logger()
        self.api = YouTubeTranscriptApi()

        # Путь для сохранения субтитров
        self.download_path = os.getenv("SUBTITLES_DIR", "downloads/subtitles")
        os.makedirs(self.download_path, exist_ok=True)

        # Токенизатор для чанкинга
        model_name = self.config["embedding_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Параметры чанкинга
        self.chunk_size = int(self.config.get("chunk_size_tokens", 200))
        self.chunk_overlap = int(self.config.get("chunk_overlap_tokens", 50))

        self.logger.info("SubtitleExtractor инициализирован.")

    def extract_video_id(self, url: str) -> Optional[str]:
        """Извлекает video_id из URL и логирует его."""
        self.logger.info("Получен URL: %s", url)
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
        match = re.search(pattern, url)
        if match:
            vid = match.group(1)
            self.logger.info("Найден video_id: %s", vid)
            return vid
        self.logger.error("Не удалось извлечь video_id из ссылки: %s", url)
        return None

    def get_subtitles(self, video_id: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Возвращает чанки субтитров: ручные → авто → yt-dlp(VTT)"""
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        self.logger.info("Проверяем URL видео: %s", video_url)
        try:
            transcripts: TranscriptList = self.api.list(video_id)
            available = [(t.language, t.is_generated) for t in transcripts]
            self.logger.info("Доступные субтитры: %s", available)

            transcript = None
            # 1. Официальные
            try:
                transcript = transcripts.find_transcript([self.config.get("language", "ru")])
                self.logger.info("Найдены официальные субтитры.")
            except Exception as e:
                self.logger.warning("Официальные субтитры не найдены: %s", e)

            # 2. Авто-субтитры
            if transcript is None:
                try:
                    transcript = transcripts.find_generated_transcript([self.config.get("language", "ru")])
                    self.logger.info("Найдены автоматические субтитры.")
                except Exception as e:
                    self.logger.error("Автоматические субтитры не найдены: %s", e)
                    return None

            # Fetch через youtube_transcript_api
            try:
                entries = list(transcript.fetch())  # list of snippets
                self.logger.info("Успешно fetched %d записей субтитров", len(entries))
                self.logger.info("Пример: %s", [e.text for e in entries[:3]])
            except Exception as e:
                self.logger.error("Ошибка при fetch субтитров: %s", e)
                if transcript.is_generated:
                    self.logger.info("Пытаемся скачать субтитры через yt-dlp (VTT) как fallback.")
                    return self._get_subtitles_via_ytdlp(video_url)
                return None

            # Постобработка списка fetched
            full_text = ""
            char_times: List[float] = []
            for entry in entries:
                text = entry.text.strip()
                start = entry.start
                full_text += text + " "
                char_times.extend([start] * (len(text) + 1))

            encoding = self.tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding.offset_mapping

            chunks: List[Dict[str, Union[str, float]]] = []
            i = 0
            while i < len(offsets):
                end_i = min(i + self.chunk_size, len(offsets))
                cs = offsets[i][0]
                ce = offsets[end_i - 1][1]
                text_chunk = full_text[cs:ce].strip()
                st = char_times[cs]
                ed = char_times[ce - 1]
                dur = max(ed - st, 0.0)
                chunks.append({"text": text_chunk, "start": st, "duration": dur})
                i += self.chunk_size - self.chunk_overlap

            self.logger.info("Чанков получено: %d", len(chunks))
            return chunks

        except TranscriptsDisabled:
            self.logger.error("Субтитры отключены для видео %s", video_id)
        except NoTranscriptFound:
            self.logger.error("Субтитры не найдены для video_id: %s", video_id)
        except Exception as e:
            self.logger.error("Ошибка при загрузке субтитров: %s", e)
            return self._get_subtitles_via_ytdlp(video_url)

        return None

    def _get_subtitles_via_ytdlp(self, video_url: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Скачивает субтитры через yt-dlp API в формате VTT"""
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': [self.config.get("language", "ru")],
            'subtitlesformat': 'vtt',
            'outtmpl': os.path.join(self.download_path, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }
        try:
            self.logger.info("Запускаем yt-dlp API с опциями: %s", ydl_opts)
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)

            # Ищем скачанный .vtt файл, может быть с языковым суффиксом
            for fname in os.listdir(self.download_path):
                if fname.startswith(info.get('id')) and fname.endswith('.vtt'):
                    vtt_path = os.path.join(self.download_path, fname)
                    self.logger.info("Субтитры скачаны в: %s", vtt_path)
                    return _parse_vtt(vtt_path)

            self.logger.error("Не найден ни один .vtt для id=%s в %s", info.get('id'), self.download_path)
            return None
        except Exception as e:
            self.logger.error("Ошибка в yt-dlp fallback через Python API: %s", e)
            return None
