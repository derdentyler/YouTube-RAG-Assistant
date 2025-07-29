import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Union

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    TranscriptList,
)
from yt_dlp import YoutubeDL

from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from src.utils.subtitles_cleaner import clean_subtitles


class SubtitleExtractor:
    """
    Извлечение и подготовка субтитров из YouTube-видео.

    Пайплайн:
      1. Получение raw‑сегментов через API или VTT‑fallback.
      2. Очистка и дедупликация подряд идущих сегментов.
      3. Time‑based chunking: окна duration/overlap из конфига.
      4. Возврат списка чистых, уникальных фрагментов для RAG.
    """

    def __init__(self) -> None:
        self.config = ConfigLoader.get_config()
        self.logger = LoggerLoader.get_logger()

        # YouTube API и язык
        self.api = YouTubeTranscriptApi()
        self.language = self.config.get("language", "ru")

        # Параметры временных окон (секунды)
        self.block_duration = int(self.config.get("subtitle_block_duration", 60))
        self.block_overlap = int(self.config.get("subtitle_block_overlap", self.block_duration // 2))

        # Путь для временного хранения VTT
        self.download_path = os.getenv("SUBTITLES_DIR", "downloads/subtitles")
        os.makedirs(self.download_path, exist_ok=True)

        self.logger.info("SubtitleExtractor инициализирован.")

    def extract_video_id(self, ref: str) -> Optional[str]:
        """
        Если ref — полный URL, извлекает video_id,
        иначе возвращает ref как video_id (если формат корректен).
        """
        if ref.startswith("http"):
            m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", ref)
            return m.group(1) if m else None
        return ref if re.fullmatch(r"[0-9A-Za-z_-]{11}", ref) else None

    def fetch_subtitles_api(self, video_id: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        """
        Получает raw‑сегменты через YouTubeTranscriptApi.
        Формат: [{"text", "start", "duration"}, ...]
        """
        try:
            transcripts: TranscriptList = self.api.list(video_id)
            try:
                tr = transcripts.find_transcript([self.language])
            except:
                tr = transcripts.find_generated_transcript([self.language])
            segments = [
                {"text": e.text, "start": e.start, "duration": e.duration}
                for e in tr.fetch()
            ]
            self.logger.info(f"API fetched {len(segments)} segments")
            return segments
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception as e:
            self.logger.error(f"API error: {e}")
            return None

    def parse_vtt(self, path: str) -> List[Dict[str, Union[str, float]]]:
        """
        Парсит VTT-файл в raw сегменты без очистки.
        """
        def _ts(val: str) -> float:
            ts = val.split()[0]
            dt = datetime.strptime(ts, "%H:%M:%S.%f")
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

        raw, buffer = [], []
        start = end = 0.0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("WEBVTT", "Kind:")):
                    continue
                if "-->" in line:
                    if buffer:
                        raw.append({
                            "text": " ".join(buffer),
                            "start": start,
                            "duration": end - start
                        })
                        buffer = []
                    a, b = line.split("-->")
                    start, end = _ts(a), _ts(b)
                else:
                    buffer.append(line)
            if buffer:
                raw.append({
                    "text": " ".join(buffer),
                    "start": start,
                    "duration": end - start
                })
        self.logger.info(f"VTT parsed {len(raw)} raw segments")
        return raw

    def fetch_subtitles_vtt(self, video_id: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        """
        Fallback: скачивает VTT через yt-dlp и парсит его.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [self.language],
            "subtitlesformat": "vtt",
            "outtmpl": os.path.join(self.download_path, "%(id)s.%(ext)s"),
            "quiet": True,
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            vid = info["id"]
            fname = next(
                (f for f in os.listdir(self.download_path)
                 if f.startswith(vid) and f.endswith(".vtt")),
                None
            )
            if not fname:
                self.logger.error(f"No .vtt for {vid}")
                return None
            return self.parse_vtt(os.path.join(self.download_path, fname))
        except Exception as e:
            self.logger.error(f"yt-dlp error: {e}")
            return None

    def chunk_by_time(self, segments: List[Dict[str, Union[str, float]]]) -> List[str]:
        """
        Объединяет очищенные и дедуплицированные сегменты
        в текстовые окна по времени.
        """
        # 1) Очистка и дедупликация подрядных повторов
        cleaned = clean_subtitles(segments)
        dedup, prev = [], None
        for seg in cleaned:
            text = seg["text"]
            if text != prev:
                dedup.append(seg)
            prev = text

        if not dedup:
            return []

        # 2) Определяем временные границы
        starts = [s["start"] for s in dedup]
        t0 = starts[0]
        t_end = starts[-1] + dedup[-1]["duration"]

        # 3) Собираем окна
        windows = []
        t = t0
        while t < t_end:
            parts = [
                s["text"]
                for s in dedup
                if t <= s["start"] < t + self.block_duration
            ]
            if parts:
                windows.append(" ".join(parts))
            t += (self.block_duration - self.block_overlap)

        return windows

    def get_subtitles(self, video_ref: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        """
        Основной метод: принимает URL или video_id,
        возвращает список {"text", start=0.0, duration=0.0}.
        """
        vid = self.extract_video_id(video_ref)
        if not vid:
            self.logger.error(f"Invalid video reference: {video_ref}")
            return None

        # 1) Пытаемся через API
        segments = self.fetch_subtitles_api(vid)
        # 2) Иначе — через VTT
        if segments is None:
            segments = self.fetch_subtitles_vtt(vid)
        if not segments:
            self.logger.error(f"No subtitles for {vid}")
            return None

        # 3) Time‑based chunking
        windows = self.chunk_by_time(segments)

        # 4) Подготовка финального списка для RAG
        return [{"text": w, "start": 0.0, "duration": 0.0} for w in windows]
