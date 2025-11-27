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
from src.core.abstractions.embeddings import Embedder
from src.core.abstractions.storage import StorageBackend


class SubtitleExtractor:
    """
    Извлечение и подготовка субтитров из YouTube-видео.

    Пайплайн:
      1. Получение raw‑сегментов через API или VTT‑fallback.
      2. Очистка и дедупликация подряд идущих сегментов.
      3. Chunking: semantic (семантический) или time (временной) из конфига.
      4. Возврат списка чистых, уникальных фрагментов для RAG.
    """

    def __init__(
        self,
        embedding_model: Optional[Embedder] = None,
        storage: Optional[StorageBackend] = None,
    ) -> None:
        self.config = ConfigLoader.get_config()
        self.logger = LoggerLoader.get_logger()
        self.storage = storage

        # YouTube API и язык
        self.api = YouTubeTranscriptApi()
        self.language = self.config.language

        # Параметры временных окон (секунды) - для обратной совместимости
        self.block_duration = self.config.subtitle_block_duration
        self.block_overlap = self.config.subtitle_block_overlap

        # Настройки chunking
        self.chunking_method = self.config.chunking.method
        self.embedding_model = embedding_model

        # Инициализация SemanticChunker если нужен
        self.semantic_chunker = None
        if self.chunking_method == "semantic":
            if embedding_model is None:
                self.logger.warning(
                    "Semantic chunking requires embedding_model. "
                    "Falling back to time-based chunking."
                )
                self.chunking_method = "time"
            else:
                from src.data_processing.semantic_chunker import SemanticChunker
                self.semantic_chunker = SemanticChunker(
                    embedding_model=embedding_model,
                    max_tokens=self.config.chunking.max_tokens,
                    similarity_threshold=self.config.chunking.similarity_threshold,
                    min_chunk_size=self.config.chunking.min_chunk_size
                )
                self.logger.info(
                    f"Semantic chunking initialized: "
                    f"max_tokens={self.config.chunking.max_tokens}, "
                    f"similarity_threshold={self.config.chunking.similarity_threshold}"
                )

        # Путь для временного хранения VTT
        self.download_path = os.getenv("SUBTITLES_DIR", "downloads/subtitles")
        os.makedirs(self.download_path, exist_ok=True)

        self.logger.info(
            f"SubtitleExtractor инициализирован с методом chunking: {self.chunking_method}"
        )

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
        возвращает список {"text", start, duration} с временными метками.
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

        self._persist_segments(vid, segments)

        # 3) Chunking в зависимости от метода
        if self.chunking_method == "semantic" and self.semantic_chunker:
            # Semantic chunking с сохранением временных меток
            chunks = self.semantic_chunker.chunk(segments)
            return chunks
        else:
            # Time-based chunking (обратная совместимость)
            windows = self.chunk_by_time(segments)
            # Для time-based chunking временные метки теряются,
            # возвращаем с нулевыми значениями
            return [{"text": w, "start": 0.0, "duration": 0.0} for w in windows]

    def _persist_segments(self, video_id: str, segments: List[Dict[str, Union[str, float]]]) -> None:
        """Optionally backup segments to the configured storage backend."""
        if not self.storage or not segments:
            return
        lines = [
            f"{seg['start']}|{seg['duration']}|{seg['text']}"
            for seg in segments
        ]
        payload = "\n".join(lines)
        key = f"{video_id}/segments.txt"
        try:
            self.storage.save_file(payload, key)
        except Exception as exc:
            self.logger.warning("Failed to persist segments for %s: %s", video_id, exc)
