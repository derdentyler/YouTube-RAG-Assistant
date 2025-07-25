import os
import re
from typing import List, Dict, Optional, Union
from datetime import datetime

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, TranscriptList
from transformers import AutoTokenizer
from yt_dlp import YoutubeDL

from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from src.utils.subtitles_cleaner import clean_subtitles


class SubtitleExtractor:
    def __init__(self) -> None:
        self.config = ConfigLoader.get_config()
        self.logger = LoggerLoader.get_logger()

        self.api = YouTubeTranscriptApi()
        self.language = self.config.get("language", "ru")
        self.chunk_size = int(self.config.get("chunk_size_tokens", 200))
        self.chunk_overlap = int(self.config.get("chunk_overlap_tokens", 50))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["embedding_model"],
            use_fast=True
        )

        self.download_path = os.getenv("SUBTITLES_DIR", "downloads/subtitles")
        os.makedirs(self.download_path, exist_ok=True)

    def extract_video_id(self, url: str) -> Optional[str]:
        match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
        return match.group(1) if match else None

    def fetch_subtitles_api(self, video_id: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        try:
            transcripts: TranscriptList = self.api.list(video_id)
            tr = None
            try:
                tr = transcripts.find_transcript([self.language])
            except:
                tr = transcripts.find_generated_transcript([self.language])
            if tr:
                return [{"text": e.text, "start": e.start, "duration": e.duration} for e in tr.fetch()]
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception as e:
            self.logger.error("API error: %s", e)
            return None

    def parse_vtt(self, path: str) -> List[Dict[str, Union[str, float]]]:
        def parse_ts(ts: str) -> float:
            ts0 = ts.split()[0]
            dt = datetime.strptime(ts0, "%H:%M:%S.%f")
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

        raw, buffer = [], []
        start, end = 0.0, 0.0

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
                    start, end = parse_ts(a), parse_ts(b)
                else:
                    buffer.append(line)
            if buffer:
                raw.append({
                    "text": " ".join(buffer),
                    "start": start,
                    "duration": end - start
                })
        return raw

    def fetch_subtitles_vtt(self, video_id: str) -> Optional[List[Dict[str, Union[str, float]]]]:
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
            fname = next((f for f in os.listdir(self.download_path)
                          if f.startswith(info["id"]) and f.endswith(".vtt")), None)
            if fname:
                return self.parse_vtt(os.path.join(self.download_path, fname))
        except Exception as e:
            self.logger.error("yt-dlp error: %s", e)
        return None

    def chunk_text(self, segments: List[Dict[str, Union[str, float]]]) -> List[Dict[str, Union[str, float]]]:
        cleaned = clean_subtitles(segments)

        full_text = ""
        char_times = []
        for seg in cleaned:
            text = seg["text"]
            full_text += text + " "
            char_times.extend([seg["start"]] * (len(text) + 1))

        offsets = self.tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False).offset_mapping
        chunks = []
        i = 0
        while i < len(offsets):
            j = min(i + self.chunk_size, len(offsets))
            cs, ce = offsets[i][0], offsets[j - 1][1]
            chunk_text = full_text[cs:ce].strip()
            if not chunk_text:
                i += self.chunk_size - self.chunk_overlap
                continue
            chunks.append({
                "text": chunk_text,
                "start": char_times[cs],
                "duration": max(char_times[ce - 1] - char_times[cs], 0.0)
            })
            i += self.chunk_size - self.chunk_overlap

        seen = set()
        unique = []
        for ch in chunks:
            if ch["text"] not in seen:
                seen.add(ch["text"])
                unique.append(ch)
        return unique

    def get_subtitles(self, video_id: str) -> Optional[List[Dict[str, Union[str, float]]]]:
        entries = self.fetch_subtitles_api(video_id)
        if not entries:
            entries = self.fetch_subtitles_vtt(video_id)
        if not entries:
            return None
        return self.chunk_text(entries)
