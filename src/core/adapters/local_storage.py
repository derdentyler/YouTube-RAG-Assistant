from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.core.abstractions.storage import StorageBackend
from src.utils.logger_loader import LoggerLoader


class LocalStorage(StorageBackend):
    """Stores files under a configurable directory on the local filesystem."""

    def __init__(self, base_dir: str = "downloads/subtitles") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = LoggerLoader.get_logger()

    def save_file(self, content: str, key: str) -> str:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.logger.info("Saved file locally at %s", path)
        return str(path)

    def load_file(self, key: str) -> Optional[str]:
        path = self.base_dir / key
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def file_exists(self, key: str) -> bool:
        return (self.base_dir / key).exists()

    def delete_file(self, key: str) -> bool:
        path = self.base_dir / key
        if path.exists():
            path.unlink()
            self.logger.info("Deleted local file %s", path)
            return True
        return False

