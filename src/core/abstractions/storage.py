from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class StorageBackend(ABC):
    """Storage backend abstraction for subtitles or other artifacts."""

    @abstractmethod
    def save_file(self, content: str, key: str) -> str:
        """Persist the content and return the storage path or URL."""

    @abstractmethod
    def load_file(self, key: str) -> Optional[str]:
        """Load a previously saved file by key."""

    @abstractmethod
    def file_exists(self, key: str) -> bool:
        """Check if a given key exists in the storage."""

    @abstractmethod
    def delete_file(self, key: str) -> bool:
        """Delete a stored file identified by its key."""

