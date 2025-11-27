from __future__ import annotations

import os

from src.core.abstractions.storage import StorageBackend
from src.core.adapters.local_storage import LocalStorage
from src.core.adapters.s3_storage import S3Storage


class StorageFactory:
    """Factory that chooses storage backend based on environment."""

    @staticmethod
    def create_storage() -> StorageBackend:
        backend = os.getenv("STORAGE_BACKEND", "local").lower()
        if backend == "s3":
            bucket = os.getenv("S3_BUCKET")
            if not bucket:
                raise ValueError("S3_BUCKET must be set for S3 storage")
            prefix = os.getenv("S3_PREFIX", "subtitles/")
            return S3Storage(bucket_name=bucket, prefix=prefix)
        if backend == "local":
            directory = os.getenv("SUBTITLES_DIR", "downloads/subtitles")
            return LocalStorage(base_dir=directory)
        raise ValueError(f"Unknown STORAGE_BACKEND={backend}")

