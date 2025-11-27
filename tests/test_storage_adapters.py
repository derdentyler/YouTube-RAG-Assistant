import os

import pytest

from src.core.adapters.local_storage import LocalStorage
from src.core.adapters.storage_factory import StorageFactory


def test_local_storage_crud(tmp_path):
    """LocalStorage should write, read, and delete files on disk."""
    storage = LocalStorage(base_dir=str(tmp_path / "subtitles"))

    path = storage.save_file("hello world", "video1/segments.txt")
    assert os.path.exists(path)
    assert storage.file_exists("video1/segments.txt")

    content = storage.load_file("video1/segments.txt")
    assert content == "hello world"

    assert storage.delete_file("video1/segments.txt")
    assert not storage.file_exists("video1/segments.txt")


def test_storage_factory_returns_local(monkeypatch, tmp_path):
    """StorageFactory should honor STORAGE_BACKEND=local."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SUBTITLES_DIR", str(tmp_path / "incoming"))
    storage = StorageFactory.create_storage()
    assert isinstance(storage, LocalStorage)
    # Ensure the requested directory matches the environment variable
    assert str(tmp_path / "incoming") in storage.save_file("data", "test.txt")

