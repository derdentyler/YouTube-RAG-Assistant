import pytest
from unittest.mock import MagicMock
import numpy as np

from src.data_processing.subtitle_manager import SubtitleManager


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    mock = MagicMock()
    # Возвращаем numpy array, чтобы поддерживался .tolist()
    mock.encode.return_value = np.array([0.1, 0.2, 0.3])
    return mock


@pytest.fixture
def subtitle_manager(mock_db, mock_embedder):
    return SubtitleManager(db_pool=mock_db, embedding_model=mock_embedder)


def test_add_subtitles(subtitle_manager, mock_db, mock_embedder):
    subtitles = [
        {"text": "Hello world", "start": 0.0, "duration": 2.5},
        {"text": "Another line", "start": 3.0, "duration": 1.5},
    ]

    subtitle_manager.add_subtitles("video123", subtitles)

    # Проверка, что encode вызывался на каждый текст
    assert mock_embedder.encode.call_count == 2

    # Проверка, что insert_subtitle вызывался с правильными аргументами
    calls = mock_db.insert_subtitle.call_args_list
    assert len(calls) == 2

    expected_args_1 = ("video123", 0.0, 2.5, "Hello world", [0.1, 0.2, 0.3])
    expected_args_2 = ("video123", 3.0, 4.5, "Another line", [0.1, 0.2, 0.3])

    assert calls[0].args == expected_args_1
    assert calls[1].args == expected_args_2


def test_get_subtitles(subtitle_manager, mock_db):
    mock_db.fetch_subtitles.return_value = [{"text": "hi"}]

    result = subtitle_manager.get_subtitles("vid456")

    mock_db.fetch_subtitles.assert_called_once_with("vid456")
    assert result == [{"text": "hi"}]


def test_clear_subtitles(subtitle_manager, mock_db):
    subtitle_manager.clear_subtitles()
    mock_db.clear_table.assert_called_once()


def test_close(subtitle_manager, mock_db):
    subtitle_manager.close()
    mock_db.close.assert_called_once()


def test_get_embedding(subtitle_manager, mock_embedder):
    result = subtitle_manager.get_embedding("test string")
    mock_embedder.encode.assert_called_once_with("test string")
    assert result == [0.1, 0.2, 0.3]
