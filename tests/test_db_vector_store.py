import pytest
from unittest.mock import MagicMock
from src.core.adapters.db_vector_store import DBVectorStore

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def mock_embedder():
    mock = MagicMock()
    # Для encode возвращаем список numpy-подобных массивов
    mock.encode.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    return mock

@pytest.fixture
def vector_store(mock_db, mock_embedder):
    return DBVectorStore(db_connector=mock_db, embedding_model=mock_embedder)


def test_add_calls_db_insert(vector_store, mock_db, mock_embedder):
    texts = ["text1", "text2"]
    metadatas = [
        {"video_id": "vid1", "start_time": 0, "end_time": 1},
        {"video_id": "vid1", "start_time": 1, "end_time": 2}
    ]

    vector_store.add(texts, metadatas)

    # Проверяем вызов encode с правильными аргументами
    mock_embedder.encode.assert_called_once_with(texts, convert_to_tensor=False)

    # Проверяем вызовы insert_subtitle для каждого текста
    assert mock_db.insert_subtitle.call_count == 2
    mock_db.insert_subtitle.assert_any_call(
        video_id="vid1",
        start_time=0,
        end_time=1,
        text="text1",
        embedding=[0.1, 0.2, 0.3]
    )
    mock_db.insert_subtitle.assert_any_call(
        video_id="vid1",
        start_time=1,
        end_time=2,
        text="text2",
        embedding=[0.4, 0.5, 0.6]
    )


def test_search_calls_db_search(vector_store, mock_db, mock_embedder):
    mock_db.search_similar_embeddings.return_value = [
        ("some text", 0.9),
        ("another text", 0.7)
    ]

    query = "test query"
    results = vector_store.search(query, k=2)

    # Проверяем вызов encode
    mock_embedder.encode.assert_called_once_with(query, convert_to_tensor=False)

    # Проверяем вызов метода поиска в базе
    emb = mock_embedder.encode.return_value
    expected_emb = emb.tolist() if hasattr(emb, "tolist") else emb
    mock_db.search_similar_embeddings.assert_called_once_with(expected_emb, top_k=2)

    # Проверяем формат результата
    assert results == [
        {"page_content": "some text", "score": 0.9},
        {"page_content": "another text", "score": 0.7}
    ]
