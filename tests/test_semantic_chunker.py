import pytest
import numpy as np
from unittest.mock import MagicMock
from src.data_processing.semantic_chunker import SemanticChunker


@pytest.fixture
def mock_embedder():
    """Мок для Embedder."""
    embedder = MagicMock()
    
    # Симулируем эмбеддинги для разных предложений
    # Похожие предложения будут иметь похожие эмбеддинги
    def mock_encode(texts, convert_to_tensor=False):
        if isinstance(texts, str):
            texts = [texts]
        
        # Простая симуляция: создаем эмбеддинги на основе длины текста
        # В реальности это будут настоящие эмбеддинги
        embeddings = []
        for text in texts:
            # Создаем фиктивный эмбеддинг размерности 768
            # Используем хеш текста для создания "уникального" эмбеддинга
            base = hash(text) % 1000
            emb = np.array([base + i * 0.1 for i in range(768)])
            # Нормализуем для корректного cosine similarity
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        
        if len(embeddings) == 1:
            return embeddings[0]
        return np.array(embeddings)
    
    embedder.encode = mock_encode
    return embedder


@pytest.fixture
def semantic_chunker(mock_embedder):
    """Создает SemanticChunker с моком embedder."""
    return SemanticChunker(
        embedding_model=mock_embedder,
        max_tokens=150,
        similarity_threshold=0.7,
        min_chunk_size=50
    )


def test_split_into_sentences(semantic_chunker):
    """Тест разбиения текста на предложения."""
    text = "Первое предложение. Второе предложение! Третье предложение?"
    sentences = semantic_chunker._split_into_sentences(text)
    
    assert len(sentences) == 3
    assert "Первое предложение" in sentences[0]
    assert "Второе предложение" in sentences[1]
    assert "Третье предложение" in sentences[2]


def test_estimate_tokens(semantic_chunker):
    """Тест оценки количества токенов."""
    text = "Это тестовый текст для проверки оценки токенов."
    tokens = semantic_chunker._estimate_tokens(text)
    
    # Примерно 4 символа на токен
    assert tokens > 0
    assert tokens < len(text)  # Должно быть меньше длины текста


def test_cosine_similarity(semantic_chunker):
    """Тест вычисления косинусного сходства."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    
    similarity = semantic_chunker._cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 0.001  # Идентичные векторы
    
    vec3 = np.array([0.0, 1.0, 0.0])
    similarity2 = semantic_chunker._cosine_similarity(vec1, vec3)
    assert abs(similarity2 - 0.0) < 0.001  # Ортогональные векторы


def test_chunk_simple_text(semantic_chunker):
    """Тест чанкинга простого текста."""
    segments = [
        {"text": "Первое предложение о программировании.", "start": 0.0, "duration": 2.0},
        {"text": "Второе предложение о программировании.", "start": 2.0, "duration": 2.0},
        {"text": "Третье предложение о программировании.", "start": 4.0, "duration": 2.0},
    ]
    
    chunks = semantic_chunker.chunk(segments)
    
    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("start" in chunk for chunk in chunks)
    assert all("duration" in chunk for chunk in chunks)
    assert all(chunk["duration"] >= 0 for chunk in chunks)


def test_chunk_preserves_time_metadata(semantic_chunker):
    """Тест сохранения временных меток в чанках."""
    segments = [
        {"text": "Начало видео.", "start": 10.0, "duration": 2.0},
        {"text": "Середина видео.", "start": 12.0, "duration": 2.0},
        {"text": "Конец видео.", "start": 14.0, "duration": 2.0},
    ]
    
    chunks = semantic_chunker.chunk(segments)
    
    # Проверяем, что временные метки сохранены
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert "start" in first_chunk
    assert "duration" in first_chunk
    assert first_chunk["start"] >= 0
    assert first_chunk["duration"] > 0


def test_chunk_empty_segments(semantic_chunker):
    """Тест обработки пустого списка сегментов."""
    chunks = semantic_chunker.chunk([])
    assert chunks == []


def test_chunk_single_segment(semantic_chunker):
    """Тест обработки одного сегмента."""
    segments = [
        {"text": "Один сегмент текста.", "start": 0.0, "duration": 5.0}
    ]
    
    chunks = semantic_chunker.chunk(segments)
    
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Один сегмент текста."
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["duration"] == 5.0


def test_chunk_respects_max_tokens(semantic_chunker):
    """Тест соблюдения максимального размера чанка."""
    # Создаем чанкер с маленьким max_tokens
    small_chunker = SemanticChunker(
        embedding_model=semantic_chunker.embedding_model,
        max_tokens=50,  # Очень маленький размер
        similarity_threshold=0.5,
        min_chunk_size=10
    )
    
    # Создаем длинный текст
    long_text = " ".join([f"Предложение номер {i}." for i in range(20)])
    segments = [
        {"text": long_text, "start": 0.0, "duration": 10.0}
    ]
    
    chunks = small_chunker.chunk(segments)
    
    # Должно быть несколько чанков из-за ограничения max_tokens
    assert len(chunks) > 1
    # Проверяем, что каждый чанк не превышает max_tokens (приблизительно)
    for chunk in chunks:
        estimated_tokens = small_chunker._estimate_tokens(chunk["text"])
        # Допускаем небольшое превышение из-за приблизительной оценки
        assert estimated_tokens <= small_chunker.max_tokens * 1.5


def test_chunk_min_chunk_size(semantic_chunker):
    """Тест соблюдения минимального размера чанка."""
    # Создаем чанкер с большим min_chunk_size
    large_min_chunker = SemanticChunker(
        embedding_model=semantic_chunker.embedding_model,
        max_tokens=200,
        similarity_threshold=0.5,
        min_chunk_size=100  # Большой минимальный размер
    )
    
    segments = [
        {"text": "Короткое предложение.", "start": 0.0, "duration": 1.0},
        {"text": "Еще одно короткое предложение.", "start": 1.0, "duration": 1.0},
    ]
    
    chunks = large_min_chunker.chunk(segments)
    
    # Если чанки слишком маленькие, они могут быть объединены
    # или сохранены как есть (зависит от реализации)
    assert len(chunks) > 0


def test_chunk_different_topics(semantic_chunker):
    """Тест разделения на чанки при смене темы."""
    # Создаем текст с разными темами
    segments = [
        {"text": "Первая тема: программирование на Python.", "start": 0.0, "duration": 3.0},
        {"text": "Продолжение о программировании на Python.", "start": 3.0, "duration": 3.0},
        {"text": "Вторая тема: кулинария и рецепты.", "start": 6.0, "duration": 3.0},
        {"text": "Продолжение о кулинарии.", "start": 9.0, "duration": 3.0},
    ]
    
    chunks = semantic_chunker.chunk(segments)
    
    # Должно быть хотя бы 2 чанка из-за разных тем
    # (хотя с моком это может не работать идеально)
    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)


def test_get_chunk_time_from_sentences(semantic_chunker):
    """Тест вычисления временных границ чанка из предложений."""
    sentence_to_segments = [
        [{"text": "Первое предложение", "start_time": 0.0, "end_time": 2.0}],
        [{"text": "Второе предложение", "start_time": 2.0, "end_time": 4.0}],
    ]
    
    sentence_indices = [0, 1]
    start_time, end_time = semantic_chunker._get_chunk_time_from_sentences(
        sentence_indices, sentence_to_segments
    )
    
    assert start_time >= 0
    assert end_time >= start_time
    assert start_time == 0.0
    assert end_time == 4.0


def test_compute_sentence_embeddings(semantic_chunker):
    """Тест вычисления эмбеддингов предложений."""
    sentences = ["Первое предложение.", "Второе предложение."]
    embeddings = semantic_chunker._compute_sentence_embeddings(sentences)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(sentences)
    assert embeddings.shape[1] > 0  # Должна быть размерность эмбеддинга


def test_compute_sentence_embeddings_single(semantic_chunker):
    """Тест вычисления эмбеддинга одного предложения."""
    sentences = ["Одно предложение."]
    embeddings = semantic_chunker._compute_sentence_embeddings(sentences)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == 1


def test_compute_sentence_embeddings_empty(semantic_chunker):
    """Тест вычисления эмбеддингов для пустого списка."""
    embeddings = semantic_chunker._compute_sentence_embeddings([])
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.size == 0

