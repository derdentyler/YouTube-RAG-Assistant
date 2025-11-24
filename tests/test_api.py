import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.api.main import app
from src.core.dependencies.providers import get_rag_model, get_config
from src.core.config.models import AppConfig, ModelConfigLlamaCpp, ModelConfigTransformers


def test_health_check():
    """Тест health check эндпоинта."""
    # Создаем мок конфига для теста
    mock_config = AppConfig(
        language='ru',
        models={
            'ru': ModelConfigLlamaCpp(
                backend='llama.cpp',
                model_path='./models/llm/test_model.gguf',  # Путь не проверяется благодаря SKIP_MODEL_FILE_CHECK
                n_ctx=1024
            ),
            'en': ModelConfigTransformers(
                backend='transformers',
                model_name='test-model',
                n_ctx=1024
            )
        },
        embedding_model='test-embedding-model',
        embedding_dimension=768
    )
    
    # Мокируем провайдер конфига
    app.dependency_overrides[get_config] = lambda: mock_config
    
    try:
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "config_language" in data
        assert "models_loaded" in data
    finally:
        # Очищаем переопределения
        app.dependency_overrides.clear()


def test_query_endpoint_with_mock():
    """Тест query эндпоинта с мок RAG модели."""
    
    # Создаем мок RAG модели
    mock_rag = MagicMock()
    mock_rag.process_query.return_value = "Тестовый ответ от модели"
    
    # Переопределяем зависимость
    def mock_get_rag_model():
        return mock_rag
    
    app.dependency_overrides[get_rag_model] = mock_get_rag_model
    
    try:
        client = TestClient(app)
        response = client.post(
            "/query",
            json={
                "video_url": "https://www.youtube.com/watch?v=test",
                "query": "Тестовый вопрос"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Тестовый ответ от модели"
        
        # Проверяем что модель была вызвана с правильными параметрами
        mock_rag.process_query.assert_called_once()
        
    finally:
        # Очищаем переопределения
        app.dependency_overrides.clear()


def test_query_endpoint_handles_errors():
    """Тест обработки ошибок в query эндпоинте."""
    
    mock_rag = MagicMock()
    mock_rag.process_query.side_effect = RuntimeError("Test error")
    
    def mock_get_rag_model():
        return mock_rag
    
    app.dependency_overrides[get_rag_model] = mock_get_rag_model
    
    try:
        client = TestClient(app)
        response = client.post(
            "/query",
            json={
                "video_url": "https://www.youtube.com/watch?v=test",
                "query": "Тестовый вопрос"
            }
        )
        
        assert response.status_code == 500
        assert "detail" in response.json()
        
    finally:
        app.dependency_overrides.clear()

