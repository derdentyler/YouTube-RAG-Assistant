import pytest
from unittest.mock import MagicMock, patch
from src.core.dependencies.container import DependencyContainer, get_container, reset_container
from src.core.config.models import AppConfig, ModelConfigLlamaCpp


@pytest.fixture
def mock_config(tmp_path):
    """Мок конфигурации для тестов."""
    model_file = tmp_path / "test_model.gguf"
    model_file.write_text("fake")
    
    return AppConfig(
        language='ru',
        models={
            'ru': ModelConfigLlamaCpp(
                backend='llama.cpp',
                model_path=str(model_file),
                n_ctx=1024
            )
        },
        embedding_model='test-model',
        embedding_dimension=768
    )


def test_container_singleton(mock_config):
    """Проверка что контейнер - синглтон."""
    reset_container()  # Сбрасываем перед тестом
    
    with patch('src.core.dependencies.container.ConfigLoader') as mock_config_loader:
        mock_instance = MagicMock()
        mock_instance.get_config.return_value = mock_config
        mock_config_loader.get_config.return_value = mock_config
        
        container1 = get_container()
        container2 = get_container()
        assert container1 is container2


def test_db_connector_singleton(mock_config):
    """Проверка что DBConnector создается один раз."""
    reset_container()
    
    with patch('src.core.dependencies.container.ConfigLoader') as mock_config_loader, \
         patch('src.core.dependencies.container.DBConnector') as mock_db_class:
        
        mock_config_loader.get_config.return_value = mock_config
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        
        container = get_container()
        db1 = container.get_db_connector()
        db2 = container.get_db_connector()
        
        assert db1 is db2
        assert mock_db_class.call_count == 1  # Создается только один раз


def test_rag_model_reuses_dependencies(mock_config):
    """Проверка что RAGModel переиспользует уже загруженные зависимости."""
    reset_container()
    
    # Мокаем все зависимости
    with patch('src.core.dependencies.container.ConfigLoader') as mock_config_loader, \
         patch('src.core.dependencies.container.DBConnector') as mock_db_class, \
         patch('src.core.dependencies.container.SentenceTransformer') as mock_embed_class, \
         patch('src.core.dependencies.container.model_factory') as mock_factory:
        
        mock_config_loader.get_config.return_value = mock_config
        mock_db = MagicMock()
        mock_embed = MagicMock()
        mock_llm = MagicMock()
        
        mock_db_class.return_value = mock_db
        mock_embed_class.return_value = mock_embed
        mock_factory.return_value = mock_llm
        
        container = get_container()
        
        # Получаем зависимости
        db = container.get_db_connector()
        embeddings = container.get_embedding_model()
        llm = container.get_llm()
        
        # Мокаем RAGModel
        with patch('src.core.dependencies.container.RAGModel') as mock_rag_class:
            mock_rag = MagicMock()
            mock_rag.db = db
            mock_rag.embedding_model = embeddings
            mock_rag.llm = llm
            mock_rag_class.return_value = mock_rag
            
            # Получаем RAG модель
            rag = container.get_rag_model()
            
            # Проверяем что использует те же объекты
            assert rag.db is db
            assert rag.embedding_model is embeddings
            assert rag.llm is llm


def test_container_cleanup(mock_config):
    """Проверка корректной очистки контейнера."""
    reset_container()
    
    # Мокаем DBConnector и ConfigLoader
    with patch('src.core.dependencies.container.ConfigLoader') as mock_config_loader, \
         patch('src.core.dependencies.container.DBConnector') as mock_db_class:
        
        mock_config_loader.get_config.return_value = mock_config
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        
        container = get_container()
        _ = container.get_db_connector()
        
        reset_container()
        
        # После reset контейнер должен быть новым
        new_container = get_container()
        assert new_container is not container

