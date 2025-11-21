import pytest
from unittest.mock import MagicMock
from src.utils.db_connector import DBConnector


def test_get_connection_without_pool_raises():
    """Проверка ошибки при отсутствии пула соединений."""
    db = DBConnector.__new__(DBConnector)
    db._pool = None
    db.logger = MagicMock()
    with pytest.raises(RuntimeError):
        with db.get_connection():
            pass


def test_close_closes_pool():
    """Проверка закрытия пула соединений."""
    db = DBConnector.__new__(DBConnector)
    mock_pool = MagicMock()
    db._pool = mock_pool

    db.close()
    mock_pool.closeall.assert_called_once()


def test_connection_released_on_exception():
    """Гарантированный возврат соединения через context manager при ошибке."""
    db = DBConnector.__new__(DBConnector)
    mock_pool = MagicMock()
    mock_conn = MagicMock()

    db._pool = mock_pool
    db.logger = MagicMock()
    mock_pool.getconn.return_value = mock_conn

    # Проверяем что соединение возвращается даже при исключении внутри context manager
    with pytest.raises(RuntimeError):
        with db.get_connection() as conn:
            raise RuntimeError("Test error")
    
    # Проверяем что соединение было возвращено в пул даже при исключении
    mock_pool.getconn.assert_called_once()
    mock_pool.putconn.assert_called_once_with(mock_conn)


def test_sql_injection_protection():
    """Проверка защиты от SQL-инъекций."""
    db = DBConnector.__new__(DBConnector)
    db.logger = MagicMock()
    
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    db._pool = mock_pool
    mock_pool.getconn.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None

    # Вызываем метод с опасным вводом
    db.insert_subtitle("hack' OR 1=1--", 0, 1, "text", [])

    # Проверяем параметризованный запрос
    mock_cursor.execute.assert_called_once()
    args, kwargs = mock_cursor.execute.call_args
    assert "%s" in args[0]
    assert args[1] == ("hack' OR 1=1--", 0, 1, "text", [])
    
    # Проверяем что соединение было возвращено
    mock_pool.putconn.assert_called_once_with(mock_conn)


def test_context_manager_properly_releases_connection():
    """Проверка что context manager корректно освобождает соединение."""
    db = DBConnector.__new__(DBConnector)
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    
    db._pool = mock_pool
    db.logger = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    
    with db.get_connection() as conn:
        assert conn == mock_conn
    
    mock_pool.getconn.assert_called_once()
    mock_pool.putconn.assert_called_once_with(mock_conn)


def test_dynamic_embedding_dimension():
    """Проверка динамического создания таблицы с разной размерностью."""
    db = DBConnector.__new__(DBConnector)
    db._pool = MagicMock()
    db.logger = MagicMock()
    db.embedding_dimension = 1024  # Нестандартная размерность
    
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None
    
    db.create_subtitles_table(mock_conn)
    
    # Проверяем что SQL содержит правильную размерность
    call_args = mock_cursor.execute.call_args[0][0]
    assert "VECTOR(1024)" in call_args
