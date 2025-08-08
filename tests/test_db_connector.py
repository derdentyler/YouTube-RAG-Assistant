import pytest
from unittest.mock import MagicMock, patch
from src.utils.db_connector import DBConnector


def test_get_connection_without_pool_raises():
    """Проверка ошибки при отсутствии пула соединений."""
    db = DBConnector.__new__(DBConnector)
    db._pool = None
    with pytest.raises(RuntimeError):
        db.get_connection()


def test_get_and_release_connection_calls_pool_methods():
    """Проверка работы с пулом соединений."""
    db = DBConnector.__new__(DBConnector)
    mock_pool = MagicMock()
    db._pool = mock_pool

    conn = db.get_connection()
    mock_pool.getconn.assert_called_once()

    db.release_connection(conn)
    mock_pool.putconn.assert_called_once_with(conn)


def test_close_closes_pool():
    """Проверка закрытия пула соединений."""
    db = DBConnector.__new__(DBConnector)
    mock_pool = MagicMock()
    db._pool = mock_pool

    db.close()
    mock_pool.closeall.assert_called_once()


def test_connection_released_on_exception():
    """Гарантированный возврат соединения при ошибке."""
    db = DBConnector.__new__(DBConnector)
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    db._pool = mock_pool
    mock_pool.getconn.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.execute.side_effect = RuntimeError("Test error")

    db.drop_table()
    mock_pool.putconn.assert_called_once_with(mock_conn)


def test_sql_injection_protection():
    """Проверка защиты от SQL-инъекций."""
    db = DBConnector.__new__(DBConnector)
    # Инициализация для работы release_connection
    db._pool = MagicMock()

    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Настраиваем контекстный менеджер для курсора
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None

    with patch.object(db, 'get_connection', return_value=mock_conn):
        # Вызываем метод с опасным вводом
        db.insert_subtitle("hack' OR 1=1--", 0, 1, "text", [])

        # Проверяем что execute был вызван
        mock_cursor.execute.assert_called_once()

        # Получаем аргументы вызова
        args, kwargs = mock_cursor.execute.call_args

        # Проверяем параметризованный запрос
        assert "%s" in args[0]  # Должен быть параметризованный запрос
        assert args[1] == ("hack' OR 1=1--", 0, 1, "text", [])  # Проверяем параметры
