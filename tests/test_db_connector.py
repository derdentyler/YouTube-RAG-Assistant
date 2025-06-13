import pytest

from src.utils.db_connector import DBConnector

# Dummy pool and connection for testing
class DummyPool:
    def __init__(self):
        self.getconn_called = False
        self.putconn_called = False
        self.closeall_called = False
        self._conn = object()

    def getconn(self):
        # Помечаем вызов и возвращаем фиктивное соединение
        self.getconn_called = True
        return self._conn

    def putconn(self, conn):
        # Помечаем вызов передачи соединения обратно
        self.putconn_called = True
        assert conn is self._conn

    def closeall(self):
        # Помечаем закрытие всех соединений
        self.closeall_called = True

# Тестируем методы DBConnector, избегая реального подключения к Postgres

def test_get_connection_without_pool_raises():
    """
    Если пул не инициализирован (_pool = None), get_connection должен бросить RuntimeError
    """
    db = DBConnector.__new__(DBConnector)
    # вручную сбрасываем пул
    db._pool = None
    with pytest.raises(RuntimeError):
        db.get_connection()


def test_get_and_release_connection_calls_pool_methods(monkeypatch):
    """
    Проверяем, что get_connection и release_connection вызывают методы пула getconn/putconn.
    """
    db = DBConnector.__new__(DBConnector)
    pool = DummyPool()
    db._pool = pool

    # get_connection возвращает то же соединение, что и pool.getconn
    conn = db.get_connection()
    assert pool.getconn_called, "Метод getconn должен быть вызван"
    assert conn is pool._conn, "get_connection должен вернуть объект из pool.getconn"

    # release_connection должен вернуть соединение обратно в пул
    db.release_connection(conn)
    assert pool.putconn_called, "Метод putconn должен быть вызван"


def test_close_closes_pool():
    """
    Проверяем, что метод close() вызывает pool.closeall().
    """
    db = DBConnector.__new__(DBConnector)
    pool = DummyPool()
    db._pool = pool

    db.close()
    assert pool.closeall_called, "close() должен вызвать pool.closeall()"
