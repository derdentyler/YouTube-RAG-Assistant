import pytest

from src.search.retriever import Retriever
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader

# Заглушка для эмбеддингов, чтобы иметь метод tolist()
class DummyEmbedding(list):
    def tolist(self):
        # Возвращаем список на основе себя
        return list(self)

# Заглушка для модели SentenceTransformer
class DummyModel:
    def __init__(self):
        # Храним все вызовы encode для проверки
        self.encode_calls = []

    def encode(self, text):
        # Записываем текст запроса
        self.encode_calls.append(text)
        # Возвращаем фиктивный эмбеддинг с методом tolist()
        return DummyEmbedding([0.5, 0.5, 0.5])

# Заглушка для DBConnector с методом поиска
class DummyDB:
    def __init__(self, results=None):
        # Результаты, которые будет возвращать метод
        self.results = results or []
        # Храним параметры вызовов для проверки
        self.search_calls = []

    def search_similar_embeddings(self, query_embedding, top_k):
        # Записываем параметры вызова
        self.search_calls.append((query_embedding, top_k))
        # Возвращаем заранее заданные результаты
        return self.results

    def close(self):
        # Для совместимости с Retriever.close()
        pass

# Заглушка для логгера, чтобы не писать в реальный лог
class DummyLogger:
    def info(self, msg):
        # Ничего не делаем
        pass

    def error(self, msg):
        # Ничего не делаем
        pass

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Обновляем ConfigLoader.get_config, чтобы он возвращал нужный ключ
    monkeypatch.setattr(ConfigLoader, 'get_config', lambda: {'embedding_model': 'dummy'})
    # Подменяем SentenceTransformer в модуле retriever на нашу DummyModel
    monkeypatch.setattr('src.search.retriever.SentenceTransformer', lambda model_name: DummyModel())
    # Подменяем LoggerLoader.get_logger на DummyLogger
    monkeypatch.setattr(LoggerLoader, 'get_logger', lambda: DummyLogger())


def test_retrieve_calls_db_and_returns_results():
    # Настраиваем DB с двумя фиктивными результатами
    dummy_results = [('text1', 0.9), ('text2', 0.8)]
    db = DummyDB(results=dummy_results)
    # Инициализируем Retriever с нашей заглушкой БД
    retr = Retriever(db_pool=db)

    # Вызываем метод retrieve с запросом и top_k=2
    res = retr.retrieve('query text', top_k=2)

    # Проверяем, что модель encode была вызвана с правильным текстом
    assert retr.embedder.encode_calls == ['query text']
    # Ожидаемый эмбеддинг после DummyModel.encode и tolist()
    expected_embedding = [0.5, 0.5, 0.5]
    # Проверяем, что поиск в БД был вызван с этим эмбеддингом и top_k=2
    assert db.search_calls == [(expected_embedding, 2)]
    # Проверяем, что метод вернул результаты из БД без изменений
    assert res == dummy_results


def test_retrieve_handles_exception_and_returns_empty():
    # Создаем DB, который бросает ошибку при поиске
    class ErrorDB(DummyDB):
        def search_similar_embeddings(self, query_embedding, top_k):
            # Симулируем ошибку базы
            raise RuntimeError('DB error')

    db = ErrorDB()
    retr = Retriever(db_pool=db)

    # При ошибке retrieve должен вернуть пустой список, а не падать
    res = retr.retrieve('q', top_k=3)
    assert res == []
