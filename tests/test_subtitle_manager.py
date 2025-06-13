import pytest

from src.data_processing.subtitle_manager import SubtitleManager
from src.utils.config_loader import ConfigLoader

# Заглушка для модели эмбеддингов
class DummyEmbedding(list):
    def tolist(self):
        return list(self)

class DummyModel:
    def encode(self, text):
        # Возвращает DummyEmbedding с tolist()
        return DummyEmbedding([0.1, 0.2, 0.3])

# Заглушка для DBConnector
class DummyDB:
    def __init__(self):
        self.insert_calls = []

    def insert_subtitle(self, video_id, start_time, end_time, text, embedding):
        self.insert_calls.append((video_id, start_time, end_time, text, embedding))

@pytest.fixture(autouse=True)
def patch_config_and_model(monkeypatch):
    # Заглушаем конфиг, чтобы вернулся нужный ключ
    monkeypatch.setattr(
        ConfigLoader,
        "get_config",
        lambda: {"embedding_model": "dummy-model-path"}
    )
    # Заглушаем SentenceTransformer, чтобы он возвращал DummyModel
    monkeypatch.setattr(
        "src.data_processing.subtitle_manager.SentenceTransformer",
        lambda model_name: DummyModel()
    )

def test_get_embedding_returns_list_of_floats():
    sm = SubtitleManager(db_pool=DummyDB())
    emb = sm.get_embedding("test text")
    assert isinstance(emb, list)
    assert emb == [0.1, 0.2, 0.3]
    assert all(isinstance(x, float) for x in emb)

def test_add_subtitles_calls_db_insert_with_embedding():
    db = DummyDB()
    sm = SubtitleManager(db_pool=db)
    subtitles = [{"text": "hello", "start": 1.0, "duration": 2.0}]

    sm.add_subtitles(video_id="vid", subtitles=subtitles)

    # Проверяем вызов insert_subtitle
    assert len(db.insert_calls) == 1
    vid, start, end, text, embedding = db.insert_calls[0]
    assert vid == "vid"
    assert start == pytest.approx(1.0)
    assert end == pytest.approx(3.0)
    assert text == "hello"
    assert embedding == [0.1, 0.2, 0.3]
