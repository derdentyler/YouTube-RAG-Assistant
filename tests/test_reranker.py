import numpy as np
import pytest
import os

from src.utils.config_loader import ConfigLoader
from src.reranker.reranker import Reranker

class DummyFB:
    def build(self, **kwargs):
        return [np.array([0.2]), np.array([0.8])]

class DummyModel:
    def load(self, path): pass
    def predict(self, X): return [0.2, 0.8]

@pytest.fixture(autouse=True)
def patch_reranker_deps(monkeypatch, tmp_path):
    # 1) Создаём файл модели
    pick = tmp_path / "dummy.pkl"
    pick.write_text("ok")

    # 2) Создаём и настраиваем инстанс ConfigLoader
    loader = ConfigLoader()
    loader.config = {"reranker": {"model_path": str(pick)}}
    # Гарантируем, что get_config() вернёт наш loader.config
    ConfigLoader._instance = loader

    # 3) Мокаем зависимости
    monkeypatch.setattr("src.reranker.reranker.LogisticRegressionReranker", lambda: DummyModel())
    monkeypatch.setattr("src.reranker.reranker.FeatureBuilder", lambda: DummyFB())

def test_init_raises_if_missing():
    # Перепишем путь так, чтобы не существовал
    loader = ConfigLoader._instance
    loader.config["reranker"]["model_path"] = "/no/such/file.pkl"
    with pytest.raises(FileNotFoundError):
        Reranker(model_path=loader.config["reranker"]["model_path"])

def test_rerank_orders_correctly():
    # В конфиге уже правильный путь из фикстуры
    model_path = ConfigLoader._instance.config["reranker"]["model_path"]
    rr = Reranker(model_path=model_path)
    texts = ["low", "high"]
    q_emb = np.array([0.])
    doc_embs = [np.array([0.]), np.array([0.])]
    res = rr.rerank(
        query_embedding=q_emb,
        doc_embeddings=doc_embs,
        query_tokens=["x"],
        doc_tokens_list=[["x"], ["y"]],
        doc_texts=texts
    )
    assert res[0][0] == "high"
    assert res[1][0] == "low"