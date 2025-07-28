import numpy as np
import os
import pytest

class DummyFB:
    def __init__(self):
        pass
    def build(self, **kwargs):
        # вернёт два вектора длины 1
        return [np.array([0.2]), np.array([0.8])]

class DummyModel:
    def __init__(self):
        pass
    def load(self, path): pass
    def predict(self, X): return [0.2, 0.8]

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, tmp_path):
    # создаём пустой файл модели
    p = tmp_path / "dummy.pkl"
    p.write_text("dummy")
    # подменяем ConfigLoader внутри Reranker, чтобы он взял tmp_path
    from src.utils.config_loader import ConfigLoader
    cfg = ConfigLoader.get_config()
    cfg["reranker"]["model_path"] = str(p)
    # подменяем зависимости
    monkeypatch.setattr("src.reranker.reranker.LogisticRegressionReranker", lambda: DummyModel())
    monkeypatch.setattr("src.reranker.reranker.FeatureBuilder", lambda: DummyFB())

def test_reranker_file_not_found(monkeypatch):
    # явно указываем несуществующий путь
    from src.reranker.reranker import Reranker as RR
    with pytest.raises(FileNotFoundError):
        RR(model_path="no_file.pkl")

def test_rerank_orders_correctly():
    # читаем путь из фикстуры (ConfigLoader уже был поправлен)
    from src.utils.config_loader import ConfigLoader
    cfg = ConfigLoader.get_config()
    path = cfg["reranker"]["model_path"]

    from src.reranker.reranker import Reranker
    rr = Reranker(model_path=path)

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
