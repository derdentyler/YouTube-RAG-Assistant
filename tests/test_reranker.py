import pytest
import numpy as np
from unittest.mock import MagicMock

from src.utils.config_loader import ConfigLoader
from src.reranker.reranker import Reranker

# --- фикстура для мока reranker и embedder ---
@pytest.fixture(autouse=True)
def patch_reranker_deps(monkeypatch, tmp_path):
    # 1) фиктивный путь модели
    model_path = tmp_path / "dummy.pkl"
    model_path.write_text("ok")

    # 2) подменяем ConfigLoader
    loader = ConfigLoader()
    loader.config = {"reranker": {"model_path": str(model_path)}}
    ConfigLoader._instance = loader

    # 3) мок ML-модели
    class DummyModel:
        def load(self, path): pass
        def predict(self, X): return [0.9, 0.1]
    monkeypatch.setattr("src.reranker.reranker.LogisticRegressionReranker", lambda: DummyModel())

    # 4) мок FeatureBuilder
    class DummyFB:
        def build(self, **kwargs):
            return [np.array([0.1]), np.array([0.9])]
    monkeypatch.setattr("src.reranker.reranker.FeatureBuilder", lambda: DummyFB())

# --- тесты ---
def test_init_raises_if_missing():
    loader = ConfigLoader._instance
    loader.config["reranker"]["model_path"] = "/no/such/file.pkl"
    dummy_embedder = MagicMock()
    import os
    with pytest.raises(FileNotFoundError):
        Reranker(model_path=loader.config["reranker"]["model_path"], embedder=dummy_embedder)

def test_rerank_orders_correctly():
    loader = ConfigLoader._instance
    model_path = loader.config["reranker"]["model_path"]
    dummy_embedder = MagicMock()
    dummy_embedder.encode.side_effect = lambda x, convert_to_tensor=False: np.array([1.0]) if isinstance(x, str) else [np.array([1.0]), np.array([0.0])]

    rr = Reranker(model_path=model_path, embedder=dummy_embedder)
    texts = ["low", "high"]
    res = rr.rerank("query", texts)

    assert set([r[0] for r in res]) == set(texts)
