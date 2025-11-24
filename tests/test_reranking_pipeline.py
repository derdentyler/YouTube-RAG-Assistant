import pytest
import numpy as np
from src.utils.config_loader import ConfigLoader
from src.reranker.reranker import Reranker

@pytest.fixture(autouse=True)
def patch_pipeline_deps(monkeypatch, tmp_path):
    # фиктивный путь модели
    model_path = tmp_path / "lr.pkl"
    model_path.write_text("fake")

    # ConfigLoader
    loader = ConfigLoader()
    loader.config = {"reranker": {"model_path": str(model_path)}}
    ConfigLoader._instance = loader

    # мок ML-модели
    class DummyModel:
        def load(self, path): pass
        def predict(self, X): return [0.9, 0.1]
    monkeypatch.setattr("src.reranker.reranker.LogisticRegressionReranker", lambda: DummyModel())

    # мок FeatureBuilder
    class DummyFB:
        def build(self, **kwargs):
            return [np.array([0.1]), np.array([0.9])]
    monkeypatch.setattr("src.reranker.reranker.FeatureBuilder", lambda: DummyFB())

def test_end_to_end_rerank():
    model_path = ConfigLoader.get_config()["reranker"]["model_path"]

    # Dummy embedder
    class DummyEmbedder:
        def encode(self, x, convert_to_tensor=False):
            if isinstance(x, str):
                return np.array([1.0])
            return [np.array([1.0]), np.array([0.0])]

    rr = Reranker(model_path=model_path, embedder=DummyEmbedder())
    doc_texts = ["a", "b"]
    res = rr.rerank("query", doc_texts)

    assert set([r[0] for r in res]) == set(doc_texts)
