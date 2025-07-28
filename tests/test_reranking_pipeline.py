import numpy as np
import pytest
from pathlib import Path

from src.reranker.ml_model import LogisticRegressionReranker
from src.utils.config_loader import ConfigLoader
from src.reranker.reranker import Reranker

@pytest.fixture(scope="module", autouse=True)
def train_and_set_model(tmp_path):
    # Тренируем модель
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    model = LogisticRegressionReranker()
    model.train(X, y)
    p = tmp_path / "lr.pkl"
    model.save(str(p))

    # Настраиваем ConfigLoader.instance
    loader = ConfigLoader()
    loader.config = {"reranker": {"model_path": str(p)}}
    ConfigLoader._instance = loader

def test_end_to_end_rerank():
    model_path = ConfigLoader._instance.config["reranker"]["model_path"]
    rr = Reranker(model_path=model_path)
    q_emb = np.array([0.0])
    doc_embs = [np.array([1.0]), np.array([0.0])]
    texts = ["low", "high"]
    res = rr.rerank(
        query_embedding=q_emb,
        doc_embeddings=doc_embs,
        query_tokens=["x"],
        doc_tokens_list=[["x"], ["y"]],
        doc_texts=texts
    )
    assert res[0][0] == "high"
    assert res[1][0] == "low"