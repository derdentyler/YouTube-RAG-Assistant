import numpy as np
import pytest
from pathlib import Path

from src.reranker.ml_model import LogisticRegressionReranker
from src.utils.config_loader import ConfigLoader
from src.reranker.features import FeatureBuilder
from src.reranker.reranker import Reranker

@pytest.fixture(autouse=True)
def train_and_set_model(tmp_path):
    """
    Тренируем reranker на тех же данных и признаках,
    которые будут использоваться в тесте ниже.
    """
    # 1) Данные для тренировки – те же, что мы будем давать в тесте
    q_emb = np.array([1.0])  # единичный эмбеддинг запроса
    doc_embs = [np.array([1.0]), np.array([0.0])]  # два документа
    q_tokens = ["a"]
    d_tokens = [["a"], ["b"]]
    texts = ["a", "b"]

    # 2) Строим фичи
    fb = FeatureBuilder()
    feats = fb.build(
        query_emb=q_emb,
        doc_embs=doc_embs,
        query_tokens=q_tokens,
        doc_tokens_list=d_tokens,
        doc_texts=texts
    )
    X = np.vstack(feats)
    y = np.array([1, 0])  # первый – релевантен, второй – нет

    # 3) Тренируем и сохраняем модель
    model = LogisticRegressionReranker()
    model.train(X, y)
    model_path = tmp_path / "lr.pkl"
    model.save(str(model_path))

    # 4) Подменяем путь в конфиге
    loader = ConfigLoader()
    loader.config = {"reranker": {"model_path": str(model_path)}}
    ConfigLoader._instance = loader

def test_end_to_end_rerank():
    # читаем путь из конфига
    model_path = ConfigLoader.get_config()["reranker"]["model_path"]
    rr = Reranker(model_path=model_path)

    # те же входные данные
    q_emb = np.array([1.0])
    doc_embs = [np.array([1.0]), np.array([0.0])]
    q_tokens = ["a"]
    d_tokens = [["a"], ["b"]]
    texts = ["a", "b"]

    # запускаем rerank
    res = rr.rerank(
        query_embedding=q_emb,
        doc_embeddings=doc_embs,
        query_tokens=q_tokens,
        doc_tokens_list=d_tokens,
        doc_texts=texts
    )

    # проверяем, что первый фрагмент – "a", второй – "b"
    assert res[0][0] == "a"
    assert res[1][0] == "b"