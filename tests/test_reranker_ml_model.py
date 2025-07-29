import numpy as np
import tempfile
import os
import pytest
from src.reranker.ml_model import LogisticRegressionReranker

def test_train_predict_save_load(tmp_path):
    # синтетический датасет: X=[[0],[1]], y=[0,1]
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    model = LogisticRegressionReranker()
    model.train(X, y)

    # сохраняем
    out = tmp_path / "lr.pkl"
    model.save(str(out))
    assert out.exists()

    # загружаем
    model2 = LogisticRegressionReranker()
    model2.load(str(out))
    preds = model2.predict(X)
    assert preds[0] < 0.5
    assert preds[1] > 0.5

def test_load_nonexistent():
    model = LogisticRegressionReranker()
    with pytest.raises(FileNotFoundError):
        model.load("no_such_file.pkl")
