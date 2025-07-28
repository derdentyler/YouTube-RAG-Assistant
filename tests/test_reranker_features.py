import numpy as np
import pytest
from src.reranker.features import (
    cosine_sim,
    token_overlap,
    stopword_ratio,
    length_diff_ratio,
    position_feature,
    FeatureBuilder,
)

def test_cosine_sim_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(cosine_sim(v, v)) == 1.0

def test_cosine_sim_invalid_dim():
    with pytest.raises(ValueError):
        cosine_sim(np.array([[1]]), np.array([1]))

def test_token_overlap_and_stopword_ratio():
    q = ["a", "b", "и"]  # 'и' — стоп-слово
    d = ["a", "c", "и", "d"]
    # overlap: only 'a' matches, query_set = {'a','b'} → 1/2
    assert pytest.approx(token_overlap(q, d)) == 0.5
    # stopwords in d = {'и'} → 1/4
    assert pytest.approx(stopword_ratio(d)) == 0.25

def test_length_diff_ratio_and_position():
    assert length_diff_ratio("a", "aaa") == pytest.approx(abs(3-1)/4)
    assert position_feature(0, 1) == 0.0
    assert position_feature(1, 4) == pytest.approx(1/3)

def test_feature_builder_simple():
    fb = FeatureBuilder()
    # два документа с одинаковым текстом
    q_emb = np.array([1.0, 0.0])
    d_embs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    q_tokens = ["x", "y"]
    d_tokens = [["x"], ["z"]]
    texts = ["x", "z"]
    feats = fb.build(
        query_emb=q_emb,
        doc_embs=d_embs,
        query_tokens=q_tokens,
        doc_tokens_list=d_tokens,
        doc_texts=texts
    )
    # Должно вернуть список длины 2, каждая запись — ndarray длины 6
    assert isinstance(feats, list) and len(feats) == 2
    assert all(isinstance(f, np.ndarray) and f.shape == (6,) for f in feats)
