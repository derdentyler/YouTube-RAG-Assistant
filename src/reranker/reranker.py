import numpy as np
from typing import List, Tuple
from .features import build_features
from .ml_model import LogisticRegressionReranker


class Reranker:
    def __init__(self, model_path: str = "models/logreg_reranker.pkl"):
        self.model = LogisticRegressionReranker()
        self.model.load(model_path)

    def rerank(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray],
        query_tokens: List[str],
        doc_tokens_list: List[List[str]],
        doc_texts: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Возвращает список текстов, отсортированных по убыванию релевантности.
        """

        total_docs = len(doc_texts)
        doc_indices = list(range(total_docs))

        # 1. Получаем признаки
        features = build_features(
            query_embedding=query_embedding,
            doc_embeddings=doc_embeddings,
            query_tokens=query_tokens,
            doc_tokens_list=doc_tokens_list,
            doc_texts=doc_texts,
            doc_indices=doc_indices,
            total_docs=total_docs
        )

        # 2. Предсказания модели
        X = np.array([f[1] for f in features])  # массив признаков
        scores = self.model.predict(X)

        # 3. Сортируем по score (высший — первый)
        reranked = sorted(
            zip(doc_texts, scores), key=lambda x: x[1], reverse=True
        )

        return reranked
