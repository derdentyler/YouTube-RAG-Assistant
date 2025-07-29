import numpy as np
from typing import List, Tuple
from src.reranker.features import FeatureBuilder
from src.reranker.ml_model import LogisticRegressionReranker
import os


class Reranker:
    """
    Reranker class: loads a trained model and re-ranks document fragments based on features.
    """
    def __init__(self, model_path: str) -> None:
        self.model = LogisticRegressionReranker()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Reranker model not found at {model_path}")
        self.model.load(model_path)
        self.feature_builder = FeatureBuilder()

    def rerank(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray],
        query_tokens: List[str],
        doc_tokens_list: List[List[str]],
        doc_texts: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Re-rank document texts by predicted relevance score.

        Args:
            query_embedding: numpy array of query embedding
            doc_embeddings: list of numpy arrays for each doc fragment embedding
            query_tokens: tokenized query
            doc_tokens_list: list of token lists for each doc fragment
            doc_texts: list of document fragment texts

        Returns:
            List of tuples (text, score), sorted by score descending
        """
        # 1. Compute feature vectors
        features = self.feature_builder.build(
            query_emb=query_embedding,
            doc_embs=doc_embeddings,
            query_tokens=query_tokens,
            doc_tokens_list=doc_tokens_list,
            doc_texts=doc_texts
        )

        # 2. Stack into feature matrix
        X = np.vstack(features)

        # 3. Predict relevance scores
        scores = self.model.predict(X)

        # 4. Zip texts and scores, sort by score descending
        reranked = sorted(
            zip(doc_texts, scores), key=lambda x: x[1], reverse=True
        )
        return reranked
