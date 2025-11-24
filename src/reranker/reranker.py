import numpy as np
from typing import List, Tuple
from src.reranker.features import FeatureBuilder
from src.reranker.ml_model import LogisticRegressionReranker
from src.core.abstractions.embeddings import Embedder
import os


class Reranker:
    """
    Reranker class: re-ranks document fragments based on features computed from embeddings and text.

    Differences from old version:
    - Accepts raw query string and list of document texts.
    - Embeddings are computed internally using a passed Embedder instance.
    - Tokenization handled inside the class for feature extraction.
    - Suitable for LangChain integration: only query + documents are needed externally.
    """

    def __init__(self, model_path: str, embedder: Embedder) -> None:
        """
        Initialize the Reranker.

        Args:
            model_path (str): path to the trained ML reranker model
            embedder (Embedder): embedding model implementing .encode()
        """
        self.embedder = embedder

        # Load ML reranker model
        self.model = LogisticRegressionReranker()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Reranker model not found at {model_path}")
        self.model.load(model_path)

        # Feature builder instance
        self.feature_builder = FeatureBuilder()

    def rerank(self, query: str, doc_texts: List[str]) -> List[Tuple[str, float]]:
        """
        Re-rank document texts based on predicted relevance.

        Args:
            query (str): the user's query
            doc_texts (List[str]): list of document strings to rerank

        Returns:
            List[Tuple[str, float]]: list of tuples (document_text, score) sorted by score descending
        """
        if not doc_texts:
            return []

        # 1 Compute embeddings using internal Embedder
        query_embedding = self.embedder.encode(query, convert_to_tensor=False)
        doc_embeddings = self.embedder.encode(doc_texts, convert_to_tensor=False)
        if hasattr(query_embedding, "cpu"):
            query_embedding = query_embedding.cpu().numpy()
        if hasattr(doc_embeddings, "cpu"):
            doc_embeddings = doc_embeddings.cpu().numpy()
        doc_embeddings_list = list(doc_embeddings)

        # 2 Tokenize query and documents for feature computation
        query_tokens = query.lower().split()
        doc_tokens_list = [doc.lower().split() for doc in doc_texts]

        # 3 Build features for ML model
        features = self.feature_builder.build(
            query_emb=query_embedding,
            doc_embs=doc_embeddings_list,
            query_tokens=query_tokens,
            doc_tokens_list=doc_tokens_list,
            doc_texts=doc_texts
        )
        X = np.vstack(features)

        # 4 Predict scores using ML model
        scores = self.model.predict(X)

        # 5 Zip texts with scores and sort descending
        reranked = sorted(
            zip(doc_texts, scores), key=lambda x: x[1], reverse=True
        )
        return reranked
