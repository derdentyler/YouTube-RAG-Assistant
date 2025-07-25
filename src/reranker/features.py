import numpy as np
from typing import List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Набор русскоязычных стоп-слов
STOPWORDS: Set[str] = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так",
    "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если",
    "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь",
    "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней",
    "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот"
}

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input embeddings must be 1D arrays.")
    return float(sklearn_cosine(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def token_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Fraction of unique query tokens appearing in document tokens."""
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens) - STOPWORDS
    doc_set = set(doc_tokens) - STOPWORDS
    return len(query_set & doc_set) / len(query_set) if query_set else 0.0


def stopword_ratio(doc_tokens: List[str]) -> float:
    """Ratio of stopwords to total tokens in document."""
    if not doc_tokens:
        return 0.0
    return sum(1 for t in doc_tokens if t in STOPWORDS) / len(doc_tokens)


def length_diff_ratio(query_text: str, doc_text: str) -> float:
    """Relative length difference between query and document text."""
    len_q, len_d = len(query_text), len(doc_text)
    if len_q + len_d == 0:
        return 0.0
    return abs(len_d - len_q) / (len_q + len_d)


def position_feature(index: int, total: int) -> float:
    """Normalized position of document in result list."""
    if total <= 1:
        return 0.0
    return index / (total - 1)


class FeatureBuilder:
    """
    Builds feature vectors for query-document pairs.
    Responsibilities separated:
    - Initialize TF-IDF vectorizer once per query
    - Compute multiple features for each document
    """

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.query_text: str | None = None

    def fit_tfidf(self, query_text: str, doc_texts: List[str]) -> None:
        """
        Fit TF-IDF on combined query and document texts.
        Must be called before computing tfidf_similarity.
        """
        self.query_text = query_text
        try:
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit([query_text] + doc_texts)
        except Exception as e:
            raise RuntimeError(f"TF-IDF vectorizer fit failed: {e}")

    def tfidf_similarity(self, doc_text: str) -> float:
        """
        Compute cosine similarity between query and document in TF-IDF space.
        Requires fit_tfidf to be called first.
        """
        if self.vectorizer is None or self.query_text is None:
            raise RuntimeError("Vectorizer not fitted. Call fit_tfidf first.")
        try:
            mat = self.vectorizer.transform([self.query_text, doc_text]).toarray()
            return cosine_sim(mat[0], mat[1])
        except Exception as e:
            # Only catch TF-IDF specific errors
            return 0.0

    def build(self,
              query_emb: np.ndarray,
              doc_embs: List[np.ndarray],
              query_tokens: List[str],
              doc_tokens_list: List[List[str]],
              doc_texts: List[str]
              ) -> List[np.ndarray]:
        """
        Compute feature vectors for all document candidates for a single query.

        Returns:
            List of feature arrays, in same order as doc_texts.
        """
        total = len(doc_texts)
        # Initialize TF-IDF once
        q_text = ' '.join(query_tokens)
        self.fit_tfidf(q_text, doc_texts)

        features: List[np.ndarray] = []
        for idx, (emb, tokens, text) in enumerate(zip(doc_embs, doc_tokens_list, doc_texts)):
            feats = [
                cosine_sim(query_emb, emb),           # embedding similarity
                token_overlap(query_tokens, tokens),  # lexical overlap
                stopword_ratio(tokens),               # noise ratio
                length_diff_ratio(q_text, text),      # length difference
                position_feature(idx, total),         # result position
                self.tfidf_similarity(text)           # TF-IDF similarity
            ]
            features.append(np.array(feats, dtype=float))
        return features
