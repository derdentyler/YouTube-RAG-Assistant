import numpy as np
from typing import List, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer

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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def token_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    overlap = query_set.intersection(doc_set)
    return len(overlap) / len(query_set)

def stopword_ratio(doc_tokens: List[str]) -> float:
    if not doc_tokens:
        return 0.0
    stop_count = sum(1 for t in doc_tokens if t in STOPWORDS)
    return stop_count / len(doc_tokens)

def length_diff_ratio(query_text: str, doc_text: str) -> float:
    len_q = len(query_text)
    len_d = len(doc_text)
    if len_q + len_d == 0:
        return 0.0
    return abs(len_d - len_q) / (len_q + 1e-5)

def position_feature(doc_index: int, total_docs: int) -> float:
    if total_docs <= 1:
        return 0.0
    return float(doc_index) / float(total_docs - 1)

def tfidf_similarity(query_text: str, doc_text: str, vectorizer: TfidfVectorizer) -> float:
    try:
        tfidf_matrix = vectorizer.transform([query_text, doc_text])
        q_vec = tfidf_matrix[0].toarray()[0]
        d_vec = tfidf_matrix[1].toarray()[0]
        return cosine_similarity(q_vec, d_vec)
    except Exception:
        return 0.0

def build_features(
    query_embedding: np.ndarray,
    doc_embeddings: List[np.ndarray],
    query_tokens: List[str],
    doc_tokens_list: List[List[str]],
    doc_texts: List[str],
    doc_indices: List[int],
    total_docs: int
) -> List[Tuple[int, np.ndarray]]:
    features = []
    query_text = ' '.join(query_tokens)

    # Обучаем TF-IDF vectorizer на запросе и всех фрагментах
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit([query_text] + doc_texts)

    for emb, tokens, text, idx in zip(doc_embeddings, doc_tokens_list, doc_texts, doc_indices):
        feats = [
            cosine_similarity(query_embedding, emb),                      # 1. косинусная близость
            token_overlap(query_tokens, tokens),                          # 2. пересечение токенов
            stopword_ratio(tokens),                                       # 3. доля стоп-слов
            length_diff_ratio(query_text, text),                          # 4. относительная длина
            position_feature(idx, total_docs),                            # 5. нормализованная позиция
            tfidf_similarity(query_text, text, tfidf_vectorizer)          # 6. TF-IDF сходство
        ]
        features.append((idx, np.array(feats, dtype=float)))

    return features
