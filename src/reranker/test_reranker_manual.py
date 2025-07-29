import numpy as np
from sentence_transformers import SentenceTransformer
from src.reranker.reranker import Reranker

def to_numpy(vec):
    """Ensure embedding is a NumPy array."""
    if hasattr(vec, "cpu"):
        return vec.cpu().numpy()
    return vec

def run_example(query_text, doc_texts, reranker, embed_model):
    # 1) Получаем эмбеддинги как NumPy
    query_emb = to_numpy(embed_model.encode(query_text, convert_to_tensor=False))
    doc_embs = [
        to_numpy(e) for e in embed_model.encode(doc_texts, convert_to_tensor=False)
    ]

    # 2) Токенизация
    query_tokens = query_text.lower().split()
    doc_tokens_list = [text.lower().split() for text in doc_texts]

    # 3) Реранкинг
    ranked = reranker.rerank(
        query_emb,
        doc_embs,
        query_tokens,
        doc_tokens_list,
        doc_texts
    )

    # 4) Печать результатов
    print(f"\nЗапрос: {query_text}")
    for rank, (text, score) in enumerate(ranked, start=1):
        print(f"{rank}. [{score:.3f}] {text}")


if __name__ == "__main__":
    # 0) Инициализация модели эмбеддингов и реранкера
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = Reranker("models/reranker/logreg_reranker.pkl")

    # Пример 1
    query1 = "Как работает закон притяжения?"
    docs1 = [
        "Закон притяжения объясняет, как мысли материализуются.",
        "Ньютон сформулировал законы механики.",
        "Видео рассказывает про принципы успеха.",
        "Футбол меня притягивает с детства"
    ]
    run_example(query1, docs1, reranker, embed_model)

    # Пример 2
    query2 = "Что такое квантовая запутанность?"
    docs2 = [
        "Квантовая запутанность описывает корреляцию между частицами.",
        "Это явление не объясняется классической физикой.",
        "Запутанность позволяет мгновенно влиять на состояние другой частицы.",
        "Ньютон предложил три закона движения."
    ]
    run_example(query2, docs2, reranker, embed_model)
