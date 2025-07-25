# src/reranker/trainer.py
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import joblib

from src.reranker.features import FeatureBuilder


def main(
    train_path: str = "downloads/reranker/train_data.json",
    model_out: str = "models/logreg_reranker.pkl"
):
    # 1. Загрузка размеченного датасета
    with open(train_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Плоский список записей: query, text, label
    records = []
    for item in raw:
        query = item["query"]
        for frag in item.get("fragments", []):
            records.append({
                "query": query,
                "text": frag["text"],
                "label": frag["label"]
            })

    # 2. Эмбеддинги и токены
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    emb_model = SentenceTransformer(model_name)

    queries = [r["query"] for r in records]
    texts   = [r["text"]  for r in records]
    labels  = np.array([r["label"] for r in records])

    # Получаем NumPy эмбеддинги
    q_embs = emb_model.encode(queries, convert_to_tensor=False)
    d_embs = emb_model.encode(texts,  convert_to_tensor=False)

    # Токенизация для фич
    q_tokens = [q.lower().split() for q in queries]
    d_tokens = [t.lower().split() for t in texts]

    # 3. Построение признаков через FeatureBuilder
    fb = FeatureBuilder()
    X_list = []
    for i in range(len(records)):
        feats = fb.build(
            query_emb=q_embs[i],
            doc_embs=[d_embs[i]],
            query_tokens=q_tokens[i],
            doc_tokens_list=[d_tokens[i]],
            doc_texts=[texts[i]]
        )
        X_list.append(feats[0])  # единственный вектор из списка

    X = np.vstack(X_list)

    # 4. Обучение модели
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)

    # 5. Сохранение модели
    Path(model_out).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(clf, model_out)
    print(f"Model saved to {model_out}")


if __name__ == "__main__":
    main()
