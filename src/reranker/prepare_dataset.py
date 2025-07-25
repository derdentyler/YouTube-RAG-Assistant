# src/reranker/prepare_dataset.py

import json
from pathlib import Path
from typing import List, Dict

from src.utils.db_connector import DBConnector
from src.answer_generator.rag_model import RAGModel


def build_dataset(
    queries_path: str,
    output_path: str,
    top_k: int = 5
):
    # 1) Загрузим запросы
    with open(queries_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    # 2) Инициализируем пул соединений и RAGModel
    db_connector = DBConnector()
    rag_model = RAGModel(db_connector=db_connector)

    dataset: List[Dict] = []

    for item in examples:
        query = item["query"]
        video_url = item["video_url"]

        # 3) Получаем от retriever список (text, score)
        results = rag_model.retriever.retrieve(query)
        # Берём только top_k фрагментов
        top_results = results[:top_k]

        # 4) Формируем запись без score, только текст
        fragments = [{"text": text} for text, _ in top_results]

        dataset.append({
            "query": query,
            "video_url": video_url,
            "fragments": fragments
        })

    # 5) Сохраняем в JSON
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    build_dataset(
        queries_path="downloads/reranker/queries.json",
        output_path="downloads/reranker/unlabeled.json",
        top_k=5
    )
