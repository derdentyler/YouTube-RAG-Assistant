import numpy as np
from src.reranker.reranker import Reranker
from sentence_transformers import SentenceTransformer

# Примерные входные данные
query_text = "Как работает закон притяжения?"
doc_texts = [
    "Закон притяжения объясняет, как мысли материализуются.",
    "Ньютон сформулировал законы механики.",
    "Видео рассказывает про принципы успеха.",
    "Это описание гравитации и её действия."
]

# Загружаем модель sentence-transformers
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ШАГ 1: Получаем эмбеддинги, строго следя за типами!

# Здесь используем convert_to_tensor=True, чтобы получить torch.Tensor
query_embedding = model.encode(query_text, convert_to_tensor=True)
doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)

# ШАГ 2: Если получили torch.Tensor — конвертируем в numpy.ndarray
if hasattr(query_embedding, "cpu"):
    query_embedding = query_embedding.cpu().numpy()

# doc_embeddings — это батч эмбеддингов, преобразуем каждый элемент
doc_embeddings = [emb.cpu().numpy() if hasattr(emb, "cpu") else emb for emb in doc_embeddings]

# ШАГ 3: Подготавливаем токены для функций фичей
query_tokens = query_text.lower().split()
doc_tokens_list = [text.lower().split() for text in doc_texts]

# Загружаем обученный реранкер
reranker = Reranker("models/logreg_reranker.pkl")

# ШАГ 4: Запускаем реранкинг
result = reranker.rerank(
    query_embedding,
    doc_embeddings,  # теперь список np.ndarray
    query_tokens,
    doc_tokens_list,
    doc_texts
)

# Вывод результата
print("\nРезультаты реранкинга:")
for i, (text, score) in enumerate(result, 1):
    print(f"{i}. [{score:.3f}] {text}")
