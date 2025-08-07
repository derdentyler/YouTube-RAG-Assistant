from typing import List, Dict, Any, Union
from src.core.abstractions.vector_store import VectorStore
from src.utils.db_connector import DBConnector
from src.core.abstractions.embeddings import Embedder

class DBVectorStore(VectorStore):
    """
    Adapter for vector storage using DBConnector.
    Uses Embedder interface for embedding text.
    """
    def __init__(self, db_connector: DBConnector, embedding_model: Embedder):
        self.db = db_connector
        self.embedding_model = embedding_model

    def add(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Embed and insert each document into the subtitles table.
        """
        docs_embs = self.embedding_model.encode(texts, convert_to_tensor=False)

        for text, meta, emb in zip(texts, metadatas, docs_embs):
            vec = emb.tolist() if hasattr(emb, 'tolist') else emb
            self.db.insert_subtitle(
                video_id=meta["video_id"],
                start_time=meta["start_time"],
                end_time=meta["end_time"],
                text=text,
                embedding=vec
            )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Compute query embedding and search in DB.
        """
        q_emb = self.embedding_model.encode(query, convert_to_tensor=False)
        raw = q_emb.tolist() if hasattr(q_emb, 'tolist') else q_emb
        results = self.db.search_similar_embeddings(raw, top_k=k)

        wrapped: List[Dict[str, Any]] = []
        for text, score in results:
            wrapped.append({"page_content": text, "score": score})
        return wrapped
