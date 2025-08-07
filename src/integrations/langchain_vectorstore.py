from typing import List, Any
from langchain_core.vectorstores import VectorStore as LCVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.utils.db_connector import DBConnector

class DBLangChainVectorStore(LCVectorStore):
    """
    LangChain VectorStore adapter over our Postgres+pgvector via DBConnector.
    """

    def __init__(self, db_connector: DBConnector, embedding_model: Embeddings):
        self.db = db_connector
        self.embedding_model = embedding_model

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        :param documents: список langchain_core.documents.Document
        :return: список сгенерированных ID (можно возвращать пустой список)
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # получаем эмбеддинги
        embeddings = self.embedding_model.embed_documents(texts)
        # вставляем в БД
        for text, meta, emb in zip(texts, metadatas, embeddings):
            self.db.insert_subtitle(
                video_id=meta["video_id"],
                start_time=meta["start_time"],
                end_time=meta["end_time"],
                text=text,
                embedding=emb.tolist() if hasattr(emb, "tolist") else emb
            )
        return []

    def similarity_search(
        self, query: str, k: int = 5, **kwargs
    ) -> List[Document]:
        """
        :return: топ-k langchain_core.documents.Document с полями page_content, metadata[\"score\"]
        """
        # 1) эмбеддим запрос
        q_emb = self.embedding_model.embed_query(query)
        q_emb_list = q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb
        # 2) ищем
        results = self.db.search_similar_embeddings(q_emb_list, top_k=k)
        # 3) упаковываем в Document
        docs = []
        for text, score in results:
            docs.append(Document(page_content=text, metadata={"score": score}))
        return docs
