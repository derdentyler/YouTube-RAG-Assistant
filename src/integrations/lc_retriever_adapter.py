from typing import List
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from src.core.abstractions.vector_store import VectorStore


class LCRetrieverAdapter(Runnable):
    """
    Адаптер для интеграции VectorStore с LangChain Retriever API.
    """

    def __init__(self, vectorstore: VectorStore, top_k: int = 5):
        super().__init__()
        self.vectorstore = vectorstore
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Стандартный метод LangChain Retriever.
        """
        results = self.vectorstore.search(query, k=self.top_k)

        seen = set()
        docs = []
        for r in results:
            text = (r.get("page_content") or "").strip()
            if not text or len(text) < 30:
                continue
            if text in seen:
                continue
            seen.add(text)
            docs.append(Document(page_content=text, metadata=r.get("metadata", {})))
        return docs

    def invoke(self, query: str, *args, **kwargs) -> List[Document]:
        """
        Runnable API совместимость.
        """
        top_k = kwargs.get("top_k", self.top_k)
        self.top_k = top_k
        return self.get_relevant_documents(query)
