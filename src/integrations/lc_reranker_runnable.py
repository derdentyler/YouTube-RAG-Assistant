from typing import List, Dict
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from src.reranker.reranker import Reranker
from src.utils.logger_loader import LoggerLoader


class LCRerankerRunnable(Runnable):
    """
    LangChain Runnable adapter для нативного Reranker.

    Вход: {"query": str, "documents": List[Document]}
    Выход: List[Document] с добавленным metadata["rerank_score"]
    """

    MIN_LENGTH = 30

    def __init__(self, reranker: Reranker):
        super().__init__()
        self.logger = LoggerLoader.get_logger()
        if not reranker:
            raise ValueError("Reranker instance cannot be None")
        self.reranker = reranker
        self.logger.info("LCRerankerRunnable initialized successfully")

    def invoke(self, *args, **kwargs) -> List[Document]:
        try:
            # Получаем query и documents
            inputs: Dict = kwargs.get("inputs") or (args[0] if args else {})
            query: str = inputs.get("query", "")
            documents: List[Document] = inputs.get("documents", [])

            if not query:
                self.logger.warning("Empty query received for reranking")
                return []
            if not documents:
                self.logger.warning("No documents received for reranking")
                return []

            # Фильтрация: пустые, короткие, дубликаты
            filtered_docs: List[Document] = []
            seen_texts = set()
            for doc in documents:
                text = doc.page_content.strip()
                if len(text) >= self.MIN_LENGTH and text not in seen_texts:
                    filtered_docs.append(doc)
                    seen_texts.add(text)

            if not filtered_docs:
                self.logger.warning("No valid documents after filtering")
                return []

            # Реранк с нативным Reranker
            texts = [doc.page_content for doc in filtered_docs]
            reranked_tuples = self.reranker.rerank(query, texts)  # type: ignore

            # Маппинг обратно на Document, сохраняем порядок и добавляем rerank_score
            text_to_doc: Dict[str, Document] = {doc.page_content: doc for doc in filtered_docs}
            sorted_docs: List[Document] = []
            for text, score in reranked_tuples:  # type: ignore
                doc = text_to_doc.get(text)
                if doc:
                    metadata = dict(doc.metadata)
                    metadata["rerank_score"] = str(score)  # безопасно сохраняем str
                    sorted_docs.append(Document(page_content=doc.page_content, metadata=metadata))

            self.logger.info(
                f"Reranked {len(sorted_docs)} documents | "
                f"sample: {[d.page_content[:80] for d in sorted_docs[:3]]}"
            )

            return sorted_docs

        except Exception as e:
            self.logger.error(f"Error in LCRerankerRunnable.invoke(): {e}", exc_info=True)
            return []
