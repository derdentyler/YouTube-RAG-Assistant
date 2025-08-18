# src/integrations/langchain_integration.py

from typing import List, Optional
from langchain_core.documents import Document
from src.utils.logger_loader import LoggerLoader
from src.utils.db_connector import DBConnector
from src.core.abstractions.embeddings import Embedder
from src.core.adapters.db_vector_store import DBVectorStore
from src.reranker.reranker import Reranker
from src.utils.prompt_loader import PromptLoader

# Адаптеры LangChain
from src.integrations.lc_retriever_adapter import LCRetrieverAdapter
from src.integrations.lc_reranker_runnable import LCRerankerRunnable


class LangChainRAG:
    """
    LangChain-пайплайн: retrieve -> optional rerank -> generate.

    Все шаги работают с Document[]. Результат invoke() всегда str для FastAPI.
    """

    def __init__(
        self,
        db_connector: DBConnector,
        embedder: Embedder,
        vectorstore: DBVectorStore,
        llm,
        config: dict,
        use_reranker: Optional[bool] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.logger = LoggerLoader.get_logger()

        # Основные зависимости
        self.db = db_connector
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        self.config = config

        # Конфиги
        self.language = config.get("language", "ru")
        retr_cfg = config.get("retriever", {}) or {}
        rer_cfg = config.get("reranker", {}) or {}

        self.retriever_top_k = int(retr_cfg.get("top_k", 5))
        self.reranker_top_k = int(rer_cfg.get("top_k", 5))

        # use_reranker можно переопределить через аргумент конструктора
        self.use_reranker = bool(rer_cfg.get("use_reranker", False)) if use_reranker is None else bool(use_reranker)

        # Адаптеры LangChain
        self.retriever = LCRetrieverAdapter(vectorstore=self.vectorstore, top_k=self.retriever_top_k)
        self.reranker_adapter = LCRerankerRunnable(reranker=reranker) if self.use_reranker and reranker else None

        # Промпт
        self.prompt_template = PromptLoader().load(self.language)

        self.logger.info(
            f"LangChainRAG initialized | use_reranker={self.use_reranker} | "
            f"retriever_top_k={self.retriever_top_k} | reranker_top_k={self.reranker_top_k}"
        )

    def _build_prompt(self, query: str, docs: List[Document]) -> str:
        """Собираем контекст и формируем промпт под LLM."""
        context = "\n".join(d.page_content for d in docs if d.page_content)
        return self.prompt_template.format(query=query, context=context)

    def invoke(self, query: str) -> str:
        """
        Выполнить полный RAG-цикл и вернуть готовый ответ (str).

        - Retriever возвращает List[Document]
        - Reranker (если используется) возвращает List[Document]
        - Prompt формируется из page_content каждого Document
        """
        try:
            if not query.strip():
                return "Пустой запрос."

            # 1) Retrieve
            retrieved_docs: List[Document] = self.retriever.invoke(query)
            if not retrieved_docs:
                return "По запросу не найдено похожих субтитров."

            # 2) Optional rerank
            if self.reranker_adapter:
                rerank_input = {"query": query, "documents": retrieved_docs}
                reranked_docs: List[Document] = self.reranker_adapter.invoke(inputs=rerank_input)
            else:
                reranked_docs = retrieved_docs

            if not reranked_docs:
                return "Документы для генерации не найдены."

            # 3) Ограничиваем top-k для контекста перед генерацией
            docs_for_prompt = reranked_docs[: self.reranker_top_k]
            self.logger.info(
                f"Using top-{self.reranker_top_k} documents for prompt | "
                f"sample: {[d.page_content[:80] for d in docs_for_prompt]}"
            )

            # 4) Prompt + LLM.generate -> str
            prompt = self._build_prompt(query, docs_for_prompt)
            answer = self.llm.generate(prompt, max_length=1024).strip()
            return answer or "Ответ не удалось сгенерировать."

        except Exception as e:
            self.logger.error(f"LangChainRAG.invoke error: {e}", exc_info=True)
            return "Ошибка: не удалось обработать запрос."
