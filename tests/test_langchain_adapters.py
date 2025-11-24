import pytest
from langchain_core.documents import Document

# ===========================
# Моки для тестов
# ===========================
class DummyEmbedder:
    def embed_documents(self, texts):
        self.called = getattr(self, "called", False)
        self.called = True
        return [[0.0] * 10 for _ in texts]

class DummyReranker:
    def __init__(self):
        self.called = False

    def rerank(self, query, docs):
        self.called = True
        return docs

class DummyLLM:
    def generate(self, prompt):
        return "generated answer"

class DummyVectorStoreWithSearch:
    def __init__(self, return_empty=False):
        self.return_empty = return_empty
        self.last_k = None

    def search(self, query, k):
        self.last_k = k
        if self.return_empty:
            return []
        return [Document(page_content=f"Doc {i}") for i in range(k)]

# ===========================
# Мини-версии адаптеров и RAG
# ===========================
class LCRerankerRunnable:
    def __init__(self, reranker):
        self.reranker = reranker

    def invoke(self, inputs):
        query = inputs["query"]
        docs = inputs["documents"]
        return self.reranker.rerank(query, docs)

class LangChainRAG:
    def __init__(self, vectorstore, llm, reranker=None, config=None, db_connector=None, embedder=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.reranker = reranker
        self.config = config or {}
        self.embedder = embedder

    def invoke(self, query):
        try:
            docs = self.vectorstore.search(query, k=self.config.get("top_k", 3))
            if self.embedder:
                _ = self.embedder.embed_documents([d.page_content for d in docs])
            if self.reranker and self.config.get("reranker", {}).get("use_reranker", False):
                docs = self.reranker.rerank(query, docs)
            return self.llm.generate("prompt")
        except Exception:
            return "Ошибка: не удалось обработать запрос."

# ===========================
# Тесты
# ===========================
def test_embedder_called_and_returns_correct_format():
    embedder = DummyEmbedder()
    texts = ["text1", "text2"]
    embeddings = embedder.embed_documents(texts)
    assert embedder.called is True
    assert isinstance(embeddings, list)
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(len(vec) == 10 for vec in embeddings)

def test_top_k_propagation():
    vectorstore = DummyVectorStoreWithSearch()
    llm = DummyLLM()
    rag = LangChainRAG(vectorstore=vectorstore, llm=llm, config={"top_k": 5})
    _ = rag.invoke("query")
    assert vectorstore.last_k == 5

def test_pipeline_with_empty_results():
    vectorstore = DummyVectorStoreWithSearch(return_empty=True)
    llm = DummyLLM()
    rag = LangChainRAG(vectorstore=vectorstore, llm=llm)
    answer = rag.invoke("query")
    assert answer == "generated answer"

def test_reranker_invoked_conditionally():
    vectorstore = DummyVectorStoreWithSearch()
    reranker = DummyReranker()
    llm = DummyLLM()

    # use_reranker=True
    rag = LangChainRAG(vectorstore=vectorstore, llm=llm, reranker=reranker,
                        config={"reranker": {"use_reranker": True}})
    _ = rag.invoke("query")
    assert reranker.called is True

    # use_reranker=False
    reranker.called = False
    rag_no = LangChainRAG(vectorstore=vectorstore, llm=llm, reranker=reranker,
                           config={"reranker": {"use_reranker": False}})
    _ = rag_no.invoke("query")
    assert reranker.called is False
