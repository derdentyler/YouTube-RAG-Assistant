from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from src.utils.db_connector import DBConnector
from src.integrations.langchain_vectorstore import DBLangChainVectorStore


class LangChainRAG:
    """Wrapper for LangChain RAG pipeline using DB-backed VectorStore."""

    def __init__(self, db_connector: DBConnector, config: dict):
        self.db_connector = db_connector
        self.config = config

        # Initialize embedding model
        self.embedding = self._get_embedding_model()

        # Initialize VectorStore adapter
        self.vectorstore = DBLangChainVectorStore(
            db_connector=db_connector,
            embedding_model=self.embedding
        )

        # Build RAG chain
        self.chain = self._create_chain()

    def _get_embedding_model(self) -> Embeddings:
        """Instantiate embedding model for LangChain."""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"],
            encode_kwargs={"normalize_embeddings": True}
        )

    def _create_chain(self):
        """Create RAG chain: retrieve -> prompt -> LLM -> parse."""
        prompt = self._load_prompt_template()

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config["retriever"]["top_k"]}
        )

        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(self._format_prompt)
            | RunnableLambda(self._generate_answer)
            | StrOutputParser()
        )

    def _load_prompt_template(self) -> str:
        from src.utils.prompt_loader import PromptLoader
        return PromptLoader().load(self.config.get("language", "ru"))

    def _format_prompt(self, data: dict) -> str:
        return self._load_prompt_template().format(
            context=data["context"],
            question=data["question"]
        )

    def _generate_answer(self, prompt: str) -> str:
        from src.answer_generator.model_factory import model_factory
        llm = model_factory(self.config)
        return llm.generate(prompt, max_length=1024)

    def invoke(self, query: str) -> str:
        return self.chain.invoke(query)