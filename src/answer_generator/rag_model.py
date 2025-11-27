import time
from typing import Optional
from sentence_transformers import SentenceTransformer
from src.core.abstractions.embeddings import Embedder
from src.core.abstractions.llm import BaseLLM
from src.core.config.models import AppConfig
from src.core.abstractions.storage import StorageBackend
from src.utils.db_connector import DBConnector
from src.utils.logger_loader import LoggerLoader
from src.data_processing.subtitle_extractor import SubtitleExtractor
from src.data_processing.subtitle_manager import SubtitleManager
from src.utils.config_loader import ConfigLoader
from src.utils.prompt_loader import PromptLoader
from src.answer_generator.model_factory import model_factory
from src.reranker.reranker import Reranker
from src.core.adapters.db_vector_store import DBVectorStore


class RAGModel:
    """
    Retrieval-Augmented Generation.
    Поддерживает два режима:
      - use_langchain=True: весь пайплайн внутри LangChainRAG
      - use_langchain=False: нативная реализация с опциональным реранкером

    Все тяжёлые объекты (LLM, VectorStore, Embedder) создаются один раз
    и переиспользуются для обоих режимов.
    
    Поддерживает внедрение зависимостей для улучшения тестируемости
    и оптимизации загрузки моделей.
    """

    def __init__(
        self,
        db_connector: DBConnector,
        embedding_model: Optional[Embedder] = None,
        llm: Optional[BaseLLM] = None,
        config: Optional[AppConfig] = None,
        storage: Optional[StorageBackend] = None,
    ):
        self.storage = storage
        self.db = db_connector
        self.logger = LoggerLoader.get_logger()

        # Загружаем конфиг
        self.config = config or ConfigLoader.get_config()
        self.language = self.config.language
        self.use_langchain = self.config.use_langchain
        self.use_reranker = self.config.reranker.use_reranker

        # --- Общие компоненты (с возможностью инъекции) ---
        if embedding_model is None:
            embed_name = self.config.embedding_model
            self.embedding_model: Embedder = SentenceTransformer(embed_name)
            self.logger.info(f"Loaded embedding model: {embed_name}")
        else:
            self.embedding_model = embedding_model
            self.logger.info("Using injected embedding model")

        # LLM грузим один раз (с возможностью инъекции)
        if llm is None:
            self.llm = model_factory(self.config)
            self.logger.info(f"Loaded LLM for language: {self.language}")
        else:
            self.llm = llm
            self.logger.info("Using injected LLM")

        # Векторное хранилище общее для всех режимов
        self.vectorstore = DBVectorStore(
            db_connector=self.db,
            embedding_model=self.embedding_model
        )

        # Модули для работы с субтитрами
        # Передаем embedding_model для semantic chunking
        self.subtitle_extractor = SubtitleExtractor(
            embedding_model=self.embedding_model,
            storage=self.storage
        )
        self.subtitle_manager = SubtitleManager(
            db_pool=self.db,
            embedding_model=self.embedding_model
        )

        # --- Выбор пайплайна ---
        if self.use_langchain:
            from src.integrations.langchain_integration import LangChainRAG
            self.pipeline = LangChainRAG(
                db_connector=self.db,
                embedder=self.embedding_model,
                vectorstore=self.vectorstore,
                llm=self.llm,
                config=self.config
            )
        else:
            # Нативный RAG
            self.prompt_template = PromptLoader().load(self.language)
            self.retriever_top_k = self.config.retriever.top_k

            self.reranker = None
            self.reranker_top_k = None
            if self.use_reranker:
                self.reranker = Reranker(
                    self.config.reranker.model_path,
                    embedder=self.embedding_model
                )
                self.reranker_top_k = self.config.reranker.top_k

        self.logger.info(
            f"Initialized RAGModel | langchain={self.use_langchain} | reranker={self.use_reranker}"
        )

    def _ensure_subtitles(self, video_id: str) -> None:
        """Проверка наличия субтитров и извлечение при отсутствии."""
        if not self.db.fetch_subtitles(video_id):
            self.logger.info(f"Subtitles missing for {video_id}, extracting...")
            extracted = self.subtitle_extractor.get_subtitles(video_id)
            if not extracted:
                raise ValueError("Subtitles not found")
            self.subtitle_manager.add_subtitles(video_id, extracted)
            self.logger.info(f"Subtitles extracted and stored for {video_id}")

    def process_query(self, video_url: str, query: str) -> str:
        """Основной метод: выбор подхода и генерация ответа."""
        try:
            video_id = self.subtitle_extractor.extract_video_id(video_url)
            if not video_id:
                self.logger.error(f"Invalid video URL: {video_url}")
                return "Ошибка: некорректный URL видео."

            try:
                self._ensure_subtitles(video_id)
            except ValueError:
                return "Ошибка: субтитры не найдены."

            if self.use_langchain:
                result = self.pipeline.invoke(query)

                # Приводим результат к строке
                if isinstance(result, str):
                    return result
                elif isinstance(result, list):
                    return "\n".join(
                        [str(item) for item in result if isinstance(item, (str, dict))]
                    )
                elif isinstance(result, dict):
                    return str(result)
                else:
                    return "Ошибка: непредвиденный формат ответа от LangChain."

            else:
                # --- Нативный RAG ---
                docs = self.vectorstore.search(query, k=self.retriever_top_k)
                self.logger.info(f"Retrieved {len(docs)} candidates")
                if not docs:
                    return "По запросу не найдено похожих субтитров."

                texts = [d["page_content"] for d in docs]

                if self.use_reranker and self.reranker:
                    reranked = self.reranker.rerank(query, texts)
                    selected = [t for t, _ in reranked[: self.reranker_top_k]]
                else:
                    selected = texts[: self.retriever_top_k]

                snippets_str = "\n".join(
                    f"\t{i}.\t{snippet}" for i, snippet in enumerate(selected, 1)
                )
                self.logger.info(
                    f"Selected {len(selected)} snippets for context:\n{snippets_str}"
                )

                context = "\n".join(selected)
                prompt = self.prompt_template.format(query=query, context=context)
                return self._generate_answer(prompt)

        except Exception as e:
            self.logger.error(f"process_query error: {e}", exc_info=True)
            return "Ошибка: не удалось обработать запрос."

    def _generate_answer(self, prompt: str) -> str:
        """Генерация ответа через LLM."""
        start = time.time()
        answer = self.llm.generate(prompt, max_length=1024)
        elapsed = time.time() - start
        self.logger.info(f"Answer generated in {elapsed:.2f}s")
        return answer.strip()
