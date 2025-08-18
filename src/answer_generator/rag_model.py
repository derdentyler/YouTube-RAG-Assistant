import time
from sentence_transformers import SentenceTransformer
from src.core.abstractions.embeddings import Embedder
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
    """

    def __init__(self, db_connector: DBConnector):
        self.db = db_connector
        self.logger = LoggerLoader.get_logger()

        # Загружаем конфиг
        self.config = ConfigLoader.get_config()
        self.language = self.config.get("language", "ru")
        self.use_langchain = self.config.get("use_langchain", False)
        self.use_reranker = self.config.get("reranker", {}).get("use_reranker", False)

        # --- Общие компоненты ---
        embed_name = self.config.get("embedding_model")
        self.embedding_model: Embedder = SentenceTransformer(embed_name)

        # LLM грузим один раз
        self.llm = model_factory(self.config)

        # Векторное хранилище общее для всех режимов
        self.vectorstore = DBVectorStore(
            db_connector=self.db,
            embedding_model=self.embedding_model
        )

        # Модули для работы с субтитрами
        self.subtitle_extractor = SubtitleExtractor()
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
            self.retriever_top_k = self.config.get("retriever", {}).get("top_k", 5)

            self.reranker = None
            self.reranker_top_k = None
            if self.use_reranker:
                rer_cfg = self.config.get("reranker", {})
                self.reranker = Reranker(
                    rer_cfg.get("model_path"),
                    embedder=self.embedding_model
                )
                self.reranker_top_k = rer_cfg.get("top_k", 5)

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
