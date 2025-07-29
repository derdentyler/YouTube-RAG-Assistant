import time
from typing import List, Tuple

from sentence_transformers import SentenceTransformer

from src.utils.db_connector import DBConnector
from src.utils.logger_loader import LoggerLoader
from src.data_processing.subtitle_extractor import SubtitleExtractor
from src.data_processing.subtitle_manager import SubtitleManager
from src.search.retriever import Retriever
from src.utils.config_loader import ConfigLoader
from src.utils.prompt_loader import PromptLoader
from src.answer_generator.model_factory import model_factory
from src.reranker.reranker import Reranker


class RAGModel:
    def __init__(self, db_connector: DBConnector):
        """
        Инициализация RAGModel с переданным DBConnector и пулом соединений.
        Используется для извлечения субтитров, поиска, (опционального) реранкинга и генерации ответа.
        """
        self.db_connector = db_connector  # Пул соединений
        self.subtitle_extractor = SubtitleExtractor()
        self.subtitle_manager = SubtitleManager(db_pool=self.db_connector)
        self.retriever = Retriever(db_pool=self.db_connector)
        self.logger = LoggerLoader.get_logger()

        # Конфиг и LLM
        config = ConfigLoader.get_config()
        self.llm_model = model_factory(config)
        self.language = config.get("language", "ru")
        self.prompt_template = PromptLoader().load(self.language)

        # Параметры reranking'а из конфига
        self.use_reranker: bool = config.get("reranker", {}).get("use_reranker", False)
        self.retriever_top_k: int = config.get("retriever", {}).get("top_k", 5)
        self.reranker_top_k: int = config.get("reranker", {}).get("top_k", 5)

        if self.use_reranker:
            # Загружаем модель эмбеддингов для reranker
            embed_model_name = config.get("embedding_model")
            self.embed_model = SentenceTransformer(embed_model_name)
            # Инициализируем reranker
            reranker_cfg = config.get("reranker", {})
            reranker_path = reranker_cfg.get("model_path")
            self.reranker = Reranker(reranker_path)

        self.logger.info(
            f"RAGModel initialized (use_reranker={self.use_reranker}"
        )

    def process_query(self, video_url: str, query: str) -> str:
        """
        Обрабатывает запрос, проверяя наличие субтитров в базе и возвращая контекст для генерации ответа.
        """
        try:
            # 1. Extract video ID
            video_id = self.subtitle_extractor.extract_video_id(video_url)
            if not video_id:
                self.logger.error(f"Cannot extract video_id from URL: {video_url}")
                return "Ошибка: некорректный URL видео."

            # 2. Ensure subtitles in DB
            subtitles = self.db_connector.fetch_subtitles(video_id)
            if not subtitles:
                self.logger.info(f"Субтитры для {video_id} не найдены, извлекаем...")
                subs = self.subtitle_extractor.get_subtitles(video_id)
                if not subs:
                    self.logger.error(f"Не удалось получить субтитры для {video_id}")
                    return "Ошибка: субтитры не найдены."
                self.subtitle_manager.add_subtitles(video_id, subs)
                subtitles = subs

            # 3. Retrieve candidates
            results: List[Tuple[str, float]] = self.retriever.retrieve(query)
            if not results:
                self.logger.warning(f"No subtitles found for query '{query}'")
                return "По запросу не найдено похожих субтитров."

            # 4. Rerank or baseline top_k
            if self.use_reranker:
                # распакуем тексты из результата retriever
                texts, _ = zip(*results)

                # 1) Получаем эмбеддинг запроса как np.ndarray
                q_emb = self.embed_model.encode(query, convert_to_tensor=False)
                # на всякий случай приводим Tensor -> numpy
                if hasattr(q_emb, "cpu"):
                    q_emb = q_emb.cpu().numpy()

                # 2) Получаем батч эмбеддингов документов в np.ndarray shape (N, D)
                d_embs_np = self.embed_model.encode(list(texts), convert_to_tensor=False)
                if hasattr(d_embs_np, "cpu"):
                    d_embs_np = d_embs_np.cpu().numpy()

                # 3) Конвертируем в список одномерных векторов
                doc_embeddings = [d_embs_np[i] for i in range(d_embs_np.shape[0])]

                # 4) Токенизация
                q_tokens = query.lower().split()
                d_tokens_list = [t.lower().split() for t in texts]

                # 5) Реранкинг
                reranked = self.reranker.rerank(
                    q_emb,
                    doc_embeddings,
                    q_tokens,
                    d_tokens_list,
                    list(texts)
                )
                selected = [t for t, _ in reranked[: self.reranker_top_k]]
            else:
                selected = [t for t, _ in results[: self.retriever_top_k]]

            context = "\n".join(selected)

            # 5. Generate answer
            prompt = self.prompt_template.format(query=query, context=context)
            self.logger.info(f"Prompt: {prompt[:200]}...")
            return self._generate_answer(query, context)

        except Exception as e:
            self.logger.error(f"Error in process_query: {e}")
            return "Ошибка: не удалось обработать запрос."

    def _generate_answer(self, query: str, context: str) -> str:
        """
        Генерирует ответ с использованием LLM на основе шаблона промпта.
        """
        start = time.time()
        try:
            prompt = self.prompt_template.format(query=query, context=context)
            self.logger.info(f"Сформированный промпт: {prompt[:500]}...")
            answer = self.llm_model.generate(prompt, max_length=1024)
            self.logger.info(f"Ответ сгенерирован за {time.time() - start:.2f} сек.")
            return answer.strip()
        except Exception as error:
            self.logger.error(f"Error in _generate_answer: {error}")
            return "Ошибка: не удалось сгенерировать ответ."
