import time
from src.utils.db_connector import DBConnector
from src.utils.logger_loader import LoggerLoader
from src.data_processing.subtitle_extractor import SubtitleExtractor
from src.data_processing.subtitle_manager import SubtitleManager
from src.search.retriever import Retriever
from src.utils.config_loader import ConfigLoader
from src.utils.prompt_loader import PromptLoader
from src.answer_generator.model_factory import model_factory


class RAGModel:
    def __init__(self, db_connector: DBConnector):
        """
        Инициализация RAGModel с переданным DBConnector и пулом соединений.
        Используется для извлечения субтитров, поиска и генерации ответа.
        """
        self.db_connector = db_connector  # Пул соединений передается через конструктор
        self.subtitle_extractor = SubtitleExtractor()
        self.subtitle_manager = SubtitleManager(db_pool=self.db_connector)
        self.retriever = Retriever(db_pool=self.db_connector)
        self.logger = LoggerLoader.get_logger()

        # Загружаем конфиг и инициализируем LLM модель через фабрику
        config = ConfigLoader.get_config()
        self.llm_model = model_factory(config)

        # Загружаем промпт по языку
        self.language = config.get("language", "ru")
        self.prompt_template = PromptLoader().load(self.language)

        self.logger.info(f"Модель {self.llm_model} инициализирована для языка: {self.language}")

    def process_query(self, video_url: str, query: str) -> str:
        """
        Обрабатывает запрос, проверяя наличие субтитров в базе и возвращая контекст для генерации ответа.
        """
        try:
            # Извлечение ID видео из URL
            video_id = self.subtitle_extractor.extract_video_id(video_url)
            if not video_id:
                self.logger.error(f"Не удалось извлечь video_id из URL: {video_url}")
                return "Ошибка: не удалось извлечь video_id из URL."

            self.logger.info(f"Видео ID: {video_id}")

            # Проверка наличия субтитров в базе данных
            subtitles = self.db_connector.fetch_subtitles(video_id)
            if not subtitles:
                self.logger.info(f"Субтитры для видео {video_id} не найдены. Извлекаем...")
                subtitles = self.subtitle_extractor.get_subtitles(video_id)

                if subtitles:
                    self.subtitle_manager.add_subtitles(video_id, subtitles)
                    self.logger.info(f"Субтитры для видео {video_id} успешно добавлены в базу.")
                else:
                    self.logger.error(f"Не удалось извлечь субтитры для видео {video_id}.")
                    return "Ошибка: не удалось извлечь субтитры для видео."

            # Поиск релевантных фрагментов субтитров
            results = self.retriever.retrieve(query)
            if not results:
                self.logger.warning(f"По запросу '{query}' не найдено похожих субтитров.")
                return "По запросу не найдено похожих субтитров."

            context = "\n".join([text for text, _ in results])

            # Генерация ответа с использованием LLM
            return self._generate_answer(query, context)

        except Exception as e:
            self.logger.error(f"Ошибка при обработке запроса: {e}")
            return "Ошибка: не удалось обработать запрос."

    def _generate_answer(self, query: str, context: str) -> str:
        """
        Генерирует ответ с использованием LLM на основе шаблона промпта.
        """
        start_time = time.time()
        try:
            prompt = self.prompt_template.format(query=query, context=context)
            self.logger.info(f"Сформированный промпт: {prompt[:500]}...")  # Логируем только первые 500 символов

            # Генерация ответа
            answer = self.llm_model.generate(prompt, max_length=1024)
            self.logger.info(f"Ответ сгенерирован за {time.time() - start_time:.2f} секунд.")
            return answer.strip()

        except Exception as error:
            self.logger.error(f"Ошибка при генерации ответа с LLM: {error}")
            return "Ошибка: не удалось сгенерировать ответ."
