from src.utils.logger_loader import LoggerLoader
from src.utils.db_connector import DBConnector
from src.answer_generator.rag_model import RAGModel


def clear_table() -> None:
    """Очистить таблицу субтитров вручную."""
    db = DBConnector()
    db.clear_table()
    db.close()


def main() -> None:
    """Основная точка входа в RAG-пайплайн."""
    logger = LoggerLoader.get_logger()
    logger.info("Запуск пайплайна RAG")

    db_connector = None

    try:
        # Инициализация пула соединений
        db_connector = DBConnector()

        # Инициализация RAG-модели с доступом к БД
        rag_model = RAGModel(db_connector=db_connector)

        # Пример входных данных
        video_url = "https://www.youtube.com/watch?v=zX6Ml0DM0LM&ab_channel=%D0%9A%D0%98%D0%9D%D0%9E%D0%9B%D0%98%D0%9A%D0%91%D0%95%D0%97KINOLIKBEZ"
        query = "Ответь кратко, о чем фильм Линча Дикие Сердцем?"

        # Запрос к модели
        answer = rag_model.process_query(video_url, query)

        print("\nОтвет от модели:")
        print(answer)

    except Exception as e:
        logger.exception(f"Ошибка во время выполнения пайплайна: {e}")

    finally:
        if db_connector:
            db_connector.close()


if __name__ == "__main__":
    main()
