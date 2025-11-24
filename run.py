from src.utils.logger_loader import LoggerLoader
from src.core.dependencies.container import get_container, reset_container


def clear_table() -> None:
    """Очистить таблицу субтитров вручную."""
    container = get_container()
    try:
        db = container.get_db_connector()
        db.clear_table()
    finally:
        reset_container()


def main() -> None:
    """Основная точка входа в RAG-пайплайн."""
    logger = LoggerLoader.get_logger()
    logger.info("Запуск пайплайна RAG")
    
    container = None
    
    try:
        # Получаем контейнер с зависимостями
        container = get_container()
        
        # Инициализация RAG-модели через контейнер
        rag_model = container.get_rag_model()
        
        # Пример входных данных
        video_url = "https://www.youtube.com/watch?v=zX6Ml0DM0LM"
        query = "Ответь кратко, о чем фильм Линча Дикие Сердцем?"
        
        # Запрос к модели
        answer = rag_model.process_query(video_url, query)
        
        print("\nОтвет от модели:")
        print(answer)
    
    except Exception as e:
        logger.exception(f"Ошибка во время выполнения пайплайна: {e}")
    
    finally:
        if container:
            reset_container()


if __name__ == "__main__":
    main()
