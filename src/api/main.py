from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Annotated, Optional
from contextlib import asynccontextmanager
from src.answer_generator.rag_model import RAGModel
from src.utils.logger_loader import LoggerLoader
from src.core.dependencies.providers import RAGModelDep, ConfigDep
from src.core.dependencies.container import get_container, reset_container
import uvicorn
from dotenv import load_dotenv
import os

# Переменные .env
load_dotenv()

# Логгер
logger = LoggerLoader.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Startup
    logger.info("Starting application...")
    container = get_container()
    logger.info("Dependency container initialized")
    
    # Предзагрузка тяжелых объектов
    logger.info("Preloading heavy models...")
    _ = container.get_rag_model()  # Загружает все зависимости
    logger.info("All dependencies loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    reset_container()
    logger.info("Application shutdown complete")


# FastAPI приложение
app: FastAPI = FastAPI(
    title="RAG API",
    description="API для обработки запросов с помощью Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan  # Используем lifespan вместо on_event
)


# ----- Pydantic схемы -----
class QueryRequest(BaseModel):
    video_url: Annotated[str, "URL видео"]
    query: Annotated[str, "Вопрос к видео"]


class QueryResponse(BaseModel):
    answer: str
    context: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    config_language: str
    models_loaded: bool


# ----- Роуты -----
@app.get("/health", response_model=HealthResponse)
def health_check(config: ConfigDep) -> HealthResponse:
    """Health check с информацией о конфигурации."""
    logger.info("Получен запрос на /health")
    container = get_container()
    models_loaded = container._rag_model is not None
    
    return HealthResponse(
        status="ok",
        config_language=config.language,
        models_loaded=models_loaded
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(
    request: QueryRequest,
    rag_model: RAGModelDep  # Внедрение через Depends
) -> QueryResponse:
    """Обработка запроса с использованием RAG модели."""
    try:
        logger.info(f"Запрос получен: video_url='{request.video_url}', query='{request.query}'")
        answer: str = rag_model.process_query(request.video_url, request.query)
        logger.info(f"Ответ сгенерирован (обрезка до 500 символов): {answer[:500]}...")
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса")

# ----- Запуск сервера -----
if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 8000))
    logger.info(f"Запуск сервера на http://127.0.0.1:{port}")
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=port, reload=True)
