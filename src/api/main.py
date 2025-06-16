from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Annotated, Optional
from src.answer_generator.rag_model import RAGModel
from src.utils.logger_loader import LoggerLoader
from src.utils.db_connector import DBConnector
import uvicorn
from dotenv import load_dotenv
import os

# Переменные .env
load_dotenv()

# Логгер
logger = LoggerLoader.get_logger()

# FastAPI приложение
app: FastAPI = FastAPI(
    title="RAG API",
    description="API для обработки запросов с помощью Retrieval-Augmented Generation",
    version="1.0.0",
)

# Инициализация DB и RAG модели
db_connector: DBConnector = DBConnector()
rag_model: RAGModel = RAGModel(db_connector=db_connector)
logger.info("RAGModel успешно инициализирована с DBConnector")

# ----- Pydantic схемы с Annotated для Swagger UI -----
class QueryRequest(BaseModel):
    video_url: Annotated[str, "URL видео"]  # Используем аннотацию для добавления подсказки в Swagger
    query: Annotated[str, "Вопрос к видео"]  # Подсказка для запроса

class QueryResponse(BaseModel):
    answer: str
    context: Optional[str] = None  # Можно включать для отладки

# ----- Роуты -----
@app.get("/health")
def health_check() -> dict:
    logger.info("Получен запрос на /health")
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        logger.info(f"Запрос получен: video_url='{request.video_url}', query='{request.query}'")
        answer: str = rag_model.process_query(request.video_url, request.query)
        logger.info(f"Ответ сгенерирован (обрезка до 500 символов): {answer[:500]}...")
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса")

# ----- Завершение работы -----
@app.on_event("shutdown")
def shutdown_event() -> None:
    db_connector.close()
    logger.info("Пул соединений закрыт")

# ----- Запуск сервера -----
if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 8000))
    logger.info(f"Запуск сервера на http://127.0.0.1:{port}")
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=port, reload=True)
