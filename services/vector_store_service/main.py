"""Vector Store Service - handles vector search operations."""
import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from src.utils.db_connector import DBConnector
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from services.contracts.vector_store_service import SearchRequest, SearchResponse
from services.shared.metrics import (
    request_count,
    request_duration,
    error_count,
    retriever_documents_found,
    retriever_search_duration
)

logger = LoggerLoader.get_logger()

# Global DB connector instance
db = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database connection at startup."""
    global db, config
    logger.info("Initializing Vector Store service...")
    config = ConfigLoader.get_config()
    db = DBConnector(embedding_dimension=config.embedding_dimension)
    logger.info("Vector Store service initialized")
    yield
    if db:
        db.close()
    logger.info("Vector Store service shutting down...")


app = FastAPI(title="Vector Store Service", lifespan=lifespan)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search vectors using pre-computed embedding."""
    start_time = time.time()
    try:
        if db is None:
            error_count.labels(service='vector_store', endpoint='/search', error_type='not_initialized').inc()
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        # Search using pre-computed embedding
        search_start = time.time()
        results = db.search_similar_embeddings(request.embedding, top_k=request.top_k)
        search_duration = time.time() - search_start
        
        # Record metrics
        num_documents = len(results)
        retriever_documents_found.labels(video_id=request.video_id or 'unknown').observe(num_documents)
        retriever_search_duration.observe(search_duration)
        
        # Format results
        documents = []
        for text, score in results:
            documents.append({
                "text": text,
                "score": float(score),
                "page_content": text  # For compatibility
            })
        
        request_count.labels(service='vector_store', endpoint='/search', status='200').inc()
        request_duration.labels(service='vector_store', endpoint='/search').observe(time.time() - start_time)
        
        return SearchResponse(documents=documents)
    except HTTPException:
        raise
    except Exception as e:
        error_count.labels(service='vector_store', endpoint='/search', error_type=type(e).__name__).inc()
        request_count.labels(service='vector_store', endpoint='/search', status='500').inc()
        logger.error(f"Error searching vectors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if db is not None else "not_ready",
        "service": "vector_store"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

