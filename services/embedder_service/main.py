"""Embedder Service - handles text embeddings."""
import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from sentence_transformers import SentenceTransformer
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from services.contracts.embedder_service import EmbedRequest, EmbedResponse
from services.shared.metrics import (
    request_count,
    request_duration,
    error_count,
    embedder_batch_size,
    embedder_embedding_duration
)

logger = LoggerLoader.get_logger()

# Global model instance
model = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding model at startup."""
    global model, config
    logger.info("Loading embedding model...")
    config = ConfigLoader.get_config()
    model = SentenceTransformer(config.embedding_model)
    logger.info(f"Embedding model loaded: {config.embedding_model}")
    yield
    logger.info("Embedder service shutting down...")


app = FastAPI(title="Embedder Service", lifespan=lifespan)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings for texts."""
    start_time = time.time()
    try:
        if model is None:
            error_count.labels(service='embedder', endpoint='/embed', error_type='not_loaded').inc()
            raise HTTPException(status_code=503, detail="Embedding model not loaded")
        
        # Record batch size
        batch_size = len(request.texts)
        embedder_batch_size.labels(model=config.embedding_model).observe(batch_size)
        
        # Generate embeddings
        embedding_start = time.time()
        embeddings = model.encode(
            request.texts,
            normalize_embeddings=request.normalize,
            convert_to_numpy=True
        )
        embedding_duration = time.time() - embedding_start
        
        # Record metrics
        request_count.labels(service='embedder', endpoint='/embed', status='200').inc()
        request_duration.labels(service='embedder', endpoint='/embed').observe(time.time() - start_time)
        embedder_embedding_duration.labels(model=config.embedding_model, batch_size=str(batch_size)).observe(embedding_duration)
        
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model=config.embedding_model,
            dimension=config.embedding_dimension
        )
    except HTTPException:
        raise
    except Exception as e:
        error_count.labels(service='embedder', endpoint='/embed', error_type=type(e).__name__).inc()
        request_count.labels(service='embedder', endpoint='/embed', status='500').inc()
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "not_ready",
        "service": "embedder"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

