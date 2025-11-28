"""Reranker Service - handles document reranking."""
import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from src.reranker.reranker import Reranker
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from sentence_transformers import SentenceTransformer
from services.contracts.reranker_service import RerankerRequest, RerankerResponse
from services.shared.metrics import (
    request_count,
    request_duration,
    error_count,
    reranker_documents_reranked,
    reranker_rerank_duration
)

logger = LoggerLoader.get_logger()

# Global reranker instance
reranker = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load reranker model at startup."""
    global reranker, config
    logger.info("Loading reranker model...")
    config = ConfigLoader.get_config()
    
    if not config.reranker.use_reranker:
        logger.warning("Reranker is disabled in config")
        yield
        return
    
    embedder = SentenceTransformer(config.embedding_model)
    reranker = Reranker(config.reranker.model_path, embedder=embedder)
    logger.info(f"Reranker model loaded: {config.reranker.model_path}")
    yield
    logger.info("Reranker service shutting down...")


app = FastAPI(title="Reranker Service", lifespan=lifespan)


@app.post("/rerank", response_model=RerankerResponse)
async def rerank(request: RerankerRequest):
    """Rerank documents using reranker model."""
    start_time = time.time()
    try:
        from services.contracts.reranker_service import RankedDocument
        
        if reranker is None:
            # If reranker is disabled, return documents as-is with score 1.0
            ranked = [RankedDocument(text=doc, score=1.0) for doc in request.documents[:request.top_k]]
            request_count.labels(service='reranker', endpoint='/rerank', status='200').inc()
            request_duration.labels(service='reranker', endpoint='/rerank').observe(time.time() - start_time)
            return RerankerResponse(ranked_documents=ranked)
        
        # Record number of documents to rerank
        reranker_documents_reranked.observe(len(request.documents))
        
        # Rerank documents
        rerank_start = time.time()
        ranked = reranker.rerank(request.query, request.documents)
        rerank_duration = time.time() - rerank_start
        
        top_results = ranked[:request.top_k]
        # Convert tuples to RankedDocument objects
        ranked_docs = [RankedDocument(text=text, score=score) for text, score in top_results]
        
        # Record metrics
        request_count.labels(service='reranker', endpoint='/rerank', status='200').inc()
        request_duration.labels(service='reranker', endpoint='/rerank').observe(time.time() - start_time)
        reranker_rerank_duration.observe(rerank_duration)
        
        return RerankerResponse(ranked_documents=ranked_docs)
    except HTTPException:
        raise
    except Exception as e:
        error_count.labels(service='reranker', endpoint='/rerank', error_type=type(e).__name__).inc()
        request_count.labels(service='reranker', endpoint='/rerank', status='500').inc()
        logger.error(f"Error reranking documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if reranker is not None else "not_ready",
        "service": "reranker"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

