"""Orchestrator Service - main API that coordinates all microservices."""
import time
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Annotated, Optional
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from services.orchestrator.microservices_rag import MicroservicesRAGOrchestrator
from src.utils.logger_loader import LoggerLoader
from src.utils.config_loader import ConfigLoader
from services.shared.metrics import (
    request_count,
    request_duration,
    error_count,
    rag_query_duration,
    rag_context_length
)
import os

logger = LoggerLoader.get_logger()
config = ConfigLoader.get_config()

# Global orchestrator instance
orchestrator: Optional[MicroservicesRAGOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize orchestrator at startup."""
    global orchestrator
    logger.info("Initializing Orchestrator service...")
    orchestrator = MicroservicesRAGOrchestrator(
        llm_service_url=os.getenv("LLM_SERVICE_URL"),
        embedder_service_url=os.getenv("EMBEDDER_SERVICE_URL"),
        reranker_service_url=os.getenv("RERANKER_SERVICE_URL"),
        vector_store_service_url=os.getenv("VECTOR_STORE_SERVICE_URL")
    )
    logger.info("Orchestrator service initialized")
    yield
    if orchestrator:
        await orchestrator.close()
    logger.info("Orchestrator service shutting down...")


app = FastAPI(
    title="RAG API (Microservices)",
    description="API for processing queries using Retrieval-Augmented Generation with microservices architecture",
    version="2.0.0",
    lifespan=lifespan
)


# Pydantic schemas
class QueryRequest(BaseModel):
    """Request for query processing."""
    video_url: str
    query: str


class QueryResponse(BaseModel):
    """Response from query processing."""
    answer: str
    context: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    services_status: dict


def get_orchestrator() -> MicroservicesRAGOrchestrator:
    """Dependency to get orchestrator instance."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services_status = {}
    
    # Check each service
    try:
        # This would ideally check each service's health endpoint
        services_status = {
            "llm": "unknown",
            "embedder": "unknown",
            "reranker": "unknown",
            "vector_store": "unknown"
        }
    except Exception as e:
        logger.warning(f"Error checking services: {e}")
    
    return HealthResponse(
        status="healthy" if orchestrator is not None else "not_ready",
        service="orchestrator",
        services_status=services_status
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    rag: Annotated[MicroservicesRAGOrchestrator, Depends(get_orchestrator)]
) -> QueryResponse:
    """Process query using microservices RAG."""
    start_time = time.time()
    try:
        logger.info(f"Query received: video_url='{request.video_url}', query='{request.query}'")
        answer = await rag.process_query(request.video_url, request.query)
        
        # Record metrics
        duration = time.time() - start_time
        has_reranker = "true" if config.reranker.use_reranker else "false"
        request_count.labels(service='orchestrator', endpoint='/query', status='200').inc()
        request_duration.labels(service='orchestrator', endpoint='/query').observe(duration)
        rag_query_duration.labels(has_reranker=has_reranker).observe(duration)
        # Approximate context length (rough estimate)
        context_tokens = len(answer.split()) * 2  # Rough estimate
        rag_context_length.observe(context_tokens)
        
        logger.info(f"Answer generated (truncated): {answer[:500]}...")
        return QueryResponse(answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        error_count.labels(service='orchestrator', endpoint='/query', error_type=type(e).__name__).inc()
        request_count.labels(service='orchestrator', endpoint='/query', status='500').inc()
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing query")

