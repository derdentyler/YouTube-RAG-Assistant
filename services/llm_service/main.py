"""LLM Service - handles text generation."""
import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from src.answer_generator.model_factory import model_factory
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from services.contracts.llm_service import GenerateRequest, GenerateResponse
from services.shared.metrics import (
    request_count,
    request_duration,
    error_count,
    llm_tokens_generated,
    llm_generation_duration
)

logger = LoggerLoader.get_logger()

# Global model instance
llm = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    global llm, config
    logger.info("Loading LLM model...")
    config = ConfigLoader.get_config()
    llm = model_factory(config)
    logger.info(f"LLM model loaded: {config.models[config.language]}")
    yield
    logger.info("LLM service shutting down...")


app = FastAPI(title="LLM Service", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using LLM."""
    start_time = time.time()
    try:
        if llm is None:
            error_count.labels(service='llm', endpoint='/generate', error_type='not_loaded').inc()
            raise HTTPException(status_code=503, detail="LLM model not loaded")
        
        # Generate text
        generation_start = time.time()
        text = llm.generate(request.prompt, max_length=request.max_tokens)
        generation_duration = time.time() - generation_start
        
        model_config = config.models[config.language]
        model_path = getattr(model_config, 'model_path', None) or getattr(model_config, 'model_name', 'unknown')
        tokens_used = len(text.split())  # Approximate
        
        # Record metrics
        request_count.labels(service='llm', endpoint='/generate', status='200').inc()
        request_duration.labels(service='llm', endpoint='/generate').observe(time.time() - start_time)
        llm_tokens_generated.labels(model=str(model_path), language=config.language).inc(tokens_used)
        llm_generation_duration.labels(model=str(model_path)).observe(generation_duration)
        
        return GenerateResponse(
            text=text,
            tokens_used=tokens_used,
            model=str(model_path)
        )
    except HTTPException:
        raise
    except Exception as e:
        error_count.labels(service='llm', endpoint='/generate', error_type=type(e).__name__).inc()
        request_count.labels(service='llm', endpoint='/generate', status='500').inc()
        logger.error(f"Error generating text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if llm is not None else "not_ready",
        "service": "llm"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

