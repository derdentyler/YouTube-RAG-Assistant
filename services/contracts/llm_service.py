"""API contract for LLM service."""
from pydantic import BaseModel
from typing import Optional


class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    stop_sequences: Optional[list[str]] = None


class GenerateResponse(BaseModel):
    """Response from text generation."""
    text: str
    tokens_used: int
    model: str

