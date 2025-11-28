"""API contract for Embedder service."""
from pydantic import BaseModel
from typing import List


class EmbedRequest(BaseModel):
    """Request for text embedding."""
    texts: List[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    """Response from embedding generation."""
    embeddings: List[List[float]]
    model: str
    dimension: int

