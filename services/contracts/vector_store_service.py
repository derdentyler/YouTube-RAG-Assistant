"""API contract for Vector Store service."""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SearchRequest(BaseModel):
    """Request for vector search."""
    embedding: List[float]
    top_k: int = 5
    video_id: Optional[str] = None


class SearchResponse(BaseModel):
    """Response from vector search."""
    documents: List[Dict[str, Any]]  # [{"text": ..., "metadata": ...}]

