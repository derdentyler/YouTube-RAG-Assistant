"""API contract for Reranker service."""
from pydantic import BaseModel
from typing import List


class RankedDocument(BaseModel):
    """Ranked document with score."""
    text: str
    score: float


class RerankerRequest(BaseModel):
    """Request for document reranking."""
    query: str
    documents: List[str]
    top_k: int = 3


class RerankerResponse(BaseModel):
    """Response from reranking."""
    ranked_documents: List[RankedDocument]

