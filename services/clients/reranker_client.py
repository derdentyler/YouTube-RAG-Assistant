"""Client for Reranker service."""
from typing import List, Tuple
from services.clients.base_client import BaseServiceClient
from services.contracts.reranker_service import RerankerRequest, RerankerResponse


class RerankerClient(BaseServiceClient):
    """Client for communicating with Reranker service."""
    
    def __init__(self, base_url: str = "http://reranker-service:8003"):
        super().__init__(base_url)
    
    async def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Rerank documents using Reranker service."""
        request = RerankerRequest(query=query, documents=documents, top_k=top_k)
        response_data = await self.post("/rerank", request.dict())
        response = RerankerResponse(**response_data)
        # Convert RankedDocument objects to tuples for compatibility
        return [(doc.text, doc.score) for doc in response.ranked_documents]

