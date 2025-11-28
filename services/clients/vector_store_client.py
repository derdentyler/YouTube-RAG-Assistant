"""Client for Vector Store service."""
from typing import List, Dict, Any
from services.clients.base_client import BaseServiceClient
from services.contracts.vector_store_service import SearchRequest, SearchResponse


class VectorStoreClient(BaseServiceClient):
    """Client for communicating with Vector Store service."""
    
    def __init__(self, base_url: str = "http://vector-store-service:8004"):
        super().__init__(base_url)
    
    async def search(self, embedding: List[float], top_k: int = 5, video_id: str = None) -> List[Dict[str, Any]]:
        """Search vectors using Vector Store service."""
        request = SearchRequest(embedding=embedding, top_k=top_k, video_id=video_id)
        response_data = await self.post("/search", request.dict())
        response = SearchResponse(**response_data)
        return response.documents

