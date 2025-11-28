"""Client for Embedder service."""
from typing import List
from services.clients.base_client import BaseServiceClient
from services.contracts.embedder_service import EmbedRequest, EmbedResponse


class EmbedderClient(BaseServiceClient):
    """Client for communicating with Embedder service."""
    
    def __init__(self, base_url: str = "http://embedder-service:8002"):
        super().__init__(base_url)
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Embedder service."""
        request = EmbedRequest(texts=texts)
        response_data = await self.post("/embed", request.dict())
        response = EmbedResponse(**response_data)
        return response.embeddings

