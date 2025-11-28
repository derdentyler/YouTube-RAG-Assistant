"""Client for LLM service."""
from services.clients.base_client import BaseServiceClient
from services.contracts.llm_service import GenerateRequest, GenerateResponse


class LLMClient(BaseServiceClient):
    """Client for communicating with LLM service."""
    
    def __init__(self, base_url: str = "http://llm-service:8001"):
        super().__init__(base_url)
    
    async def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text using LLM service."""
        request = GenerateRequest(prompt=prompt, max_tokens=max_tokens)
        response_data = await self.post("/generate", request.dict())
        response = GenerateResponse(**response_data)
        return response.text

