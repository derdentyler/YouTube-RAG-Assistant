"""Base HTTP client for microservices communication."""
import httpx
from typing import Optional
from src.utils.logger_loader import LoggerLoader


class BaseServiceClient:
    """Base HTTP client for microservices communication."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.logger = LoggerLoader.get_logger()
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def post(self, endpoint: str, data: dict) -> dict:
        """POST request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = await self.client.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error calling {url}: {e}")
            raise
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

