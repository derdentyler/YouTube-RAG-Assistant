"""Integration tests for microservices."""
import pytest
from services.clients.llm_client import LLMClient
from services.clients.embedder_client import EmbedderClient
from services.clients.reranker_client import RerankerClient
from services.clients.vector_store_client import VectorStoreClient


@pytest.mark.asyncio
async def test_llm_service_integration():
    """Test LLM service client (requires service to be running)."""
    client = LLMClient(base_url="http://localhost:8001")
    try:
        response = await client.generate("Test prompt", max_tokens=50)
        assert isinstance(response, str)
        assert len(response) > 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_embedder_service_integration():
    """Test Embedder service client (requires service to be running)."""
    client = EmbedderClient(base_url="http://localhost:8002")
    try:
        embeddings = await client.embed(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768  # dimension
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_reranker_service_integration():
    """Test Reranker service client (requires service to be running)."""
    client = RerankerClient(base_url="http://localhost:8003")
    try:
        documents = ["doc1", "doc2", "doc3"]
        ranked = await client.rerank("test query", documents, top_k=2)
        assert len(ranked) <= 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_vector_store_service_integration():
    """Test Vector Store service client (requires service to be running)."""
    client = VectorStoreClient(base_url="http://localhost:8004")
    try:
        # Create dummy embedding
        embedding = [0.1] * 768
        documents = await client.search(embedding, top_k=5)
        assert isinstance(documents, list)
    finally:
        await client.close()

