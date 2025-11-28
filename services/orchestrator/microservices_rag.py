"""Microservices-based RAG orchestrator."""
import os
from typing import Optional
from services.clients.llm_client import LLMClient
from services.clients.embedder_client import EmbedderClient
from services.clients.reranker_client import RerankerClient
from services.clients.vector_store_client import VectorStoreClient
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from src.utils.prompt_loader import PromptLoader
from src.data_processing.subtitle_extractor import SubtitleExtractor
from src.utils.db_connector import DBConnector


class MicroservicesRAGOrchestrator:
    """RAG orchestrator using microservices architecture."""
    
    def __init__(
        self,
        llm_service_url: Optional[str] = None,
        embedder_service_url: Optional[str] = None,
        reranker_service_url: Optional[str] = None,
        vector_store_service_url: Optional[str] = None
    ):
        self.config = ConfigLoader.get_config()
        self.logger = LoggerLoader.get_logger()
        
        # Initialize service clients
        self.llm_client = LLMClient(
            base_url=llm_service_url or os.getenv("LLM_SERVICE_URL", "http://llm-service:8001")
        )
        self.embedder_client = EmbedderClient(
            base_url=embedder_service_url or os.getenv("EMBEDDER_SERVICE_URL", "http://embedder-service:8002")
        )
        self.reranker_client = RerankerClient(
            base_url=reranker_service_url or os.getenv("RERANKER_SERVICE_URL", "http://reranker-service:8003")
        )
        self.vector_store_client = VectorStoreClient(
            base_url=vector_store_service_url or os.getenv("VECTOR_STORE_SERVICE_URL", "http://vector-store-service:8004")
        )
        
        # Initialize subtitle management (still needed for extraction)
        self.db = DBConnector(embedding_dimension=self.config.embedding_dimension)
        self.subtitle_extractor = SubtitleExtractor(embedding_model=None, storage=None)  # Embeddings done by service
        
        # Load prompt template
        self.prompt_template = PromptLoader().load(self.config.language)
        self.retriever_top_k = self.config.retriever.top_k
        self.reranker_top_k = self.config.reranker.top_k if self.config.reranker.use_reranker else None
    
    def extract_video_id(self, video_url: str) -> Optional[str]:
        """Extract video ID from URL."""
        return self.subtitle_extractor.extract_video_id(video_url)
    
    async def _ensure_subtitles(self, video_id: str) -> None:
        """Ensure subtitles exist, extract if needed."""
        if not self.db.fetch_subtitles(video_id):
            self.logger.info(f"Subtitles missing for {video_id}, extracting...")
            extracted = self.subtitle_extractor.get_subtitles(video_id)
            if not extracted:
                raise ValueError("Subtitles not found")
            
            # Generate embeddings using embedder service
            texts = [seg["text"] for seg in extracted]
            embeddings = await self.embedder_client.embed(texts)
            
            # Store with embeddings
            for seg, emb in zip(extracted, embeddings):
                end_time = seg["start"] + seg.get("duration", 0.0)
                self.db.insert_subtitle(
                    video_id=video_id,
                    start_time=seg["start"],
                    end_time=end_time,
                    text=seg["text"],
                    embedding=emb
                )
            
            self.logger.info(f"Subtitles extracted and stored for {video_id}")
    
    async def process_query(self, video_url: str, query: str) -> str:
        """Process query using microservices."""
        try:
            video_id = self.extract_video_id(video_url)
            if not video_id:
                self.logger.error(f"Invalid video URL: {video_url}")
                return "Ошибка: некорректный URL видео."
            
            # Ensure subtitles exist
            try:
                await self._ensure_subtitles(video_id)
            except ValueError:
                return "Ошибка: субтитры не найдены."
            
            # Get query embedding from embedder service
            embeddings = await self.embedder_client.embed([query])
            query_embedding = embeddings[0]
            
            # Search in vector store
            documents = await self.vector_store_client.search(query_embedding, top_k=self.retriever_top_k, video_id=video_id)
            self.logger.info(f"Retrieved {len(documents)} candidates")
            
            if not documents:
                return "По запросу не найдено похожих субтитров."
            
            texts = [doc.get("text") or doc.get("page_content") for doc in documents]
            
            # Optionally rerank
            if self.config.reranker.use_reranker:
                ranked = await self.reranker_client.rerank(query, texts, top_k=self.reranker_top_k)
                selected = [text for text, _ in ranked]
            else:
                selected = texts[:self.retriever_top_k]
            
            snippets_str = "\n".join(
                f"\t{i}.\t{snippet}" for i, snippet in enumerate(selected, 1)
            )
            self.logger.info(
                f"Selected {len(selected)} snippets for context:\n{snippets_str}"
            )
            
            # Build prompt and generate answer
            context = "\n".join(selected)
            prompt = self.prompt_template.format(query=query, context=context)
            answer = await self.llm_client.generate(prompt, max_tokens=1024)
            
            return answer.strip()
            
        except Exception as e:
            self.logger.error(f"process_query error: {e}", exc_info=True)
            return "Ошибка: не удалось обработать запрос."
    
    async def close(self):
        """Close all service clients."""
        await self.llm_client.close()
        await self.embedder_client.close()
        await self.reranker_client.close()
        await self.vector_store_client.close()
        if self.db:
            self.db.close()

