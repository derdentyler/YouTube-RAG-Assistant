from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional, Dict, Union
import os


class ModelConfigLlamaCpp(BaseModel):
    """Конфигурация для llama.cpp моделей."""
    backend: Literal["llama.cpp"]
    model_path: str
    n_ctx: int = Field(default=2048, ge=512, le=32768, description="Context window size")
    
    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Проверка существования файла модели."""
        if not os.path.exists(v):
            raise ValueError(f"Model file not found: {v}")
        return v


class ModelConfigTransformers(BaseModel):
    """Конфигурация для Transformers моделей."""
    backend: Literal["transformers"]
    model_name: str
    n_ctx: Optional[int] = Field(default=2048, ge=512, le=32768)


class RetrieverConfig(BaseModel):
    """Конфигурация ретривера."""
    top_k: int = Field(default=5, ge=1, le=100, description="Number of documents to retrieve")
    similarity_metric: Literal["cosine", "euclidean", "dot_product"] = "cosine"


class RerankerConfig(BaseModel):
    """Конфигурация реранкера."""
    use_reranker: bool = False
    top_k: int = Field(default=3, ge=1, le=50, description="Number of documents after reranking")
    model_path: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_model_path_if_enabled(self):
        """Проверка наличия model_path если use_reranker=True."""
        if self.use_reranker and not self.model_path:
            raise ValueError("model_path is required when use_reranker=True")
        if self.use_reranker and self.model_path and not os.path.exists(self.model_path):
            raise ValueError(f"Reranker model file not found: {self.model_path}")
        return self


class AppConfig(BaseModel):
    """Главная конфигурация приложения."""
    language: Literal["ru", "en"] = "ru"
    use_langchain: bool = False
    
    # Модели LLM для разных языков
    models: Dict[str, Union[ModelConfigLlamaCpp, ModelConfigTransformers]]
    
    # Embedding модель
    embedding_model: str
    embedding_dimension: int = Field(
        default=768,
        ge=128,
        le=4096,
        description="Dimension of embedding vectors"
    )
    
    # Настройки компонентов
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    
    # Параметры обработки субтитров
    subtitle_block_duration: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Duration of subtitle blocks in seconds"
    )
    subtitle_block_overlap: int = Field(
        default=10,
        ge=0,
        le=300,
        description="Overlap between subtitle blocks in seconds"
    )
    
    @model_validator(mode='after')
    def validate_subtitle_overlap(self):
        """Проверка что overlap меньше duration."""
        if self.subtitle_block_overlap >= self.subtitle_block_duration:
            raise ValueError(
                f"subtitle_block_overlap ({self.subtitle_block_overlap}) must be less than "
                f"subtitle_block_duration ({self.subtitle_block_duration})"
            )
        return self
    
    @model_validator(mode='after')
    def validate_language_model_exists(self):
        """Проверка что для выбранного языка есть модель."""
        if self.language not in self.models:
            raise ValueError(
                f"No model configured for language '{self.language}'. "
                f"Available: {list(self.models.keys())}"
            )
        return self

