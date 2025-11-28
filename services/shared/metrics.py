"""Prometheus metrics for microservices."""
from prometheus_client import Counter, Histogram

# === REQUEST METRICS ===
request_count = Counter(
    'service_requests_total',
    'Total requests',
    ['service', 'endpoint', 'status']
)

request_duration = Histogram(
    'service_request_duration_seconds',
    'Request duration in seconds',
    ['service', 'endpoint']
)

# === ERROR METRICS ===
error_count = Counter(
    'service_errors_total',
    'Total errors',
    ['service', 'endpoint', 'error_type']
)

# === LLM METRICS ===
llm_tokens_generated = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model', 'language']
)

llm_generation_duration = Histogram(
    'llm_generation_duration_seconds',
    'Time to generate text',
    ['model']
)

# === EMBEDDER METRICS ===
embedder_batch_size = Histogram(
    'embedder_batch_size',
    'Number of texts embedded at once',
    ['model']
)

embedder_embedding_duration = Histogram(
    'embedder_embedding_duration_seconds',
    'Time to generate embeddings',
    ['model', 'batch_size']
)

# === RETRIEVER METRICS ===
retriever_documents_found = Histogram(
    'retriever_documents_found',
    'Number of documents retrieved',
    ['video_id']
)

retriever_search_duration = Histogram(
    'retriever_search_duration_seconds',
    'Time to search vectors',
    []
)

# === RERANKER METRICS ===
reranker_documents_reranked = Histogram(
    'reranker_documents_reranked',
    'Number of documents reranked',
    []
)

reranker_rerank_duration = Histogram(
    'reranker_rerank_duration_seconds',
    'Time to rerank documents',
    []
)

# === ORCHESTRATOR METRICS ===
rag_query_duration = Histogram(
    'rag_query_duration_seconds',
    'Total time for full RAG query',
    ['has_reranker']
)

rag_context_length = Histogram(
    'rag_context_length_tokens',
    'Number of tokens in context sent to LLM',
    []
)

