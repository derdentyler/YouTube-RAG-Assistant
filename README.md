
# Video RAG API

This project is an API for handling requests using Retrieval-Augmented Generation (RAG). The API accepts video URLs and questions, generating answers using the RAG model and PostgreSQL database. The project uses FastAPI for request handling, PostgreSQL with the pgvector extension for storing vector data, and Docker for containerization.

## Installation and Setup (without Docker)

### Step 1: Install Poetry

To install the project dependencies, you need to use **Poetry**. If you don't have it installed yet, you can install Poetry using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Step 2: Install Project Dependencies

Once Poetry is installed, navigate to the project root directory and run the following command to install all dependencies:

```bash
poetry install
```

### Step 3: Install Llama-cpp-python

Since llama-cpp-python contains OS-specific native binaries, it is recommended to install it manually using pip after running poetry install. This ensures that pip selects the correct precompiled wheel for your operating system.

```bash
pip install llama-cpp-python==0.2.89
```

### Additional notes for users

After running `poetry install`, if you are using a virtual environment (venv), **make sure to activate it first** before running the `pip install` command.

If you use Poetry's default virtual environment (created automatically), activate it via:

- **On Linux/macOS:**

  ```bash
  source $(poetry env info --path)/bin/activate
  ```

- **On Windows (PowerShell)**

  ```bash
  .\$(poetry env info --path)\Scripts\Activate.ps1
  ```


### Step 4: Download the Model

Before running the API, you need to download the model in **.gguf** format and save it in the `models` folder. Use the provided link to download the model and save the file in the `models/` directory of your project.

For example, load [saiga_llama3_8b-q4_k_m.gguf](https://huggingface.co/itlwas/saiga_llama3_8b-Q4_K_M-GGUF/resolve/main/saiga_llama3_8b-q4_k_m.gguf?download=true)

### Step 5: Configure the Settings

The project uses a configuration file where important parameters such as the model path, model settings, and retriever settings are specified. The configuration is validated using Pydantic models, which ensures type safety and catches configuration errors at startup.

**Important:** All configuration errors are detected when the application starts, not during runtime. Make sure your configuration file is valid before running the application.

Example configuration (`config.yaml`):

```yaml
language: "ru"

use_langchain: false

models:
  ru:
    backend: "llama.cpp"
    model_path: "./models/llm/saiga_llama3_8b-q4_k_m.gguf"
    n_ctx: 8192
  en:
    backend: "transformers"
    model_name: "mistralai/Mistral-7B-Instruct-v0.1"

embedding_model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embedding_dimension: 768  # Dimension of embedding vectors

retriever:
  top_k: 6
  similarity_metric: "cosine"

reranker:
  use_reranker: true
  top_k: 3
  model_path: "models/reranker/logreg_reranker.pkl"

# Chunking settings
chunking:
  method: "semantic"  # "semantic" or "time"
  max_tokens: 150
  similarity_threshold: 0.7
  min_chunk_size: 50

# Subtitle fragment time in seconds and overlap (for time-based chunking fallback)
subtitle_block_duration: 60
subtitle_block_overlap: 10
```

**Common Configuration Errors:**

- **Missing model file**: If `model_path` is specified but the file doesn't exist, the application will fail to start with a clear error message.
- **Invalid language**: The `language` field must be either `"ru"` or `"en"`. If you specify a different language, you'll get a validation error.
- **Missing model for language**: If you set `language: "en"` but don't have a model configured for `"en"` in the `models` section, the application will fail to start.
- **Invalid overlap**: `subtitle_block_overlap` must be less than `subtitle_block_duration`, otherwise you'll get a validation error.
- **Missing reranker model**: If `use_reranker: true` but `model_path` is not specified or the file doesn't exist, the application will fail to start.

### Step 6: Run the Application

To start the API, run the following command:

```bash
poetry run start-api
```

This command will launch the FastAPI application using Uvicorn, and you can access the API at `http://localhost:8000`.

## Kubernetes deployment

Kubernetes позволяет проверить RAG на локальном кластере (minikube/kind/k3s) по тем же манифестам, которые используют namespace, ConfigMap, Secret, Deployment и NodePort Service.

1. Обновите `k8s/secret.yaml` реальными значениями Supabase/Postgres (замените `<...>`).  
2. Расположите `.gguf` и `logreg_reranker.pkl` в PV (см. `k8s/persistent-volume.yaml`).  
3. Соберите Docker-образ `video-rag-api:latest` и (если используете minikube) скажите `eval $(minikube docker-env)` перед сборкой.  
4. Примените все манифесты: `kubectl apply -f k8s/`.  
5. Проверьте статус: `kubectl get pods,svc,pvc -n video-rag`.  
6. Получите URL через `minikube service video-rag-api-service -n video-rag --url` или обращайтесь на NodePort `30080`.  
7. Для подробного руководства и команд см. `k8s/README.md`.

### Step 7: Testing the API

The API supports Swagger UI for testing all available endpoints. To access Swagger UI, open the following link in your browser:

```
http://localhost:8000/docs
```

## Installation and Setup (Docker)

We provide a Dockerized version of the API for easy local development and production deployment.

### Building the Docker Image

Build the Docker image (tagged as video-rag-api:latest)

```bash
docker compose build
```

### Running in Development Mode

Start containers, mount local code for live reload (Uvicorn --reload)

```bash
docker compose up
```
### Running with Rebuild
Whenever you update dependencies in pyproject.toml or modify the Dockerfile:
```bash
docker compose up --build -d
```

### Stopping and Cleaning Up
This stops and removes all containers and networks, but preserves volumes (e.g. downloaded subtitles or models)
```bash
docker compose down
```

### Unit tests

To use unit tests:

```bash
poetry run pytest
```

## Semantic Chunking Module

**What is it?**  
Semantic chunking is an advanced method for splitting video subtitles into meaningful chunks based on semantic similarity rather than fixed time windows. This approach preserves topic boundaries and improves retrieval quality for RAG systems.

**How it works**  
1. **Sentence splitting**: Subtitles are split into individual sentences using language-aware patterns.
2. **Embedding computation**: Each sentence is converted to an embedding vector using the embedding model.
3. **Semantic grouping**: Sentences are grouped into chunks based on cosine similarity between consecutive sentences.
4. **Size control**: Chunks are limited to 100-200 tokens to maintain small, focused chunks.
5. **Time preservation**: Original timestamps (start_time, duration) are preserved for each chunk.

**Advantages over time-based chunking:**
- **Semantic coherence**: Chunks contain semantically related content, not arbitrary time slices
- **Better retrieval**: Related information stays together, improving search relevance
- **Adaptive sizing**: Chunk size adapts to content structure, not fixed time intervals
- **Topic boundaries**: Natural topic transitions are preserved as chunk boundaries

**Configuration**

Example configuration (`config.yaml`):

```yaml
chunking:
  method: "semantic"  # "semantic" or "time" for fallback
  max_tokens: 150     # Maximum chunk size in tokens (approximately)
  similarity_threshold: 0.7  # Minimum similarity to merge sentences (0.0-1.0)
  min_chunk_size: 50  # Minimum chunk size in tokens
```

**Parameters:**
- `method`: Chunking strategy - `"semantic"` for semantic chunking or `"time"` for time-based chunking (backward compatibility)
- `max_tokens`: Maximum chunk size. Smaller values (100-150) improve search precision, larger values (200+) provide more context
- `similarity_threshold`: Cosine similarity threshold for merging sentences. Higher values (0.8-0.9) create more conservative chunks, lower values (0.5-0.6) create larger chunks
- `min_chunk_size`: Minimum chunk size to avoid overly small fragments

**Fallback to time-based chunking:**

If `embedding_model` is not available or `method: "time"` is set, the system falls back to time-based chunking using `subtitle_block_duration` and `subtitle_block_overlap` parameters.

## Reranking Module

**What is it?**  
A post-retrieval step that reorders candidate transcript fragments using a trained ML model (Logistic Regression) to improve relevance.

**How it works**  
1. **Retriever** returns top-K fragments (by cosine similarity).  
2. **Reranker** loads `logreg_reranker.pkl` and computes feature vectors (cosine, token overlap, stopword ratio, length difference, position, TF‑IDF similarity).  
3. The model scores each fragment and sorts them in descending order.

**Use reranker**

Example configuration (`config.yaml`):

```yaml
reranker:
  use_reranker: true
  top_k: 3
  model_path: "models/reranker/logreg_reranker.pkl"
```

**Retraining the model**  

1. Prepare data/reranker/train_data.json with entries:
```yaml
{
  "query": "sample question?",
  "fragments": [
    {"text": "candidate 1", "label": 1},
    {"text": "candidate 2", "label": 0},
    …
  ]
}
```

2. Run trainer
```bash
python src/reranker/trainer.py \
  --train-path data/reranker/train_data.json \
  --model-out models/reranker/logreg_reranker.pkl
```


## Architecture

### Dependency Injection

The application uses Dependency Injection pattern for managing dependencies:

- **DependencyContainer**: Manages lifecycle of heavy objects (DB connections, ML models)
- **FastAPI Depends**: Injects dependencies into endpoints
- **Lifespan management**: Proper initialization and cleanup of resources

**Benefits:**
- Better testability: endpoints can be tested with mock dependencies
- Resource efficiency: models loaded once and reused
- Clean separation of concerns
- Easy to swap implementations

**Example:**
```python
from src.core.dependencies.providers import RAGModelDep

@app.post("/query")
def query_endpoint(
    request: QueryRequest,
    rag_model: RAGModelDep  # Automatically injected
) -> QueryResponse:
    return rag_model.process_query(request.video_url, request.query)
```

### Testing with DI

Override dependencies in tests:

```python
def test_endpoint():
    mock_rag = MagicMock()
    app.dependency_overrides[get_rag_model] = lambda: mock_rag
    
    client = TestClient(app)
    response = client.post("/query", json={...})
    
    app.dependency_overrides.clear()
```

## Technologies

- **FastAPI** - for creating the API
- **Uvicorn** - ASGI server
- **PostgreSQL** with **pgvector** extension - for storing vector data
- **LangChain** - for pipline customization
- **Docker** - for containerizing the project
- **Poetry** - for dependency management

## DataBase

Create remote Postgres Database. For example, using [superbase](https://supabase.com/).

## Development

For development and testing, you need to create a `.env` file based .env.example with the following configuration:

```bash
PYTHONPATH=src
LOG_FILE=logs/app.log
SUBTITLES_DIR=downloads/subtitles
SUPABASE_URL=https://*************.supabase.co
SUPABASE_KEY=****************
USER=postgres.***************
HOST=***************.pooler.supabase.com
PORT=5432
DBNAME=postgres
```

Make sure that the `.env` file is correctly configured with your Supabase instance details and model path.

## License

This project is licensed under the MIT License.

## Future Plans

In the future, I plan to add the following features:

- **Refactoring**: LangSmith integration.
- **User Interface (UI)**: To provide a user-friendly interface for interacting with the API.

## Contact

For any questions or suggestions, feel free to reach out: 📧 [alexander.polybinsky@gmail.com
]()