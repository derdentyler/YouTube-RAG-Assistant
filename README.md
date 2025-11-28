# Video RAG API

An API that answers natural-language questions about YouTube videos using Retrieval-Augmented Generation. It keeps subtitles in PostgreSQL (with the pgvector extension), chunks them semantically, retrieves the most relevant fragments, and feeds them into a local LLM.

## Technology stack

- FastAPI + Uvicorn for the HTTP server
- PostgreSQL with pgvector for storing embeddings
- llama.cpp + Transformers for LLMs
- Sentence Transformers for embeddings
- Poetry for dependency management
- Docker + Kubernetes for deployment

## Getting started

### Prerequisites

1. Install Python 3.12+ and [Poetry](https://python-poetry.org/docs/).
2. Install a local PostgreSQL-compatible service (Supabase is recommended).
3. Download the required model files into `models/llm/` and `models/reranker/`.
4. Create a `.env.local` file (see `.env.example`).

### Install dependencies

```bash
poetry install
poetry run pip install llama-cpp-python==0.2.89
```

### Download the models

Place the `.gguf` file under `models/llm/` and the reranker under `models/reranker/`. For example:

```bash
mkdir -p models/llm models/reranker
download https://huggingface.co/.../saiga_llama3_8b-q4_k_m.gguf -O models/llm/saiga_llama3_8b-q4_k_m.gguf
cp path/to/logreg_reranker.pkl models/reranker/
```

### Configure the application

- Copy `.env.local.example` (or `.env.example`) to `.env.local` and fill in Supabase credentials.
- Export the variables for your shell before running locally:

```bash
cp .env.local .env
export $(cat .env | xargs)
```

- The application reads `CONFIG_PATH`, defaults to `config/config.yaml`, and validates everything with `AppConfig` (Pydantic). Override it with `CONFIG_PATH=config/config.local.yaml` or `config/config.aws.yaml` depending on the target environment.

### Run locally

```bash
poetry run start-api
```

Visit `http://localhost:8000/docs` for the Swagger UI.

### Docker

```bash
docker compose up --build
```

The Docker stack automatically mounts the `models/`, `src/`, and `config/` directories so you can iterate quickly. Use `docker compose down` to stop the stack.

### Kubernetes (Minikube)

1. Ensure models and reranker files are copied into Minikube's PV (`/mnt/data/video-rag-models`).
2. Apply the manifests under `k8s/`:

```bash
kubectl apply -f k8s/
```

3. Check the status and hit the health endpoint:

```bash
kubectl get pods,svc,pvc -n video-rag
minikube service video-rag-api-service -n video-rag --url
curl http://<NodeIP>:30080/health
```

### AWS Production (EKS)

1. Prepare AWS resources (EKS cluster, RDS PostgreSQL with pgvector, S3 bucket, IAM role).
2. Build and push the Docker image to ECR:

```bash
docker build -t video-rag-api:latest .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
docker tag video-rag-api:latest <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/video-rag-api:latest
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/video-rag-api:latest
```

3. Edit `k8s-aws/secret.yaml` with real secrets and apply all manifests:

```bash
kubectl apply -f k8s-aws/
kubectl rollout status deployment/video-rag-api -n video-rag
```

4. Verify with:

```bash
kubectl get svc -n video-rag
curl http://$(kubectl get svc video-rag-api-service -n video-rag -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')/health
```

## Architecture notes

### Config management

- `AppConfig` (Pydantic) defines the structure for all configuration sections (models, chunking, retriever, reranker, AWS, etc.).
- `ConfigLoader` loads the YAML file pointed to by `CONFIG_PATH` and validates it before the application starts.

### Storage adapters

- `StorageBackend` is an abstraction for storing auxiliary artifacts such as downloaded subtitles.
- `LocalStorage` writes to disk (`downloads/subtitles` by default).
- `S3Storage` pushes files to an S3 bucket. The `StorageFactory` selects the implementation via `STORAGE_BACKEND`.

### Dependency injection

- `DependencyContainer` lazily loads heavy objects (DB connection, embedding model, LLM, storage) in RAM and keeps them alive for the server lifetime.
- FastAPI injects `RAGModel` via `Depends`, so endpoints reuse shared resources instead of reloading them on every request.

### Semantic chunking

The semantic chunker splits subtitles into sentences, computes embeddings, and groups them by cosine similarity while preserving timestamps. This improves retrieval quality compared to naive time-based windows.

### Reranking module

Optional logistic-regression reranker (`models/reranker/logreg_reranker.pkl`) reranks the top-K candidates returned by the retriever using features like cosine similarity, overlap, and length.

## Testing

```bash
poetry run pytest
```

New tests in `tests/test_storage_adapters.py` cover the storage adapters and factory.

## Further reading

- Local Kubernetes manifests: `k8s/` (Minikube-friendly, hostPath + NodePort).
- AWS manifests + IAM policy: `k8s-aws/` (EKS, load balancer, S3-backed init container).

## Contact

Questions? Reach out to `alexander.polybinsky@gmail.com`.
