# Microservices Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the RAG application as microservices.

## Architecture

The application is split into 5 microservices:

1. **Orchestrator** - Main API that coordinates all services (port 8000)
2. **LLM Service** - Text generation (port 8001)
3. **Embedder Service** - Text embeddings (port 8002)
4. **Reranker Service** - Document reranking (port 8003)
5. **Vector Store Service** - Vector search operations (port 8004)

## Prerequisites

- Kubernetes cluster (minikube, kind, or cloud)
- kubectl configured
- Docker images built and available

## Deployment Steps

### 1. Build Docker Images

```bash
# Build all service images
docker build -f services/llm_service/Dockerfile -t rag-llm-service:latest .
docker build -f services/embedder_service/Dockerfile -t rag-embedder-service:latest .
docker build -f services/reranker_service/Dockerfile -t rag-reranker-service:latest .
docker build -f services/vector_store_service/Dockerfile -t rag-vector-store-service:latest .
docker build -f services/orchestrator/Dockerfile -t video-rag-orchestrator:latest .
```

### 2. Load Images into Minikube (if using minikube)

```bash
minikube image load rag-llm-service:latest
minikube image load rag-embedder-service:latest
minikube image load rag-reranker-service:latest
minikube image load rag-vector-store-service:latest
minikube image load video-rag-orchestrator:latest
```

### 3. Create ConfigMap and Secret

First, create the ConfigMap and Secret (reuse from `k8s/` directory or create new ones):

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
```

### 4. Create Persistent Volume for Models

```bash
kubectl apply -f k8s/persistent-volume.yaml
kubectl apply -f k8s/persistent-volume-claim.yaml
```

### 5. Deploy Microservices

```bash
# Create namespace
kubectl apply -f k8s-microservices/namespace.yaml

# Deploy services
kubectl apply -f k8s-microservices/llm-service-deployment.yaml
kubectl apply -f k8s-microservices/embedder-service-deployment.yaml
kubectl apply -f k8s-microservices/reranker-service-deployment.yaml
kubectl apply -f k8s-microservices/vector-store-service-deployment.yaml
kubectl apply -f k8s-microservices/orchestrator-deployment.yaml
```

### 6. Verify Deployment

```bash
# Check all pods
kubectl get pods -n video-rag

# Check services
kubectl get svc -n video-rag

# Check logs
kubectl logs -n video-rag -l app=orchestrator
kubectl logs -n video-rag -l app=llm-service
```

### 7. Access the API

```bash
# Get NodePort URL
minikube service orchestrator -n video-rag --url

# Or use NodePort directly
curl http://$(minikube ip):30080/health
```

## Scaling

Each service can be scaled independently:

```bash
# Scale embedder service (handles most requests)
kubectl scale deployment embedder-service -n video-rag --replicas=5

# Scale orchestrator
kubectl scale deployment orchestrator -n video-rag --replicas=3
```

## Monitoring

Each service exposes a `/health` endpoint:

```bash
# Check LLM service health
kubectl port-forward -n video-rag svc/llm-service 8001:8001
curl http://localhost:8001/health

# Check embedder service health
kubectl port-forward -n video-rag svc/embedder-service 8002:8002
curl http://localhost:8002/health
```

## Cleanup

```bash
kubectl delete -f k8s-microservices/
kubectl delete namespace video-rag
```

