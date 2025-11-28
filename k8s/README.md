# Kubernetes (local) deployment

Use the manifests in this directory to run `video-rag-api` on a local cluster (Minikube, kind, k3d).

## Directory layout

- `namespace.yaml`: Creates the `video-rag` namespace.
- `configmap.yaml`: Drops `config/config.yaml` into `/app/config/config.yaml` inside the pod.
- `secret.yaml`: Supplies Supabase/PostgreSQL credentials plus application environment variables.
- `persistent-volume.yaml` / `persistent-volume-claim.yaml`: Provide hostPath storage for `.gguf` and reranker files.
- `deployment.yaml`: Runs the container with ConfigMap + Secret + PVC + health probes.
- `service.yaml`: Exposes the app via a NodePort service (ports `8000` inside, `30080` outside).

## Local deployment steps

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/persistent-volume.yaml
kubectl apply -f k8s/persistent-volume-claim.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Or apply everything at once:

```bash
kubectl apply -f k8s/
```

## Verification

```bash
kubectl get all -n video-rag
kubectl logs -n video-rag -l app=video-rag-api
minikube service video-rag-api-service -n video-rag --url
curl http://<NodeIP>:30080/health
kubectl exec -n video-rag deployment/video-rag-api -- ls /app/models/llm
```

## Cleanup

```bash
kubectl delete -f k8s/
kubectl delete namespace video-rag
kubectl delete pv video-rag-models-pv
```

## Models

1. Create directories inside Minikube (e.g., `/mnt/data/video-rag-models/llm`).
2. Copy `.gguf`/reranker files via `minikube cp` or `minikube ssh`.
3. The `models` volume is mounted read-only inside `/app/models`.

## Notes

- For AWS production use `k8s-aws/` instead of this folder.
- Secrets and ConfigMaps must match the environment you are targeting (`local` vs `aws`).
