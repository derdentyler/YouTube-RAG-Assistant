# Kubernetes deployment

В директории `k8s/` описаны манифесты для базового деплоя `video-rag-api` в Kubernetes (локально или в облаке). Это обособленный `Namespace`, ConfigMap с конфигом, Secret, PV/PVC для моделей, Deployment и NodePort Service.

## Что делает каждая манифест

- `namespace.yaml` — создаёт namespace `video-rag` для изоляции ресурсов.  
- `configmap.yaml` — разворачивает `config/config.yaml` из репозитория и монтирует его как файл.  
- `secret.yaml` — хранит переменные окружения (Supabase, БД, пути). Перед применением замените значения `<...>` на реальные.  
- `persistent-volume.yaml` / `persistent-volume-claim.yaml` — гарантируют доступное хранилище (hostPath) для загрузки моделей.  
- `deployment.yaml` — запускает `video-rag-api:latest` с `ConfigMap`, `Secret`, PVC и пробами `/health`.  
- `service.yaml` — NodePort (30080) для проверки API извне.

## Как развернуть

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/persistent-volume.yaml
kubectl apply -f k8s/persistent-volume-claim.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Или сразу:

```bash
kubectl apply -f k8s/
```

## Проверка

1. `kubectl get all -n video-rag` — убедитесь, что `pod`, `deployment`, `service`, `pvc` созданы.  
2. `kubectl logs -n video-rag -l app=video-rag-api` — проверка логов.  
3. `minikube service video-rag-api-service -n video-rag --url` или вручную `curl http://<NodeIP>:30080/health`.  
4. `kubectl exec -it -n video-rag deployment/video-rag-api -- env | grep SUPABASE` — убедиться, что Secret подхватился.  
5. `kubectl exec -it -n video-rag deployment/video-rag-api -- ls /app/models/llm` — проверить наличие моделей.

## Очистка

```bash
kubectl delete -f k8s/
kubectl delete namespace video-rag
kubectl delete pv video-rag-models-pv
```

## Добавление моделей

1. Создайте директории на хосте (для minikube):  
   ```
   minikube ssh
   sudo mkdir -p /mnt/data/video-rag-models/llm
   exit
   ```
2. Скопируйте `.gguf` и reranker в hostPath или используйте `kubectl cp`.

## Советы

- Используйте `minikube start --memory=6g --cpus=2`.  
- Обновляйте образ `video-rag-api:latest` перед деплоем.  
- Настройте `Secret` с реальными значениями и, по необходимости, добавьте `ConfigMap` для `.env`.

