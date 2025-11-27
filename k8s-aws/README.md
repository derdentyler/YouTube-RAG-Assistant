# AWS Deployment Guide

This directory holds AWS-specific Kubernetes manifests and guidance for running `video-rag-api` in production on EKS.

## Prerequisites

- AWS CLI configured (`aws configure`)
- `kubectl` installed and pointed at your cluster
- `eksctl` installed for service account creation (optional but recommended)
- Docker installed and authenticated with AWS ECR

## Step 1: Provision AWS resources

### 1.1 Create an EKS cluster

```bash
eksctl create cluster \
  --name video-rag-cluster \
  --region us-east-1 \
  --nodegroup-name workers \
  --node-type t3.xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4
```

### 1.2 Create an RDS PostgreSQL instance with pgvector

```bash
aws rds create-db-instance \
  --db-instance-identifier video-rag-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 16.1 \
  --master-username postgres \
  --master-user-password <YOUR_PASSWORD> \
  --allocated-storage 20 \
  --vpc-security-group-ids <YOUR_SECURITY_GROUP> \
  --db-subnet-group-name <YOUR_SUBNET_GROUP> \
  --publicly-accessible false \
  --backup-retention-period 7
```

Once the instance is ready:

```bash
psql -h <RDS_HOST> -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 1.3 Create an S3 bucket for models and subtitles

```bash
aws s3 mb s3://my-video-rag-bucket --region us-east-1
aws s3 cp models/llm/saiga.gguf s3://my-video-rag-bucket/models/llm/
aws s3 cp models/reranker/logreg_reranker.pkl s3://my-video-rag-bucket/models/reranker/
```

### 1.4 Create IAM policy and service account

```bash
aws iam create-policy --policy-name VideoRAGPodPolicy --policy-document file://k8s-aws/iam-policy.json
eksctl create iamserviceaccount \
  --name video-rag-sa \
  --namespace video-rag \
  --cluster video-rag-cluster \
  --attach-policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/VideoRAGPodPolicy \
  --approve
```

## Step 2: Build, tag, and push Docker image

```bash
aws ecr create-repository --repository-name video-rag-api --region us-east-1
ECR_URL=$(aws ecr describe-repositories --repository-names video-rag-api --query 'repositories[0].repositoryUri' --output text)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URL
docker build -t video-rag-api:latest .
docker tag video-rag-api:latest $ECR_URL:latest
docker push $ECR_URL:latest
```

## Step 3: Deploy Kubernetes manifests

Update `k8s-aws/secret.yaml` with real endpoints and keys, then:

```bash
aws eks update-kubeconfig --name video-rag-cluster --region us-east-1
kubectl apply -f k8s-aws/namespace.yaml
kubectl apply -f k8s-aws/secret.yaml
kubectl apply -f k8s-aws/configmap.yaml
kubectl apply -f k8s-aws/serviceaccount.yaml
kubectl apply -f k8s-aws/persistent-volume.yaml
kubectl apply -f k8s-aws/persistent-volume-claim.yaml
kubectl apply -f k8s-aws/deployment.yaml
kubectl apply -f k8s-aws/service.yaml
kubectl rollout status deployment/video-rag-api -n video-rag
```

## Step 4: Verify the deployment

```bash
kubectl get pods,svc,pvc -n video-rag
kubectl logs -n video-rag -l app=video-rag-api
LB_URL=$(kubectl get svc video-rag-api-service -n video-rag -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl http://$LB_URL/health
```

## Cleanup

```bash
kubectl delete -f k8s-aws/
eksctl delete cluster --name video-rag-cluster --region us-east-1
aws rds delete-db-instance --db-instance-identifier video-rag-db --skip-final-snapshot
aws s3 rb s3://my-video-rag-bucket --force
```
