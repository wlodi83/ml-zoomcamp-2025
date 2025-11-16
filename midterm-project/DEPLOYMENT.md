   # Cloud Deployment Guide - AWS EKS + Route53

This guide explains how to deploy the Report Failure Prediction API to AWS EKS (Kubernetes) and expose it via Route53 at **ml.pushmetrics.io**

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **kubectl** configured to access the production EKS cluster
3. **Docker** installed locally
4. **AWS ECR repository** created for the project
5. **AWS Certificate Manager (ACM)** certificate for `*.pushmetrics.io`
6. **AWS Load Balancer Controller** installed in the production EKS cluster

## Architecture

```
Internet → Route53 (ml.pushmetrics.io)
         → AWS ALB (Application Load Balancer)
         → Kubernetes Ingress
         → Kubernetes Service
         → Pods (1 replica)
```

## Deployment Files

The `k8s/` directory contains:

- **deployment.yaml** - Deployment with 1 replica, resource limits, health checks
- **service.yaml** - ClusterIP service exposing port 80
- **ingress.yaml** - ALB Ingress with TLS/HTTPS configuration

## Quick Start

### Step 1: Configure ECR and ACM

Edit `deploy.sh`:
```bash
AWS_REGION="us-east-1"
ECR_REPO="<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/report-failure-prediction"
```

Edit `k8s/ingress.yaml`:
```yaml
alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-east-1:ACCOUNT_ID:certificate/CERT_ID
```

### Step 2: Deploy

**Option A: Automated deployment**
```bash
./deploy.sh
```

**Option B: Manual deployment**
```bash
# Build and push to ECR
docker build -t report-failure-prediction:latest .
docker tag report-failure-prediction:latest <ECR_REPO>:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ECR_REPO>
docker push <ECR_REPO>:latest

# Update deployment.yaml with ECR image
# Then apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Wait for rollout
kubectl rollout status deployment/report-failure-prediction
```

### Step 3: Configure Route53

1. Get the ALB DNS name:
```bash
kubectl describe ingress report-failure-prediction | grep Address
```

2. Create a CNAME record in Route53:
   - **Name**: ml.pushmetrics.io
   - **Type**: CNAME
   - **Value**: <ALB_DNS_NAME>
   - **TTL**: 300

3. Wait 5-10 minutes for DNS propagation

### Step 4: Verify

```bash
# Check deployment status
kubectl get pods -l app=report-failure-prediction
kubectl get svc report-failure-prediction
kubectl get ingress report-failure-prediction

# Test the endpoint
curl https://ml.pushmetrics.io/health

# Test prediction
curl -X POST https://ml.pushmetrics.io/predict \
  -H "Content-Type: application/json" \
  -d '{
    "num_blocks": 7,
    "num_sql_blocks": 2,
    "num_writeback_blocks": 0,
    "num_viz_blocks": 1,
    "num_tableau_blocks": 0,
    "num_email_blocks": 1,
    "num_slack_blocks": 1,
    "num_api_blocks": 0,
    "num_sftp_blocks": 0,
    "num_storage_blocks": 0,
    "num_control_blocks": 0,
    "num_parameters": 1,
    "num_databases": 1,
    "historical_failure_count": 5,
    "historical_executions": 100,
    "historical_failure_rate": 0.05,
    "avg_historical_duration": 10.5,
    "hours_since_last_success": 24.0,
    "hour_of_day": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_business_hours": 1,
    "is_rerun": 0,
    "is_scheduled": 1,
    "database_types": "bigquery"
  }'
```

## Monitoring

```bash
# View logs
kubectl logs -l app=report-failure-prediction --tail=100 -f

# Check pod health
kubectl describe pod -l app=report-failure-prediction

# View ingress details
kubectl describe ingress report-failure-prediction
```

## Scaling

```bash
# Manual scaling
kubectl scale deployment report-failure-prediction --replicas=5

# Auto-scaling (requires metrics-server)
kubectl autoscale deployment report-failure-prediction \
  --cpu-percent=70 --min=2 --max=10
```

## Updating

```bash
# Build new version
docker build -t report-failure-prediction:v2 .
docker tag report-failure-prediction:v2 <ECR_REPO>:v2
docker push <ECR_REPO>:v2

# Rolling update
kubectl set image deployment/report-failure-prediction \
  report-failure-prediction=<ECR_REPO>:v2

# Monitor rollout
kubectl rollout status deployment/report-failure-prediction

# Rollback if needed
kubectl rollout undo deployment/report-failure-prediction
```

## Troubleshooting

### Pods not starting
```bash
kubectl describe pod -l app=report-failure-prediction
kubectl logs -l app=report-failure-prediction
```

### Ingress not creating ALB
```bash
kubectl describe ingress report-failure-prediction
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
```

### DNS not resolving
```bash
# Check CNAME record
nslookup ml.pushmetrics.io
dig ml.pushmetrics.io
```

## Cost Optimization

- **ALB**: ~$16-20/month
- **Pods**: 1 replica × (256Mi RAM, 200m CPU) - minimal cost
- **ECR**: Storage costs for Docker images (~$0.10/GB/month)

Total estimated cost: **~$20-30/month**

## Security

- ✅ TLS/HTTPS enforced via ACM certificate
- ✅ HTTP automatically redirects to HTTPS
- ✅ Resource limits prevent resource exhaustion
- ✅ Health checks ensure only healthy pods receive traffic
- ✅ Rolling updates ensure zero-downtime deployments

## Live URL

Once deployed: **https://ml.pushmetrics.io**

- Health: https://ml.pushmetrics.io/health
- Info: https://ml.pushmetrics.io/info
- Predict: POST to https://ml.pushmetrics.io/predict
