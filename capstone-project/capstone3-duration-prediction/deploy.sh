#!/bin/bash
# Workflow Duration Prediction - Deployment Script
# This script builds, pushes, and deploys the service to AWS EKS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Duration Prediction - Deployment${NC}"
echo -e "${GREEN}============================================${NC}"

# Configuration
AWS_REGION=${AWS_REGION:-"eu-central-1"}
ECR_REGISTRY=${ECR_REGISTRY:-""}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
CERTIFICATE_ARN=${CERTIFICATE_ARN:-""}

# Validate required variables
if [ -z "$ECR_REGISTRY" ]; then
    echo -e "${RED}ERROR: ECR_REGISTRY environment variable is required${NC}"
    echo "Usage: ECR_REGISTRY=123456789012.dkr.ecr.eu-central-1.amazonaws.com ./deploy.sh"
    exit 1
fi

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  AWS_REGION: $AWS_REGION"
echo "  ECR_REGISTRY: $ECR_REGISTRY"
echo "  IMAGE_TAG: $IMAGE_TAG"

# Step 1: Login to ECR
echo -e "\n${YELLOW}Step 1: Logging in to ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
echo -e "${GREEN}Logged in successfully${NC}"

# Step 2: Build Docker image
echo -e "\n${YELLOW}Step 2: Building Docker image...${NC}"
docker build --platform linux/amd64 -t duration-prediction:$IMAGE_TAG .
echo -e "${GREEN}Image built successfully${NC}"

# Step 3: Tag image for ECR
echo -e "\n${YELLOW}Step 3: Tagging image for ECR...${NC}"
docker tag duration-prediction:$IMAGE_TAG $ECR_REGISTRY/duration-prediction:$IMAGE_TAG
echo -e "${GREEN}Image tagged successfully${NC}"

# Step 4: Push to ECR
echo -e "\n${YELLOW}Step 4: Pushing image to ECR...${NC}"
docker push $ECR_REGISTRY/duration-prediction:$IMAGE_TAG
echo -e "${GREEN}Image pushed successfully${NC}"

# Step 5: Update Kubernetes manifests with variables
echo -e "\n${YELLOW}Step 5: Updating Kubernetes manifests...${NC}"

# Backup original files
cp k8s/deployment.yaml k8s/deployment.yaml.bak
cp k8s/ingress.yaml k8s/ingress.yaml.bak

# Substitute variables
sed -i.tmp "s|\${ECR_REGISTRY}|$ECR_REGISTRY|g" k8s/deployment.yaml
sed -i.tmp "s|\${IMAGE_TAG}|$IMAGE_TAG|g" k8s/deployment.yaml
sed -i.tmp "s|\${CERTIFICATE_ARN}|$CERTIFICATE_ARN|g" k8s/ingress.yaml

rm -f k8s/*.tmp
echo -e "${GREEN}Manifests updated successfully${NC}"

# Step 6: Apply Kubernetes manifests
echo -e "\n${YELLOW}Step 6: Applying Kubernetes manifests...${NC}"
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
if [ -n "$CERTIFICATE_ARN" ]; then
    kubectl apply -f k8s/ingress.yaml
else
    echo -e "${YELLOW}Skipping ingress (no CERTIFICATE_ARN provided)${NC}"
fi
echo -e "${GREEN}Manifests applied successfully${NC}"

# Step 7: Wait for rollout
echo -e "\n${YELLOW}Step 7: Waiting for deployment rollout...${NC}"
kubectl rollout status deployment/duration-prediction --timeout=120s
echo -e "${GREEN}Deployment rolled out successfully${NC}"

# Step 8: Show status
echo -e "\n${YELLOW}Step 8: Deployment status:${NC}"
kubectl get pods -l app=duration-prediction
kubectl get svc duration-prediction

# Restore original files
mv k8s/deployment.yaml.bak k8s/deployment.yaml
mv k8s/ingress.yaml.bak k8s/ingress.yaml

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "\nEndpoints:"
echo "  Internal: http://duration-prediction.default.svc.cluster.local/predict"
if [ -n "$CERTIFICATE_ARN" ]; then
    echo "  External: https://duration.pushmetrics.io/predict"
fi
