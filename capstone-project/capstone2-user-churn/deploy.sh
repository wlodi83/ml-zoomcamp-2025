#!/bin/bash
# User Churn Prediction - Deployment Script
# ML Zoomcamp Capstone 2 Project
#
# This script builds and deploys the Docker image to AWS ECR and Kubernetes.
#
# Prerequisites:
#   - AWS CLI configured with proper credentials
#   - kubectl configured for your cluster
#   - Docker installed and running
#
# Usage:
#   ./deploy.sh
#
# Environment variables (set before running):
#   AWS_REGION        - AWS region (default: eu-central-1)
#   ECR_REGISTRY      - ECR registry URL
#   ECR_REPO          - Full ECR repository URL
#   IMAGE_TAG         - Docker image tag (default: latest)
#   CERTIFICATE_ARN   - ACM certificate ARN for HTTPS

set -e

# Configuration
AWS_REGION="${AWS_REGION:-eu-central-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
APP_NAME="user-churn-predictor"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}User Churn Prediction - Deployment${NC}"
echo -e "${GREEN}======================================${NC}"

# Check required environment variables
if [ -z "$ECR_REGISTRY" ]; then
    echo -e "${RED}Error: ECR_REGISTRY not set${NC}"
    echo "Please set: export ECR_REGISTRY=123456789012.dkr.ecr.eu-central-1.amazonaws.com"
    exit 1
fi

ECR_REPO="${ECR_REPO:-${ECR_REGISTRY}/${APP_NAME}}"

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  AWS_REGION:    $AWS_REGION"
echo "  ECR_REPO:      $ECR_REPO"
echo "  IMAGE_TAG:     $IMAGE_TAG"

# Step 1: Login to ECR
echo -e "\n${YELLOW}Step 1: Logging in to ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

# Step 2: Build Docker image
echo -e "\n${YELLOW}Step 2: Building Docker image...${NC}"
docker build --platform linux/amd64 -t $APP_NAME:$IMAGE_TAG .

# Step 3: Tag image for ECR
echo -e "\n${YELLOW}Step 3: Tagging image for ECR...${NC}"
docker tag $APP_NAME:$IMAGE_TAG $ECR_REPO:$IMAGE_TAG

# Step 4: Push to ECR
echo -e "\n${YELLOW}Step 4: Pushing image to ECR...${NC}"
docker push $ECR_REPO:$IMAGE_TAG

# Step 5: Update Kubernetes deployment
echo -e "\n${YELLOW}Step 5: Updating Kubernetes manifests...${NC}"

# Replace variables in deployment.yaml
sed -i.bak "s|\${ECR_REPO}|$ECR_REPO|g" k8s/deployment.yaml
sed -i.bak "s|\${IMAGE_TAG}|$IMAGE_TAG|g" k8s/deployment.yaml

# Replace variables in ingress.yaml if CERTIFICATE_ARN is set
if [ -n "$CERTIFICATE_ARN" ]; then
    sed -i.bak "s|\${CERTIFICATE_ARN}|$CERTIFICATE_ARN|g" k8s/ingress.yaml
fi

# Step 6: Apply Kubernetes configurations
echo -e "\n${YELLOW}Step 6: Applying Kubernetes configurations...${NC}"
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Step 7: Wait for deployment
echo -e "\n${YELLOW}Step 7: Waiting for deployment to complete...${NC}"
kubectl rollout status deployment/$APP_NAME --timeout=120s

# Step 8: Show status
echo -e "\n${YELLOW}Step 8: Deployment status:${NC}"
kubectl get deployment $APP_NAME
kubectl get pods -l app=$APP_NAME
kubectl get service $APP_NAME
kubectl get ingress $APP_NAME

# Restore original k8s files
mv k8s/deployment.yaml.bak k8s/deployment.yaml 2>/dev/null || true
mv k8s/ingress.yaml.bak k8s/ingress.yaml 2>/dev/null || true

echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${GREEN}======================================${NC}"

# Show endpoint
INGRESS_HOST=$(kubectl get ingress $APP_NAME -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "")
if [ -n "$INGRESS_HOST" ]; then
    echo -e "\nService available at: ${GREEN}https://$INGRESS_HOST${NC}"
    echo -e "Health check: ${GREEN}https://$INGRESS_HOST/health${NC}"
fi
