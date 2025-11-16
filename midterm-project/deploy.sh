#!/bin/bash
set -e

# Configuration
PROJECT_NAME="report-failure-prediction"
CLUSTER_NAME="prod-6"
AWS_REGION="eu-central-1"
ECR_REPO="133737826969.dkr.ecr.eu-central-1.amazonaws.com/ml"
IMAGE_TAG="${ECR_REPO}:latest"

echo "=========================================="
echo "Deploying ${PROJECT_NAME} to Kubernetes"
echo "=========================================="

# Step 1: Ensure you're on the correct kubectl context
echo ""
echo "Step 1: Checking kubectl context..."
CURRENT_CONTEXT=$(kubectl config current-context)
echo "Current context: ${CURRENT_CONTEXT}"

if [[ "$CURRENT_CONTEXT" != *"$CLUSTER_NAME"* ]]; then
    echo "⚠️  Warning: Current context doesn't match cluster name '${CLUSTER_NAME}'"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Build Docker image for AMD64 (Linux x86_64)
echo ""
echo "Step 2: Building Docker image for AMD64 platform..."
docker buildx build --platform linux/amd64 -t ${PROJECT_NAME}:latest --load .
echo "✓ Docker image built successfully"

# Step 3: Tag image for ECR
echo ""
echo "Step 3: Tagging image for ECR..."
docker tag ${PROJECT_NAME}:latest ${IMAGE_TAG}
echo "✓ Image tagged: ${IMAGE_TAG}"

# Step 4: Authenticate with ECR
echo ""
echo "Step 4: Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO%/*}
echo "✓ Authenticated with ECR"

# Step 5: Push image to ECR
echo ""
echo "Step 5: Pushing image to ECR..."
docker push ${IMAGE_TAG}
echo "✓ Image pushed successfully"

# Step 6: Update deployment manifest with image
echo ""
echo "Step 6: Updating deployment manifest..."
sed -i.bak "s|<YOUR_ECR_REPO>/report-failure-prediction:latest|${IMAGE_TAG}|g" k8s/deployment.yaml
echo "✓ Deployment manifest updated"

# Step 7: Apply Kubernetes manifests
echo ""
echo "Step 7: Applying Kubernetes manifests..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
echo "✓ Kubernetes manifests applied"

# Restore original deployment.yaml
mv k8s/deployment.yaml.bak k8s/deployment.yaml

# Step 8: Wait for deployment to be ready
echo ""
echo "Step 8: Waiting for deployment to be ready..."
kubectl rollout status deployment/${PROJECT_NAME} --timeout=300s
echo "✓ Deployment is ready"

# Step 9: Get service information
echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
echo ""
echo "Pods:"
kubectl get pods -l app=${PROJECT_NAME}
echo ""
echo "Service:"
kubectl get service ${PROJECT_NAME}
echo ""
echo "Ingress:"
kubectl get ingress ${PROJECT_NAME}
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Check the ingress for the ALB DNS name:"
echo "   kubectl describe ingress ${PROJECT_NAME}"
echo ""
echo "2. Create a Route53 CNAME record:"
echo "   ml.pushmetrics.io -> <ALB_DNS_NAME>"
echo ""
echo "3. Wait for DNS propagation (5-10 minutes)"
echo ""
echo "4. Test the endpoint:"
echo "   curl https://ml.pushmetrics.io/health"
echo ""
echo "=========================================="
echo "✓ Deployment completed successfully!"
echo "=========================================="
