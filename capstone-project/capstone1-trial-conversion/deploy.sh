#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-central-1}"
ECR_REGISTRY="${ECR_REGISTRY:-}"
ECR_REPO="${ECR_REPO:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

if [[ -z "${ECR_REGISTRY}" || -z "${ECR_REPO}" ]]; then
  echo "Missing ECR settings."
  echo "Set ECR_REGISTRY (e.g. 123456789012.dkr.ecr.eu-central-1.amazonaws.com)"
  echo "Set ECR_REPO (e.g. \${ECR_REGISTRY}/trial-conversion-predictor)"
  exit 1
fi

IMAGE="${ECR_REPO}:${IMAGE_TAG}"

echo "Building image (linux/amd64)..."
docker buildx build --platform linux/amd64 -t trial-conversion-predictor --load .

echo "Tagging image as ${IMAGE}..."
docker tag trial-conversion-predictor:latest "${IMAGE}"

echo "Logging into ECR ${ECR_REGISTRY} (region ${AWS_REGION})..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "Pushing image ${IMAGE}..."
docker push "${IMAGE}"

echo "Updating k8s/deployment.yaml image to ${IMAGE}..."
python - <<PY
from pathlib import Path
path = Path("k8s/deployment.yaml")
data = path.read_text().splitlines()
updated = []
for line in data:
    if line.lstrip().startswith("image: "):
        indent = line[:len(line) - len(line.lstrip())]
        updated.append(f"{indent}image: ${IMAGE}")
    else:
        updated.append(line)
path.write_text("\\n".join(updated) + "\\n")
PY

echo "Applying Kubernetes manifests..."
kubectl apply -f k8s/

echo "Deployment complete."
