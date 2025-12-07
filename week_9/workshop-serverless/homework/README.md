# ML Zoomcamp Week 9 Homework - Serverless Deployment

## Summary of Answers (Questions 1-4)

- **Question 1**: Output node name = `output`
- **Question 2**: Target size = `200x200`
- **Question 3**: First pixel R channel = `-1.073`
- **Question 4**: Model output = `0.09`

## Files Created

### Core Lambda Function
- `lambda_function.py` - Main Lambda handler with preprocessing and inference
- `test_lambda.py` - Local testing script (without Docker)

### Docker Deployment
- `Dockerfile` - Docker image configuration
- `DOCKER_INSTRUCTIONS.md` - Step-by-step Docker instructions
- `test_docker.py` - Test script for Docker container

### Development/Testing
- `homework.py` - Solution script for questions 1-4

## Quick Start

### Local Testing (Questions 1-4)
```bash
source venv/bin/activate
python homework.py
```

### Test Lambda Function Locally
```bash
source venv/bin/activate
python test_lambda.py
```

### Docker Deployment (Questions 5-6)

**Question 5: Check base image size**
```bash
docker pull agrigorev/model-2025-hairstyle:v1
docker images agrigorev/model-2025-hairstyle:v1
```

**Question 6: Build and test Lambda container**

```bash
# Build
docker build -t hair-classifier .
```

**Testing on ARM64 Macs (M1/M2/M3):**

The base image is AMD64-only, so the Lambda HTTP endpoint doesn't work on ARM Macs. Use direct invocation instead:

```bash
# Option 1: Using the bash script
./test_docker_direct.sh

# Option 2: Using the Python script
python3 test_docker_local.py

# Option 3: Manual testing
docker run -d --rm --name hair-test hair-classifier
docker exec hair-test python3 -c "
import lambda_function
result = lambda_function.lambda_handler(
    {'url': 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'},
    None
)
print(result)
"
docker stop hair-test
```

**Testing on Intel/AMD64 systems or in AWS:**

```bash
# Run container
docker run -it --rm -p 8080:8080 hair-classifier

# In another terminal, test it
curl -X POST http://localhost:8080/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'
```

## Preprocessing Steps (from Homework 8)

1. Resize to 200x200
2. Convert to numpy array
3. Normalize from [0, 255] to [0, 1]
4. Apply ImageNet normalization:
   - mean = [0.485, 0.456, 0.406]
   - std = [0.229, 0.224, 0.225]
5. Transpose from HWC to CHW format
6. Add batch dimension: (3, 200, 200) → (1, 3, 200, 200)
7. Convert to float32

## Dependencies

- Python 3.13
- onnxruntime 1.20.0 (compatible with macOS 13.2)
- numpy
- pillow
- requests (for testing)

## AWS Lambda Deployment

### Prerequisites
- AWS CLI installed and configured
- Docker installed
- AWS account with appropriate permissions

### Deployment Steps

**1. Set environment variables:**
```bash
export AWS_REGION="us-east-1"
export ACCOUNT_ID="YOUR_AWS_ACCOUNT_ID"
export REPO_NAME="hair-classifier"
```

**2. Create ECR repository:**
```bash
aws ecr create-repository --repository-name hair-classifier --region us-east-1
```

**3. Build Docker image for AMD64 (AWS Lambda platform):**
```bash
docker build --platform linux/amd64 -t hair-classifier:latest .
```

**4. Tag and push to ECR:**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag hair-classifier:latest ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/hair-classifier:latest

# Push to ECR
docker push ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/hair-classifier:latest
```

**5. Create IAM role for Lambda:**
```bash
aws iam create-role \
  --role-name lambda-hair-classifier-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach basic execution policy
aws iam attach-role-policy \
  --role-name lambda-hair-classifier-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

**6. Create Lambda function:**
```bash
aws lambda create-function \
  --function-name hair-classifier \
  --package-type Image \
  --code ImageUri=${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/hair-classifier:latest \
  --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-hair-classifier-role \
  --timeout 30 \
  --memory-size 1024 \
  --region us-east-1
```

**7. Test the Lambda function:**
```bash
aws lambda invoke \
  --function-name hair-classifier \
  --cli-binary-format raw-in-base64-out \
  --payload '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}' \
  --region us-east-1 \
  response.json

cat response.json
# Expected output: {"prediction": -0.10220833122730255}
```

### Update Lambda Function (after code changes)

```bash
# Rebuild and push image
docker build --platform linux/amd64 -t hair-classifier:latest .
docker tag hair-classifier:latest ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/hair-classifier:latest
docker push ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/hair-classifier:latest

# Update Lambda function
aws lambda update-function-code \
  --function-name hair-classifier \
  --image-uri ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/hair-classifier:latest \
  --region us-east-1
```

### Cleanup AWS Resources

When you're done, clean up to avoid charges:

```bash
# Delete Lambda function
aws lambda delete-function --function-name hair-classifier --region us-east-1

# Delete ECR repository
aws ecr delete-repository --repository-name hair-classifier --region us-east-1 --force

# Detach policy and delete IAM role
aws iam detach-role-policy \
  --role-name lambda-hair-classifier-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name lambda-hair-classifier-role
```

## Notes

- The Docker image uses a different model (`hair_classifier_empty.onnx`) than the one used in questions 1-4
- The preprocessing steps remain the same
- The lambda_function.py handles both local and Docker model paths
- **ARM64 Mac Users:** The Lambda HTTP endpoint doesn't work locally due to platform emulation, but the Docker image works perfectly when deployed to AWS Lambda
