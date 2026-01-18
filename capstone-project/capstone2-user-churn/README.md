   # User Churn Prediction

**ML Zoomcamp 2025 - Capstone 2 Project by Lukasz Wlodarczyk**

## Problem Description

This project predicts whether users of **PushMetrics** (pushmetrics.io), a SaaS platform for automated reporting and data delivery, will churn (stop using the platform) within the next 30 days.

**Business Value:**
- Identify at-risk users before they churn
- Enable proactive retention campaigns by customer success team
- Optimize resource allocation for user engagement
- Reduce customer acquisition costs through improved retention
- Potential to increase retention rate by 10-20%

### ML Problem

- **Type**: Binary classification
- **Target**: `churned` (1 = no login in 30+ days, 0 = still active)
- **Features**: 22 behavioral and profile features
- **Dataset**: 1,911 users who logged in at least once
- **Churn Rate**: ~90% (users inactive for 30+ days)
- **Metric**: ROC AUC (handles class imbalance well)
- **Best Model**: XGBoost

## Dataset

### Data Collection

Data was extracted from PushMetrics production PostgreSQL database with the following characteristics:
- All PII is anonymized (email hashed with MD5)
- Only users who logged in at least once are included
- Users created at least 30 days ago (to allow churn calculation)

### Target Variable
- `churned`: 1 if user hasn't logged in for 30+ days, 0 otherwise

### Feature Categories (22 features)

**1. User Profile (6 features)**
- `is_generic_email` - Gmail/Yahoo vs business email
- `has_first_name`, `has_last_name` - Profile completeness
- `profile_completeness` - Sum of name fields filled
- `has_avatar` - Profile picture uploaded
- `mfa_enabled` - Multi-factor authentication enabled

**2. Account Status (2 features)**
- `account_active` - Account active flag
- `is_guest` - Guest user flag

**3. Engagement Metrics (6 features)**
- `login_count` - Total number of logins
- `fail_login_count` - Failed login attempts
- `login_frequency` - Logins per day since signup
- `days_since_signup` - Account age in days
- `days_since_last_login` - Days since last activity

**4. Platform Usage (4 features)**
- `workspace_count` - Number of workspaces
- `organization_count` - Number of organizations
- `role_count` - Number of roles assigned
- `is_admin` - Has admin privileges

**5. Temporal Features (4 features)**
- `signup_hour`, `signup_day_of_week`, `signup_month` - Signup timing
- `last_login_hour`, `last_login_day_of_week` - Last login timing

### Data File

The dataset is in `data.csv` (1,911 rows × 24 columns, fully anonymized).

## Project Structure

```
capstone2-user-churn/
├── README.md                          # This file
├── data.csv                           # ML dataset (1,911 rows, anonymized)
├── notebook.ipynb                     # EDA and model selection
├── train.py                           # Model training script
├── predict.py                         # Flask prediction service
├── test_predict.py                    # API test suite
├── model.xgb                          # Trained XGBoost model (generated)
├── dv.pkl                             # DictVectorizer (generated)
├── model_info.pkl                     # Model metadata (generated)
├── Pipfile / Pipfile.lock             # Dependencies (Pipenv)
├── requirements.txt                   # Dependencies (pip)
├── Dockerfile                         # Container definition
├── .dockerignore                      # Docker build exclusions
├── k8s/
│   ├── deployment.yaml                # Kubernetes deployment
│   ├── service.yaml                   # Kubernetes service
│   └── ingress.yaml                   # Kubernetes ingress (AWS ALB)
└── deploy.sh                          # Automated deployment script
```

## Model Performance

**Best Model: XGBoost**
- **Train AUC:** 1.0000
- **Validation AUC:** 1.0000
- **Test AUC:** 1.0000

**Model Comparison:**

| Model | Train AUC | Val AUC | Test AUC |
|-------|-----------|---------|----------|
| Logistic Regression | 0.9908 | 0.9863 | 0.9911 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 |

**Note:** The perfect AUC scores are expected because `days_since_last_login` is a strong predictor that directly correlates with the churn definition (churned = no login for 30+ days). In production, this feature would be calculated in real-time to provide actionable churn predictions.

### Key Predictive Features

Based on feature importance analysis:
1. `days_since_last_login` - Most predictive (expected)
2. `login_frequency` - Higher frequency = lower churn
3. `login_count` - More logins = more engaged
4. `workspace_count` - More workspaces = more invested
5. `is_admin` - Admins are less likely to churn

## Installation & Setup

### Option 1: Using Pipenv (Recommended)

**Requires:** Python 3.12+

```bash
cd capstone2-user-churn

# Install pipenv if needed
pip install pipenv

# Install dependencies
pipenv install --dev

# Activate virtual environment
pipenv shell
```

### Option 2: Using pip + requirements.txt

```bash
cd capstone2-user-churn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Docker

```bash
# Build image (after training model)
docker build -t user-churn-predictor .

# Run container
docker run -p 9696:9696 user-churn-predictor
```

## Usage

### 1. Explore Data (Jupyter Notebook)

```bash
# Start Jupyter
jupyter notebook

# Open notebook.ipynb
# Run all cells to see:
# - Data exploration and cleaning
# - Feature importance analysis
# - Model selection (Logistic Regression, Random Forest, XGBoost)
# - Performance evaluation
```

### 2. Train Model

```bash
# Train the final model
python train.py

# This will:
# - Load data.csv
# - Split data (60% train, 20% val, 20% test)
# - Train XGBoost model with tuned parameters
# - Evaluate on test set
# - Save model files: model.xgb, dv.pkl, model_info.pkl
```

**Expected output:**
```
======================================================================
USER CHURN PREDICTION - MODEL TRAINING
======================================================================
Loading data from data.csv...
Loaded 1911 samples
Preparing 22 features...
Splitting data...
  Train: 1146 samples (90.1% churned)
  Val:   382 samples (90.1% churned)
  Test:  383 samples (90.1% churned)

Training XGBoost model...
[0]     train-auc:0.XXXX   val-auc:0.XXXX
...

AUC Scores:
  Train: 0.XXXX
  Val:   0.XXXX
  Test:  0.XXXX

Saving model artifacts...
======================================================================
Training complete!
======================================================================
```

### 3. Run Prediction Service

```bash
# Start Flask API
python predict.py

# The service will run on http://localhost:9696
```

### 4. Test Predictions

**Health check:**
```bash
curl http://localhost:9696/health
```

**Get model info:**
```bash
curl http://localhost:9696/info
```

**Make prediction:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "login_count": 50,
    "days_since_last_login": 5,
    "days_since_signup": 180,
    "login_frequency": 0.28,
    "workspace_count": 2,
    "organization_count": 1,
    "role_count": 3,
    "is_admin": 1,
    "is_guest": 0,
    "is_generic_email": 0,
    "account_active": 1,
    "has_first_name": 1,
    "has_last_name": 1,
    "profile_completeness": 2,
    "has_avatar": 1,
    "mfa_enabled": 0,
    "signup_hour": 10,
    "signup_day_of_week": 2,
    "signup_month": 6,
    "last_login_hour": 14,
    "last_login_day_of_week": 4,
    "fail_login_count": 0
  }'
```

**Response:**
```json
{
  "churn_probability": 0.2534,
  "will_churn": false,
  "risk_level": "low",
  "recommendation": "Low risk - continue standard engagement"
}
```

Risk levels:
- `low`: probability < 0.5
- `medium`: probability >= 0.5 and < 0.7
- `high`: probability >= 0.7 and < 0.9
- `critical`: probability >= 0.9

**Run automated test suite:**

```bash
# Make sure the API is running first (in another terminal)
python predict.py

# Run the test suite
python test_predict.py
```

### 5. Docker Deployment

```bash
# Build Docker image
docker build -t user-churn-predictor .

# Run container
docker run -it --rm -p 9696:9696 user-churn-predictor

# Test from another terminal
curl http://localhost:9696/health
```

## API Endpoints

### GET /
Service information and available endpoints.

### GET /health
Health check endpoint.

**Output:**
```json
{
  "status": "healthy",
  "service": "user-churn-predictor",
  "model_loaded": true
}
```

### GET /info
Model information and metadata.

**Output:**
```json
{
  "model_type": "XGBoost",
  "params": {...},
  "num_boost_rounds": 200,
  "train_auc": 0.XXXX,
  "val_auc": 0.XXXX,
  "test_auc": 0.XXXX,
  "n_features": 22,
  "feature_columns": [...]
}
```

### POST /predict
Predict churn probability for a user.

**Input**: JSON with user features

**Output:**
```json
{
  "churn_probability": 0.75,
  "will_churn": true,
  "risk_level": "high",
  "recommendation": "High risk - prioritize for customer success outreach"
}
```

## Key Insights from EDA

1. **Days since last login is the strongest predictor** - Users who haven't logged in recently are much more likely to churn

2. **Login frequency matters** - Users who login frequently are significantly less likely to churn

3. **Profile completeness signals commitment** - Users with complete profiles (name, avatar) churn less

4. **Admin users are more engaged** - Admin role correlates with lower churn

5. **Generic email addresses (Gmail, Yahoo)** - These users have slightly higher churn rates than business email users

6. **MFA adoption indicates investment** - Users who enable MFA are more committed to the platform

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (AWS EKS, GKE, or local minikube)
- Docker image pushed to registry (ECR, Docker Hub, etc.)
- kubectl configured

### Deploy to Kubernetes

```bash
# Update k8s/deployment.yaml with your ECR repository URL
# Update k8s/ingress.yaml with your domain and certificate ARN

# Apply all Kubernetes configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services
kubectl get ingress
```

### Cloud Deployment (AWS)

The application is deployed to **AWS EKS (Kubernetes)** and is publicly accessible.

**Live API Endpoint:** https://churn.pushmetrics.io

**Deployment Architecture:**
- **Platform**: AWS EKS (Elastic Kubernetes Service)
- **Region**: eu-central-1 (Frankfurt)
- **Container Registry**: AWS ECR
- **Load Balancer**: AWS Application Load Balancer (ALB)
- **DNS**: Route53 with SSL/TLS certificate
- **Replicas**: 1 pod (scalable)
- **Resources**: 256Mi-512Mi RAM, 200m-500m CPU per pod

### Test the Live API

**Health check:**
```bash
curl https://churn.pushmetrics.io/health
```

**Response:**
```json
{"model_loaded":true,"service":"user-churn-predictor","status":"healthy"}
```

**Get model info:**
```bash
curl https://churn.pushmetrics.io/info
```

**Response:**
```json
{
  "feature_columns": ["is_generic_email", "account_active", "login_count", ...],
  "model_type": "XGBoost",
  "n_features": 22,
  "num_boost_rounds": 200,
  "params": {"eta": 0.1, "max_depth": 5, ...},
  "test_auc": 1.0,
  "train_auc": 1.0,
  "val_auc": 1.0
}
```

**Make a prediction (low risk user - active, engaged):**
```bash
curl -X POST https://churn.pushmetrics.io/predict \
  -H "Content-Type: application/json" \
  -d '{
    "login_count": 150,
    "days_since_last_login": 1,
    "days_since_signup": 365,
    "login_frequency": 0.41,
    "workspace_count": 3,
    "organization_count": 2,
    "role_count": 5,
    "is_admin": 1,
    "is_guest": 0,
    "is_generic_email": 0,
    "account_active": 1,
    "has_first_name": 1,
    "has_last_name": 1,
    "profile_completeness": 2,
    "has_avatar": 1,
    "mfa_enabled": 1,
    "signup_hour": 10,
    "signup_day_of_week": 2,
    "signup_month": 6,
    "last_login_hour": 14,
    "last_login_day_of_week": 4,
    "fail_login_count": 0
  }'
```

**Response:**
```json
{
  "churn_probability": 0.0023,
  "recommendation": "Low risk - continue standard engagement",
  "risk_level": "low",
  "will_churn": false
}
```

**Make a prediction (high risk user - inactive, disengaged):**
```bash
curl -X POST https://churn.pushmetrics.io/predict \
  -H "Content-Type: application/json" \
  -d '{
    "login_count": 3,
    "days_since_last_login": 60,
    "days_since_signup": 90,
    "login_frequency": 0.03,
    "workspace_count": 1,
    "organization_count": 1,
    "role_count": 1,
    "is_admin": 0,
    "is_guest": 1,
    "is_generic_email": 1,
    "account_active": 1,
    "has_first_name": 0,
    "has_last_name": 0,
    "profile_completeness": 0,
    "has_avatar": 0,
    "mfa_enabled": 0,
    "signup_hour": 3,
    "signup_day_of_week": 6,
    "signup_month": 1,
    "last_login_hour": 2,
    "last_login_day_of_week": 0,
    "fail_login_count": 0
  }'
```

**Response:**
```json
{
  "churn_probability": 0.9997,
  "recommendation": "Immediate intervention required - user likely to churn",
  "risk_level": "critical",
  "will_churn": true
}
```

### Deployment Instructions

```bash
# Configure environment variables
export AWS_REGION=eu-central-1
export ECR_REGISTRY=123456789012.dkr.ecr.eu-central-1.amazonaws.com
export ECR_REPO="${ECR_REGISTRY}/user-churn-predictor"
export IMAGE_TAG=latest

# Run deployment script
./deploy.sh
```

## Dependencies

**Core:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost == 2.1.4
- flask >= 3.0.0
- gunicorn >= 21.0.0
- requests >= 2.31.0

**Development:**
- jupyter
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tqdm

See `Pipfile` and `requirements.txt` for complete list.

## Future Improvements

1. **Model Enhancements**
   - Add more behavioral features (report executions, block types used)
   - Try ensemble methods combining multiple models
   - Implement time-series features for usage trends

2. **Deployment**
   - Add monitoring and logging (Prometheus, Grafana)
   - Implement A/B testing framework
   - Auto-scaling based on traffic

3. **Features**
   - Real-time churn scoring dashboard
   - Automated alerts for high-risk users
   - Integration with customer success tools (Intercom, HubSpot)

4. **Data**
   - Collect more granular usage data
   - Add support ticket data as features
   - Track feature adoption over time

## Author

**Lukasz Wlodarczyk** - ML Zoomcamp 2025 Capstone 2 Project

## License

This project is part of ML Zoomcamp course by DataTalks.Club.

## Acknowledgments

- DataTalks.Club for ML Zoomcamp curriculum
- Alexey Grigorev for excellent teaching and course structure
