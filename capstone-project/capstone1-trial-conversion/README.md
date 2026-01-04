# Trial-to-Paid Conversion Prediction

**ML Zoomcamp 2025 - Capstone 1 Project Lukasz Wlodarczyk**

## Problem Description

This project predicts whether users in a 14-day trial period will convert to paid subscriptions for **PushMetrics** (pushmetrics.io), a SaaS platform for automated reporting and data delivery.

**Business Value:**
- Identify high-risk trial users before they churn
- Enable targeted interventions (sales outreach, onboarding assistance)
- Optimize onboarding process based on conversion patterns
- Potential to increase conversion rate by 10-15%

### ML Problem

- **Type**: Binary classification
- **Target**: `converted_to_paid` (1 = converted to paid, 0 = churned)
- **Features**: 57 behavioral features from trial period
- **Dataset**: 237 trial conversions with 43.04% conversion rate
- **Metric**: ROC AUC (handles class imbalance well)
- **Best Model**: Random Forest (n_estimators=50, max_depth=5, min_samples_leaf=5) with validation AUC = 0.8289, test AUC = 0.6499

**Dataset:**
- **237 trial conversions** from Stripe API (2017-2025)
- **102 converted (43.04%)**, 135 churned (56.96%)
- **64 columns total**, 57 features used for modeling (after excluding IDs and dates)
- Data combines Stripe subscription data with PostgreSQL usage analytics (fully anonymized)

## Project Structure

```
capstone1-trial-conversion/
├── README.md                          # This file
├── data.csv                           # ML dataset (237 rows × 64 features, anonymized)
├── notebook.ipynb                     # EDA and model selection
├── train.py                           # Model training script
├── predict.py                         # Flask prediction service
├── test_predict.py                    # API test suite
├── model.pkl                          # Trained Random Forest model (generated)
├── dv.pkl                             # DictVectorizer (generated)
├── Pipfile / Pipfile.lock            # Dependencies (Pipenv)
├── requirements.txt                   # Dependencies (pip)
├── Dockerfile                         # Container definition
├── k8s/
│   ├── deployment.yaml               # Kubernetes deployment
│   ├── service.yaml                  # Kubernetes service
│   └── ingress.yaml                  # Kubernetes ingress (AWS ALB)
```

## Dataset

### Data Collection

Data was created using a two-step pipeline:
1. **Stripe API extraction**: Extracted all customer subscriptions to identify trial periods and conversion outcomes (accurate trial start/end dates)
2. **PostgreSQL feature extraction**: Joined Stripe trial dates with usage analytics to extract behavioral features during trial period

The dataset is in `data.csv` (59 KB, 237 rows × 64 features, fully anonymized).

### Target Variable
- `converted_to_paid`: 1 = converted to paid, 0 = churned

### Feature Categories

**1. Onboarding Speed (Most Predictive)**
- `hours_to_first_execution` - Time to first report execution
- `hours_to_first_report` - Time to first report creation
- `hours_to_first_sql_connection` - Time to database connection
- `activated_day1`, `activated_day2` - Quick activation flags

**2. Usage Intensity**
- `reports_created`, `executions_total`, `blocks_created`
- `executions_week1`, `executions_week2` - Weekly usage
- `execution_growth_rate` - Usage trend

**3. Success/Quality**
- `execution_success_rate` - % of successful executions
- `avg_execution_duration` - Report performance

**4. Feature Adoption**
- Block types: `has_sql_block`, `has_plotly_block`, `has_kpi_block`, etc.
- Advanced features: `has_parameters`, `has_control_flow`

**5. Database Integrations**
- `sql_connections_count`, `unique_db_types_count`
- Enterprise DBs: `has_bigquery`, `has_snowflake`, `has_redshift`

**6. Automation**
- `schedules_created`, `schedules_active`
- `created_schedule` - Did they automate?

**7. Team Collaboration**
- `users_invited`, `active_users_count`
- `team_engagement_rate`

**8. User Profile**
- `is_generic_email` - Gmail vs business email
- Temporal: `trial_start_hour`, `trial_start_day_of_week`

## Model Performance

**Best Model: Random Forest**
- **Validation AUC:** 0.8289
- **Test AUC:** 0.6499
- **Train AUC:** 0.8750

**Model Comparison:**

| Model | Train AUC | Val AUC | Test AUC |
|-------|-----------|---------|----------|
| Logistic Regression | 0.8272 | 0.7637 | 0.5291 |
| Random Forest | 0.8750 | **0.8289** | 0.6499 |
| XGBoost | 1.0000 | 0.7637 | 0.6737 |

**Key Findings:**
- Random Forest achieved the best validation AUC (0.8289) with moderate overfitting (n_estimators=50, max_depth=5, min_samples_leaf=5)
- XGBoost showed perfect training AUC (1.00) but higher overfitting (Val: 0.7637)
- Test AUC for Random Forest: 0.6499 (48-sample test set)
- Top RF features include `users_invited`, `is_generic_email`, `trial_start_hour`, `trial_start_day_of_week`, `hours_to_first_execution`, `execution_growth_rate`, and `reports_created`
- Usage intensity and scheduling signals (`executions_week1`, `reports_week1`, `schedules_active`) are also among the most important

## Installation & Setup

### Option 1: Using Pipenv (Recommended)

**Requires:** Python 3.12+

```bash
cd capstone1-trial-conversion

# Install pipenv if needed
pip install pipenv

# Install dependencies
pipenv install --dev

# Activate virtual environment
pipenv shell
```

### Option 2: Using pip + requirements.txt

```bash
cd capstone1-trial-conversion

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Docker

```bash
# Build image (after training model)
docker build -t trial-conversion-predictor .

# Run container
docker run -p 9696:9696 trial-conversion-predictor
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
# - Train Random Forest model with tuned parameters
# - Evaluate on test set
# - Save model files: model.pkl, dv.pkl
```

**Expected output:**
```
================================================================================
TRIAL-TO-PAID CONVERSION PREDICTION - MODEL TRAINING
================================================================================

Loading data from data.csv...
✓ Loaded 237 samples

Preparing features...
✓ Prepared 57 features

Splitting data...
✓ Train: 141 samples (43.0% converted)
✓ Val:   48 samples (42.6% converted)
✓ Test:  48 samples (43.8% converted)

Training Random Forest model...

AUC Scores:
  Train: 0.8750
  Val:   0.8289
  Test:  0.6499

✓ Saved model to: model.pkl
✓ Saved vectorizer to: dv.pkl
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

**Make prediction:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "users_invited": 5,
    "is_generic_email": 0,
    "trial_start_hour": 10,
    "trial_start_day_of_week": 2,
    "hours_to_first_execution": 2.5,
    "execution_growth_rate": 0.15,
    "reports_created": 5,
    "executions_total": 25,
    "execution_success_rate": 0.92,
    "activated_day1": 1,
    "activated_day2": 1,
    "reports_week1": 1,
    "executions_week1": 5,
    "schedules_active": 1,
    "has_sql_block": 1,
    "has_bigquery": 1,
    "schedules_created": 2,
    "sql_connections_count": 1,
    "blocks_created": 15
  }'
```

**Response:**
```json
{
  "conversion_probability": 0.4503,
  "will_convert": false,
  "risk_level": "medium",
  "recommendation": "Moderate risk - consider sales outreach or onboarding assistance"
}
```

Risk levels:
- `low`: probability >= 0.5
- `medium`: probability >= 0.32
- `high`: probability < 0.32

**Run automated test suite:**

The project includes an automated test suite that validates all API endpoints:

```bash
# Make sure the API is running first (in another terminal)
python predict.py

# Run the test suite
python test_predict.py
```

The test suite will test predictions with 3 different scenarios:
- High conversion probability user
- Medium conversion probability user
- Low conversion probability user

### 5. Docker Deployment

```bash
# Build Docker image
docker build -t trial-conversion-predictor .

# Run container
docker run -it --rm -p 9696:9696 trial-conversion-predictor

# Test from another terminal
curl http://localhost:9696/health
```

## Model Development Process

### 1. Data Preparation
- Train/validation/test split (60/20/20) with random_state=42
- Handles missing values (fill with 0 or 999 for time-based features)
- One-hot encoding for categorical features (DictVectorizer)

### 2. Models Evaluated
1. **Logistic Regression** (baseline)
   - Simple linear model with K-Fold cross-validation (5 folds)
   - Best C=1 with mean AUC: 0.6543
   - Training AUC: 0.8272, Validation AUC: 0.7637, Test AUC: 0.5291
   - Low overfitting (train-val diff: 0.0635)

2. **Random Forest** (best model)
   - Tree ensemble with grid search over hyperparameters
   - Best params: n_estimators=50, max_depth=5, min_samples_leaf=5
   - Training AUC: 0.8750, Validation AUC: 0.8289, Test AUC: 0.6499
   - Moderate overfitting (train-val diff: 0.0461)
   - Best validation performance

3. **XGBoost**
   - Gradient boosting with sequential parameter tuning
   - Best params: eta=0.1, max_depth=10, min_child_weight=1
   - Training AUC: 1.0000, Validation AUC: 0.7637, Test AUC: 0.6737
   - Significant overfitting (train-val diff: 0.2363)

### 3. Feature Importance

**From EDA (Correlation with Target):**
1. `hours_to_first_execution` - Time to first report execution
2. `activated_day1` - Quick activation indicator
3. `reports_created` - Engagement level
4. `execution_success_rate` - Quality of user experience
5. `has_bigquery`/`has_snowflake` - Enterprise database indicators

**From Random Forest Model (Top 10 by Importance):**
1. `users_invited` (0.247) - Number of team members invited - strongest predictor of conversion
2. `is_generic_email` (0.096) - Business email vs generic (Gmail) - enterprise signal
3. `trial_start_hour` (0.069) - Time of day trial started - behavioral pattern
4. `trial_start_day_of_week` (0.063) - Day of week trial started
5. `hours_to_first_execution` (0.053) - Speed to first report execution
6. `execution_growth_rate` (0.028) - Usage trend during trial
7. `blocks_created` (0.027) - Content creation activity
8. `trial_duration_days` (0.027) - Length of trial period
9. `executions_week1` (0.026) - First week engagement
10. `reports_created` (0.024) - Number of reports created

**Key Insight:** Team collaboration (`users_invited`) is the strongest predictor, even more important than product usage metrics. This suggests that trials converting to paid are those where users are building solutions for their team, not just individual experimentation.

### 4. Performance Metrics
- **Primary metric**: ROC AUC (handles class imbalance well)
- **Best Model**: Random Forest
- **Training AUC**: 0.8750
- **Validation AUC**: 0.8289
- **Test AUC**: 0.6499
- **Overfitting**: 0.0461 (train-val difference)

**Note**: Test AUC is lower than validation AUC due to small dataset size (48 test samples). The model generalizes reasonably on validation but shows variance on the small test set.

## API Endpoints

### POST /predict
Predict conversion probability for a trial user.

**Input**: JSON with features (see example in Usage section)

**Output**:
```json
{
  "conversion_probability": 0.87,
  "will_convert": true,
  "risk_level": "low",
  "recommendation": "High conversion likelihood - standard onboarding"
}
```

### GET /health
Health check endpoint.

**Output**:
```json
{
  "status": "healthy",
  "service": "trial-conversion-predictor"
}
```

## Key Insights from EDA

1. **Onboarding Speed is Critical**
   - Median time to first execution: 999 hours (many users never execute)
   - Users who execute within 24h: 20.5% conversion
   - Users who never execute: 48.6% conversion

2. **Failure Signals Matter**
   - `executions_failed` counts failed report runs during trial; high values suggest users are trying to set up but hit errors. Improving onboarding/stability here can lift conversion.

3. **Usage Patterns**
   - Users who invite more teammates convert more often
   - Conversion by users invited: 0 (4.9%), 1 (37.5%), 2-3 (53.8%), 4-5 (68.2%), 6-10 (64.3%), 11+ (67.6%)

4. **Enterprise Indicators**
   - BigQuery users: 100% conversion (small sample)
   - Generic email users: 4.8% conversion vs business email users: 51.3%

5. **Automation Adoption**
   - Users who create schedules: 35.7% conversion
   - Users without schedules: 44.0% conversion

## Kubernetes Deployment (Optional)

### Prerequisites
- Kubernetes cluster (AWS EKS, GKE, or local minikube)
- Docker image pushed to registry (ECR, Docker Hub, etc.)

### Deploy to Kubernetes

```bash
# Update k8s/deployment.yaml with your ECR repository URL
# Update k8s/ingress.yaml with your domain and certificate ARN

# Apply all Kubernetes configurations
kubectl apply -f k8s/

# Or apply individually
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services
kubectl get ingress
```

### Deployment Architecture

```
User Request
    ↓
AWS Load Balancer (ALB)
    ↓
Kubernetes Service (ClusterIP)
    ↓
Prediction Pods (Flask + Random Forest)
    ↓
Response (probability + risk level)
```

### Deployment Architecture (AWS)

- **Platform**: AWS EKS (Elastic Kubernetes Service)
- **Region**: eu-central-1 (Frankfurt)
- **Container Registry**: AWS ECR
- **Load Balancer**: AWS Application Load Balancer (ALB)
- **DNS**: Route53 with SSL/TLS certificate
- **Replicas**: 1 pod (scalable)
- **Resources**: 256Mi-512Mi RAM, 200m-500m CPU per pod

### Cloud Deployment

The application is deployed to **AWS EKS (Kubernetes)** and publicly accessible at:

🌐 **https://trial-conversion.pushmetrics.io** (replace with your dedicated subdomain)

#### Live Endpoints

- **Health Check**: https://trial-conversion.pushmetrics.io/health
- **Service Info**: https://trial-conversion.pushmetrics.io/
- **Predictions**: POST to https://trial-conversion.pushmetrics.io/predict

#### Testing the Live API

**Health check:**
```bash
curl https://trial-conversion.pushmetrics.io/health
```

**Get service info:**
```bash
curl https://trial-conversion.pushmetrics.io/
```

**Make a prediction:**
```bash
curl -X POST https://trial-conversion.pushmetrics.io/predict \
  -H "Content-Type: application/json" \
  -d '{
    "users_invited": 5,
    "is_generic_email": 0,
    "trial_start_hour": 10,
    "trial_start_day_of_week": 2,
    "hours_to_first_execution": 2.5,
    "execution_growth_rate": 0.15,
    "reports_created": 5,
    "executions_total": 25,
    "execution_success_rate": 0.92,
    "activated_day1": 1,
    "activated_day2": 1,
    "reports_week1": 1,
    "executions_week1": 5,
    "schedules_active": 1,
    "has_sql_block": 1,
    "has_bigquery": 1,
    "schedules_created": 2,
    "sql_connections_count": 1,
    "blocks_created": 15
  }'
```

### Deployment Instructions

For detailed deployment instructions, see `DEPLOYMENT.md`.

Quick deployment:
```bash
# Configure and run deployment script
./deploy.sh
```

Set these env vars before running `deploy.sh`:
```bash
export AWS_REGION=eu-central-1
export ECR_REGISTRY=123456789012.dkr.ecr.eu-central-1.amazonaws.com
export ECR_REPO="${ECR_REGISTRY}/trial-conversion-predictor"
export IMAGE_TAG=latest
```

What `deploy.sh` does:
- Builds and tags the Docker image locally (linux/amd64)
- Logs into ECR
- Pushes `${ECR_REPO}:${IMAGE_TAG}`
- Updates `k8s/deployment.yaml` to use the new image tag
- Applies all manifests in `k8s/`

The deployment includes:
- Kubernetes manifests in `k8s/` directory
- Automated deployment script (`deploy.sh`)
- HTTPS with AWS Certificate Manager
- Health checks and auto-recovery
- Route53 DNS configuration

### Testing Evidence

The API has been successfully deployed and tested. See `TESTING_EVIDENCE.md` for:
- ✅ Live endpoint test results (health, info, predictions)
- ✅ Kubernetes deployment status
- ✅ DNS and SSL/TLS configuration verification
- ✅ Performance and availability metrics

## Future Improvements

1. **Model Enhancements**
   - More feature engineering (interaction features, polynomial features)
   - Hyperparameter tuning with larger search space
   - Try ensemble methods

2. **Deployment**
   - Add monitoring and logging (Prometheus, Grafana)
   - A/B testing framework for model versions
   - Auto-scaling based on traffic

3. **Features**
   - Real-time risk scoring dashboard
   - Automated intervention triggers for high-risk users
   - Integration with CRM systems

4. **Data**
   - Collect more trial conversions over time
   - Add more behavioral features (page views, support tickets)
   - Experiment with longer trial periods (30 days)

## Dependencies

**Core:**
- pandas 2.3.3
- numpy 2.0.2
- scikit-learn 1.6.1
- xgboost 2.1.4
- flask 3.1.2
- gunicorn 23.0.0
- tqdm 4.67.1

**Development:**
- jupyter 1.1.1
- matplotlib 3.9.4
- seaborn 0.13.2

**Deployment:**
- Docker
- Kubernetes (kubectl)

See `Pipfile` and `requirements.txt` for complete list.

## Author

**Lukasz Wlodarczyk** - ML Zoomcamp 2025 Capstone 1 Project

## License

This project is part of ML Zoomcamp course by DataTalks.Club.

## Acknowledgments

- DataTalks.Club for ML Zoomcamp curriculum
- Alexey Grigorev for excellent teaching and course structure
