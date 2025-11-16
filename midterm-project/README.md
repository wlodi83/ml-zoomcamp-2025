# Report Failure Prediction - ML Midterm Zoomcamp Project

## Problem Description

This project predicts whether a report execution will **fail or succeed** in the [PushMetrics](https://pushmetrics.io) platform using machine learning.

### Business Context

PushMetrics is an automated reporting platform that:
- Executes SQL queries against various databases (BigQuery, Snowflake, ClickHouse, Exasol, etc.)
- Generates visualizations (Plotly charts, tables, KPI cards, Tableau Dashboards)
- Delivers results via email, Slack, or cloud storage (S3, GCS, Azure Blob)
- Runs reports on schedules or on-demand (manually or via webhooks)

### Why Predict Failures?

- **Proactive Alerts**: Warn users before a report is likely to fail
- **Resource Optimization**: Avoid wasting compute resources on failing reports
- **Improved Reliability**: Identify patterns that lead to failures
- **Better UX**: Reduce customer support tickets

### ML Problem

- **Type**: Binary classification
- **Target**: `failed` (0 = success, 1 = failure)
- **Features**: Report structure (13 block types), historical performance, timing, database types
- **Dataset**: 441,810 report executions with 6.13% failure rate
- **Metric**: ROC AUC (handles class imbalance well)
- **Best Model**: XGBoost with test AUC = 0.9857

## Dataset

### Data Collection

Data was extracted from production PostgreSQL database using a SQL query that:
- Anonymizes sensitive data with MD5 hashing (report_id, workspace_id)
- Aggregates historical failure statistics (60-day lookback window)
- Extracts report structure features (counts of different block types)
- Captures execution context (time of day, scheduled vs manual, reruns)
- Includes database types used in the report

### Features (25 total)

**Report Structure (13 features):**
- `num_blocks` - Total number of blocks in report
- `num_sql_blocks` - SQL query blocks
- `num_writeback_blocks` - Database write operations
- `num_viz_blocks` - Visualization blocks (Plotly, tables, KPIs)
- `num_tableau_blocks` - Tableau embeds, Tableau Dashboards, Tableau Views
- `num_email_blocks` - Email delivery blocks
- `num_slack_blocks` - Slack message blocks
- `num_api_blocks` - API request blocks
- `num_sftp_blocks` - SFTP file transfer blocks
- `num_storage_blocks` - Cloud storage blocks (S3, GCS, Azure)
- `num_control_blocks` - Control flow (if/else, loops)
- `num_parameters` - Input parameters
- `num_databases` - Number of unique databases

**Historical Performance (5 features):**
- `historical_failure_count` - Number of past failures (60 days)
- `historical_executions` - Total past executions (60 days)
- `historical_failure_rate` - Failure rate (0-1)
- `avg_historical_duration` - Average duration in seconds
- `hours_since_last_success` - Time since last successful run

**Timing (6 features):**
- `hour_of_day` - Hour (0-23)
- `day_of_week` - Day (0=Sunday, 6=Saturday)
- `is_weekend` - Weekend flag (0/1)
- `is_business_hours` - Business hours flag (9-17)
- `is_rerun` - Manual rerun flag (0/1)
- `is_scheduled` - Scheduled execution flag (0/1)

**Database (1 feature):**
- `database_types` - Comma-separated database types (e.g., "bigquery,snowflake")

### Data File

The dataset is in `report_executions.csv` (93 MB, 441,810 rows).

**Note**: This file is NOT committed to git (too large). You can:
1. Download it from https://drive.google.com/file/d/1KUUlbMZsZlwNAYD311RbfF3O27cybVAj/view?usp=sharing

## Project Structure

```
ml/
├── README.md                 # This file
├── DEPLOYMENT.md             # Cloud deployment guide
├── TESTING_EVIDENCE.md       # Live deployment testing proof
├── notebook.ipynb            # EDA, feature analysis, model selection
├── train.py                  # Training script
├── predict.py                # Flask prediction API
├── test_api.py               # API test suite
├── deploy.sh                 # Automated deployment script
├── Pipfile                   # Dependency management (Pipenv)
├── Pipfile.lock              # Locked dependencies
├── requirements.txt          # Dependencies list
├── Dockerfile                # Container definition
├── k8s/                      # Kubernetes manifests
│   ├── deployment.yaml       # K8s deployment config
│   ├── service.yaml          # K8s service config
│   └── ingress.yaml          # K8s ingress (ALB) config
├── report_executions.csv     # Dataset (not in git)
├── model.xgb                 # Trained XGBoost model (generated)
├── dv.pkl                    # DictVectorizer (generated)
└── model_info.pkl            # Model metadata (generated)
```

## Installation & Setup

### Option 1: Using Pipenv (Recommended)

```bash
# Install pipenv if needed
pip install pipenv

# Install dependencies
cd midterm-project
pipenv install --dev

# Activate virtual environment
pipenv shell
```

### Option 2: Using pip + requirements.txt

```bash
# Generate requirements.txt from Pipfile
pipenv requirements > requirements.txt

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Docker

```bash
# Build image (after training model)
docker build -t report-failure-prediction .

# Run container
docker run -p 9696:9696 report-failure-prediction
```

## Usage

### 1. Explore Data (Jupyter Notebook)

```bash
# Start Jupyter
pipenv run jupyter notebook

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
# - Load report_executions.csv
# - Split data (60% train, 20% val, 20% test)
# - Train XGBoost model with tuned parameters
# - Evaluate on test set
# - Save model files: model.xgb, dv.pkl, model_info.pkl
```

**Expected output:**
```
Training set:   265,086 rows (60.0%)
  Failure rate: 6.10%
Validation set:  88,362 rows (20.0%)
  Failure rate: 6.11%
Test set:        88,362 rows (20.0%)
  Failure rate: 6.26%

Training XGBoost model...
[0]     train-auc:0.97433   val-auc:0.97329
[50]    train-auc:0.99312   val-auc:0.98731
[100]   train-auc:0.99635   val-auc:0.98782
[150]   train-auc:0.99739   val-auc:0.98793
[199]   train-auc:0.99807   val-auc:0.98796

Training AUC:    0.9981
Validation AUC:  0.9880
Test AUC:        0.9857
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

**Model info:**
```bash
curl http://localhost:9696/info
```

**Make prediction:**
```bash
curl -X POST http://localhost:9696/predict \
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

**Response:**
```json
{
  "failure_probability": 0.0234,
  "failure": false,
  "risk_level": "low"
}
```

Risk levels:
- `low`: probability < 0.25
- `medium`: probability < 0.5
- `high`: probability < 0.75
- `critical`: probability ≥ 0.75

**Run automated test suite:**

The project includes an automated test suite that validates all API endpoints:

```bash
# Make sure the API is running first (in another terminal)
python predict.py

# Run the test suite
python test_api.py
```

The test suite will:
- Test the `/health` endpoint
- Test the `/info` endpoint
- Test predictions with 3 different scenarios:
  - Low risk report (simple, good history)
  - High risk report (complex, bad history)
  - Medium risk report

Expected output:
```
============================================================
TEST SUMMARY
============================================================
✓ PASS: Health Check
✓ PASS: Model Info
✓ PASS: Low Risk Report (Simple, Good History)
✓ PASS: High Risk Report (Complex, Bad History)
✓ PASS: Medium Risk Report

Total: 5/5 tests passed

🎉 All tests passed!
```

### 5. Docker Deployment

```bash
# Build Docker image
docker build -t report-failure-prediction .

# Run container
docker run -it --rm -p 9696:9696 report-failure-prediction

# Test from another terminal
curl http://localhost:9696/health
```

## Model Development Process

### 1. Data Preparation
- Train/validation/test split (60/20/20) with random_state=1
- Handles missing values (fill with 0)
- One-hot encoding for categorical features (DictVectorizer)

### 2. Models Evaluated
1. **Logistic Regression** (baseline)
   - Simple linear model with K-Fold cross-validation (5 folds)
   - Hyperparameter tuning for C parameter [0.001, 0.01, 0.1, 1, 5, 10]
   - Best C=5 with validation AUC: 0.9645
   - Fast training and inference

2. **Random Forest**
   - Tree ensemble, captures non-linear relationships
   - Grid search over hyperparameters:
     - n_estimators: [10, 50, 100, 200]
     - max_depth: [5, 10, 15, 20]
     - min_samples_leaf: [1, 3, 5, 10]
   - Best params: n_estimators=200, max_depth=15, min_samples_leaf=3
   - Validation AUC: 0.9893

3. **XGBoost** (best model)
   - Gradient boosting with sequential parameter tuning
   - Tuned hyperparameters:
     - eta: [0.3, 1.0, 0.1, 0.05, 0.01] → best: 0.1
     - max_depth: [6, 3, 4, 10] → best: 10
     - min_child_weight: [1, 10, 30] → best: 1
   - 200 boosting rounds
   - Training AUC: 0.9981, Validation AUC: 0.9880, Test AUC: 0.9857

### 3. Feature Importance
Top features contributing to predictions (XGBoost):
1. Historical failure rate
2. Is scheduled
3. Is rerun
4. Historical failure count
5. Database types (especially Vertica)
6. Number of parameters

### 4. Performance Metrics
- **Primary metric**: ROC AUC
- **Training AUC**: 0.9981
- **Validation AUC**: 0.9880
- **Test AUC**: 0.9857
- **Overfitting**: 0.0101 difference between train and validation

## API Endpoints

### POST /predict
Predict failure probability for a report execution.

**Input**: JSON with 25 features (see example above)

**Output**:
```json
{
  "failure_probability": 0.1234,
  "failure": false,
  "risk_level": "low"
}
```

### GET /health
Health check endpoint.

**Output**:
```json
{
  "status": "healthy",
  "model": "XGBoost",
  "test_auc": 0.9857
}
```

### GET /info
Model information and metadata.

**Output**:
```json
{
  "model_type": "XGBoost",
  "params": {
    "eta": 0.1,
    "max_depth": 10,
    "min_child_weight": 1,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "seed": 1
  },
  "num_boost_rounds": 200,
  "train_auc": 0.9981,
  "val_auc": 0.9880,
  "test_auc": 0.9857,
  "n_features": 39,
  "feature_cols": [...]
}
```

## Dependencies

**Core:**
- pandas 1.5.3
- numpy 1.23.5
- scikit-learn 1.2.2
- xgboost 1.7.5
- flask 2.3.2
- gunicorn 21.2.0
- tqdm

**Development:**
- jupyter
- matplotlib
- seaborn

See `Pipfile` for complete list.

## Cloud Deployment

The application is deployed to **AWS EKS (Kubernetes)** and publicly accessible at:

🌐 **https://ml.pushmetrics.io**

### Live Endpoints

- **Health Check**: https://ml.pushmetrics.io/health
- **Model Info**: https://ml.pushmetrics.io/info
- **Predictions**: POST to https://ml.pushmetrics.io/predict

### Testing the Live API

**Health check:**
```bash
curl https://ml.pushmetrics.io/health
```

**Get model information:**
```bash
curl https://ml.pushmetrics.io/info
```

**Make a prediction:**
```bash
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

### Deployment Architecture

- **Platform**: AWS EKS (Elastic Kubernetes Service)
- **Region**: eu-central-1 (Frankfurt)
- **Container Registry**: AWS ECR
- **Load Balancer**: AWS Application Load Balancer (ALB)
- **DNS**: Route53 with SSL/TLS certificate
- **Replicas**: 1 pod (scalable)
- **Resources**: 256Mi-512Mi RAM, 200m-500m CPU per pod

### Deployment Instructions

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

Quick deployment:
```bash
# Configure and run deployment script
./deploy.sh
```

The deployment includes:
- Kubernetes manifests in `k8s/` directory
- Automated deployment script (`deploy.sh`)
- HTTPS with AWS Certificate Manager
- Health checks and auto-recovery
- Route53 DNS configuration

### Testing Evidence

The API has been successfully deployed and tested. See [TESTING_EVIDENCE.md](TESTING_EVIDENCE.md) for:
- ✅ Live endpoint test results (health, info, predictions)
- ✅ Kubernetes deployment status
- ✅ DNS and SSL/TLS configuration verification
- ✅ Performance and availability metrics

## Future Improvements

1. **Model Enhancements**
   - Try other algorithms
   - More feature engineering (interaction features)

2. **Deployment**
   - Add monitoring and logging
   - A/B testing framework

3. **Features**
   - Real-time predictions before execution
   - Automated retraining pipeline

4. **Data**
   - Collect more features (query complexity, data volume, report update)
   - Longer historical window (6 months → 1 year)

## Author

**Lukasz Wlodarczyk** - ML Midterm Zoomcamp 2025 Project

## License

This project is part of ML Midterm Zoomcamp course by DataTalks.Club.

## Acknowledgments

- DataTalks.Club for ML Midterm Zoomcamp curriculum
