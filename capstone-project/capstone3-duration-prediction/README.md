# Workflow Execution Duration Prediction

**ML Zoomcamp 2025 - Capstone Project 3**

Author: Lukasz Wlodarczyk

## Problem Description

This project predicts how long a workflow (noteflow) execution will take to complete on the PushMetrics platform. Unlike classification problems that predict binary outcomes, this is a **regression** task where we predict continuous duration values in seconds.

### Business Value

- **Resource Planning**: Allocate compute resources based on predicted duration
- **User Experience**: Show estimated completion time to users before they run workflows
- **Scheduling Optimization**: Schedule long-running workflows during off-peak hours
- **SLA Management**: Alert users when workflows may exceed expected duration thresholds

### Problem Type

- **Type**: Regression
- **Target Variable**: `duration_seconds` (continuous, right-skewed distribution)
- **Evaluation Metrics**: RMSE, MAE, R²

## Cloud Deployment

The service is deployed to AWS EKS and available at: **https://duration.pushmetrics.io**

### Quick Test

```bash
# Health check
curl https://duration.pushmetrics.io/health

# Get model info
curl https://duration.pushmetrics.io/info

# Make a prediction
curl -X POST https://duration.pushmetrics.io/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_blocks": 10,
    "sql_blocks": 3,
    "historical_avg_duration": 25.0
  }'
```

## Dataset

### Overview

| Metric | Value |
|--------|-------|
| Total Samples | 50,000 |
| Features | 25 |
| Time Range | January 2024 - January 2026 |
| Source | PushMetrics Production PostgreSQL |
| Unique Workspaces | 47 |
| Unique Pages | 750 |

### Target Variable Statistics

| Statistic | Value |
|-----------|-------|
| Mean | 19.96 seconds |
| Median | 8.61 seconds |
| Std Dev | 71.72 seconds |
| Min | 0.15 seconds |
| Max | 3,366.36 seconds |
| 25th Percentile | 3.74 seconds |
| 75th Percentile | 16.99 seconds |
| 95th Percentile | 52.31 seconds |

### Run Status Distribution

| Status | Count | Percentage |
|--------|-------|------------|
| Success | 44,066 | 88.1% |
| Failed | 5,928 | 11.9% |
| Cancelled | 6 | 0.01% |

### Feature Categories

**Timing Features (5)**
- `start_hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Sunday, 6=Saturday)
- `month`: Month (1-12)
- `is_weekend`: Weekend flag (0/1)
- `is_business_hours`: Business hours flag (9am-5pm) (0/1)

**Block Composition Features (15)**
- `total_blocks`: Total number of blocks in workflow
- `sql_blocks`: SQL query blocks
- `tableau_blocks`: Tableau visualization blocks
- `email_blocks`: Email delivery blocks
- `slack_blocks`: Slack notification blocks
- `parameter_blocks`: Parameter/variable blocks
- `code_blocks`: Python code blocks
- `plotly_blocks`: Plotly chart blocks
- `kpi_blocks`: KPI card blocks
- `api_blocks`: API request blocks
- `writeback_blocks`: Database writeback blocks
- `conditional_blocks`: If/else blocks
- `loop_blocks`: For loop blocks
- `ai_blocks`: OpenAI/LLM blocks
- `storage_blocks`: Cloud storage blocks (S3, GCS, SFTP)

**Historical Performance Features (5)**
- `historical_run_count`: Total previous runs of this workflow
- `historical_avg_duration`: Average duration of previous runs
- `historical_median_duration`: Median duration of previous runs
- `historical_stddev_duration`: Standard deviation of previous run durations
- `historical_failure_rate`: Historical failure rate (0-1)

### Data Anonymization

All data has been anonymized:
- `page_hash`: MD5 hash of notebook_page_id
- `workspace_hash`: MD5 hash of workspace_id
- `run_id`: Internal ID (non-sensitive)

## Project Structure

```
capstone3-duration-prediction/
├── README.md              # This file
├── data.csv               # Dataset (50,000 samples)
├── notebook.ipynb         # Jupyter notebook with EDA and modeling
├── train.py               # Model training script
├── predict.py             # Flask prediction service
├── test_predict.py        # API test suite
├── model.xgb              # Trained XGBoost model
├── dv.pkl                 # DictVectorizer artifact
├── model_info.pkl         # Model metadata
├── requirements.txt       # Python dependencies
├── Pipfile                # Pipenv dependencies
├── Dockerfile             # Docker image definition
├── .dockerignore          # Docker build exclusions
├── deploy.sh              # Kubernetes deployment script
└── k8s/
    ├── deployment.yaml    # Kubernetes deployment
    ├── service.yaml       # Kubernetes service
    └── ingress.yaml       # Kubernetes ingress (AWS ALB)
```

## Model Performance

### Model Comparison

| Model | Train RMSE | Val RMSE | Test RMSE | Train R² | Val R² | Test R² |
|-------|------------|----------|-----------|----------|--------|---------|
| Linear Regression | 56.34 | 66.61 | 64.48 | 0.3182 | 0.3135 | 0.2134 |
| Ridge Regression | 56.34 | 66.61 | 64.48 | 0.3182 | 0.3135 | 0.2134 |
| Random Forest | 41.81 | 63.10 | 62.42 | 0.6246 | 0.3841 | 0.2626 |
| **XGBoost** | **39.12** | **62.67** | **61.48** | **0.6713** | **0.3925** | **0.2849** |

### Final Model: XGBoost Regressor

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| RMSE | 39.12 | 62.67 | 61.48 |
| MAE | 7.14 | 9.80 | 9.67 |
| R² | 0.6713 | 0.3925 | 0.2849 |

**Hyperparameters:**
- `eta`: 0.1
- `max_depth`: 8
- `min_child_weight`: 5
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `num_boost_round`: 33 (early stopping)

### Performance Notes

The R² value of 0.28 on the test set reflects the inherent variability in workflow execution times:
- Workflows can timeout, fail at different stages, or have varying database response times
- External factors (database load, network latency) introduce unpredictable variance
- The model still provides useful predictions for resource planning and scheduling
- MAE of 9.67 seconds means predictions are off by ~10 seconds on average

## Installation & Setup

### Option 1: Using Pipenv (Recommended)

```bash
# Install pipenv if not already installed
pip install pipenv

# Install dependencies
pipenv install

# Activate environment
pipenv shell

# Run training
python train.py

# Start prediction service
python predict.py
```

### Option 2: Using pip with venv

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Start prediction service
python predict.py
```

### Option 3: Using Docker

```bash
# Build Docker image
docker build -t duration-prediction .

# Run container
docker run -p 9696:9696 duration-prediction

# Test health endpoint
curl http://localhost:9696/health
```

## Usage

### 1. Jupyter Notebook (EDA & Modeling)

```bash
jupyter notebook notebook.ipynb
```

The notebook includes:
- Data loading and exploration
- Target variable analysis
- Feature correlation analysis
- Model training (Linear, Ridge, Random Forest, XGBoost)
- Hyperparameter tuning
- Model comparison and selection
- Feature importance analysis

### 2. Training

```bash
python train.py
```

Output:
- `model.xgb`: Trained XGBoost model
- `dv.pkl`: DictVectorizer for feature transformation
- `model_info.pkl`: Model metadata and metrics

### 3. Prediction Service

```bash
python predict.py
```

Or with gunicorn:
```bash
gunicorn --bind 0.0.0.0:9696 --workers 2 predict:app
```

### 4. Testing

```bash
# Test local server
python test_predict.py

# Test remote server
python test_predict.py --url https://duration.pushmetrics.io
```

### 5. Docker Deployment

```bash
# Build image
docker build -t duration-prediction .

# Run locally
docker run -p 9696:9696 duration-prediction

# Test
curl http://localhost:9696/health
```

## API Endpoints

### GET /
Service information.

```bash
curl http://localhost:9696/
```

Response:
```json
{
  "description": "Predicts how long a workflow execution will take based on workflow structure and history",
  "endpoints": {
    "GET /": "Service information",
    "GET /health": "Health check",
    "GET /info": "Model information",
    "POST /predict": "Predict duration"
  },
  "service": "Workflow Execution Duration Predictor",
  "version": "1.0.0"
}
```

### GET /health
Health check endpoint.

```bash
curl http://localhost:9696/health
```

Response:
```json
{
  "model_loaded": true,
  "service": "duration-prediction",
  "status": "healthy"
}
```

### GET /info
Model information.

```bash
curl http://localhost:9696/info
```

Response:
```json
{
  "feature_columns": ["start_hour", "day_of_week", "month", "is_weekend", "is_business_hours", "total_blocks", "sql_blocks", "tableau_blocks", "email_blocks", "slack_blocks", "parameter_blocks", "code_blocks", "plotly_blocks", "kpi_blocks", "api_blocks", "writeback_blocks", "conditional_blocks", "loop_blocks", "ai_blocks", "storage_blocks", "historical_run_count", "historical_avg_duration", "historical_median_duration", "historical_stddev_duration", "historical_failure_rate"],
  "metrics": {
    "test_r2": 0.2849,
    "test_rmse": 61.4759,
    "train_r2": 0.6713,
    "train_rmse": 39.1234,
    "val_r2": 0.3925,
    "val_rmse": 62.6656
  },
  "model_type": "XGBoost Regressor",
  "n_features": 25,
  "num_boost_round": 33,
  "params": {
    "colsample_bytree": 0.8,
    "eta": 0.1,
    "eval_metric": "rmse",
    "max_depth": 8,
    "min_child_weight": 5,
    "objective": "reg:squarederror",
    "seed": 42,
    "subsample": 0.8,
    "verbosity": 1
  }
}
```

### POST /predict
Predict workflow execution duration.

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_blocks": 10,
    "sql_blocks": 3,
    "tableau_blocks": 2,
    "email_blocks": 1,
    "slack_blocks": 1,
    "historical_run_count": 100,
    "historical_avg_duration": 25.0,
    "historical_median_duration": 20.0,
    "historical_stddev_duration": 10.0,
    "historical_failure_rate": 0.03
  }'
```

Response:
```json
{
  "duration_category": "normal",
  "duration_description": "Normal execution (5-30s)",
  "predicted_duration": 22.8,
  "recommendation": "Standard workflow - monitor for anomalies"
}
```

### Duration Categories

| Category | Duration Range | Recommendation |
|----------|---------------|----------------|
| quick | < 5s | No special handling needed |
| normal | 5-30s | Standard workflow - monitor for anomalies |
| moderate | 30s-2min | Consider scheduling during off-peak hours |
| long | 2-5min | Recommend scheduling and resource optimization |
| very_long | > 5min | Review workflow complexity, consider breaking into smaller workflows |

## Key Insights from EDA

1. **Historical Performance is Key**: `historical_avg_duration` and `historical_median_duration` are the strongest predictors of actual duration

2. **Block Composition Matters**:
   - Writeback blocks are the slowest (avg 40+ seconds)
   - SQL and Tableau blocks have significant duration impact
   - Email/Slack blocks add minimal overhead

3. **Timing Effects**:
   - Morning hours (7-9 AM) have slightly higher failure rates (11.5%)
   - Weekend executions show similar patterns to weekdays
   - Business hours have marginally faster execution

4. **Workflow Complexity**:
   - More blocks correlate with longer duration
   - Loops and conditionals add unpredictability
   - AI/LLM blocks have variable execution times

5. **Duration Distribution**:
   - Highly right-skewed (median 8.6s vs mean 20s)
   - 95% of executions complete within 52 seconds
   - Long-tail caused by complex workflows and timeouts

## Kubernetes Deployment

### Prerequisites

- AWS CLI configured
- kubectl configured for your cluster
- Docker installed
- ECR repository created

### Deploy to AWS EKS

```bash
# Set environment variables
export ECR_REGISTRY=123456789012.dkr.ecr.eu-central-1.amazonaws.com
export IMAGE_TAG=latest
export CERTIFICATE_ARN=arn:aws:acm:eu-central-1:123456789012:certificate/xxx

# Run deployment
./deploy.sh
```

### Kubernetes Architecture

```
Internet
    ↓
AWS ALB (HTTPS:443)
    ↓
Ingress (duration.pushmetrics.io)
    ↓
Service (ClusterIP, port 80)
    ↓
Pod (Flask + XGBoost, port 9696)
```

### Verify Deployment

```bash
# Check pods
kubectl get pods -l app=duration-prediction

# Check service
kubectl get svc duration-prediction

# Check ingress
kubectl get ingress duration-prediction
```

### Live API Examples

**Health Check:**
```bash
curl https://duration.pushmetrics.io/health
```
Response:
```json
{"model_loaded":true,"service":"duration-prediction","status":"healthy"}
```

**Model Info:**
```bash
curl https://duration.pushmetrics.io/info
```
Response:
```json
{
  "feature_columns": ["start_hour", "day_of_week", "month", ...],
  "metrics": {"test_r2": 0.2849, "test_rmse": 61.4759, "train_r2": 0.6713, "train_rmse": 39.1234, "val_r2": 0.3925, "val_rmse": 62.6656},
  "model_type": "XGBoost Regressor",
  "n_features": 25,
  "num_boost_round": 33
}
```

**Prediction:**
```bash
curl -X POST https://duration.pushmetrics.io/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_blocks": 10,
    "sql_blocks": 3,
    "tableau_blocks": 2,
    "email_blocks": 1,
    "slack_blocks": 1,
    "historical_run_count": 100,
    "historical_avg_duration": 25.0,
    "historical_median_duration": 20.0,
    "historical_stddev_duration": 10.0,
    "historical_failure_rate": 0.03
  }'
```
Response:
```json
{
  "duration_category": "normal",
  "duration_description": "Normal execution (5-30s)",
  "predicted_duration": 22.8,
  "recommendation": "Standard workflow - monitor for anomalies"
}
```

## Dependencies

### Production
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost == 2.1.4
- flask >= 3.0.0
- gunicorn >= 21.0.0
- requests >= 2.31.0

### Development
- jupyter
- matplotlib
- seaborn
- tqdm

## Future Improvements

1. **Feature Engineering**
   - Add SQL query complexity features (joins, subqueries)
   - Include database type and connection latency
   - Add user/workspace activity patterns

2. **Model Enhancements**
   - Try LightGBM for faster training
   - Ensemble multiple models
   - Quantile regression for prediction intervals

3. **Monitoring**
   - Track prediction accuracy over time
   - Alert on model drift
   - A/B test different model versions

4. **Infrastructure**
   - Add horizontal pod autoscaling
   - Implement model versioning
   - Set up continuous training pipeline

## License

This project is part of the ML Zoomcamp course by DataTalks.Club.

## Acknowledgments

- [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club
- [PushMetrics](https://pushmetrics.io) for providing the production data
