#!/usr/bin/env python3
"""
Workflow Execution Duration Prediction - Flask Prediction Service

This service provides REST API endpoints for predicting workflow execution duration.

Endpoints:
    GET  /           - Service information
    GET  /health     - Health check
    GET  /info       - Model information
    POST /predict    - Predict duration for a workflow

Usage:
    python predict.py

    Or with gunicorn:
    gunicorn --bind 0.0.0.0:9696 --workers 2 predict:app
"""

import os
import pickle
import warnings

import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, request

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

MODEL_FILE = os.environ.get('MODEL_FILE', 'model.xgb')
DV_FILE = os.environ.get('DV_FILE', 'dv.pkl')
MODEL_INFO_FILE = os.environ.get('MODEL_INFO_FILE', 'model_info.pkl')
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 9696))

# =============================================================================
# Load Model and Artifacts
# =============================================================================

app = Flask('duration-prediction')

# Load XGBoost model
print(f'Loading model from {MODEL_FILE}...')
model = xgb.Booster()
model.load_model(MODEL_FILE)
print('Model loaded successfully!')

# Load DictVectorizer
print(f'Loading DictVectorizer from {DV_FILE}...')
with open(DV_FILE, 'rb') as f:
    dv = pickle.load(f)
print('DictVectorizer loaded successfully!')

# Load model info
print(f'Loading model info from {MODEL_INFO_FILE}...')
with open(MODEL_INFO_FILE, 'rb') as f:
    model_info = pickle.load(f)
print('Model info loaded successfully!')

# Get feature names
feature_names = list(dv.get_feature_names_out())

# =============================================================================
# Helper Functions
# =============================================================================

def get_duration_category(duration_seconds: float) -> tuple:
    """
    Categorize predicted duration and provide recommendations.

    Returns:
        tuple: (category, description, recommendation)
    """
    if duration_seconds <= 5:
        return ('quick', 'Quick execution (< 5s)',
                'No special handling needed')
    elif duration_seconds <= 30:
        return ('normal', 'Normal execution (5-30s)',
                'Standard workflow - monitor for anomalies')
    elif duration_seconds <= 120:
        return ('moderate', 'Moderate execution (30s-2min)',
                'Consider scheduling during off-peak hours')
    elif duration_seconds <= 300:
        return ('long', 'Long execution (2-5min)',
                'Recommend scheduling and resource optimization')
    else:
        return ('very_long', 'Very long execution (> 5min)',
                'Urgent: Review workflow complexity, consider breaking into smaller workflows')


# Default feature values for missing inputs
DEFAULT_FEATURES = {
    'start_hour': 12,
    'day_of_week': 2,
    'month': 6,
    'is_weekend': 0,
    'is_business_hours': 1,
    'total_blocks': 5,
    'sql_blocks': 1,
    'tableau_blocks': 0,
    'email_blocks': 1,
    'slack_blocks': 0,
    'parameter_blocks': 0,
    'code_blocks': 0,
    'plotly_blocks': 0,
    'kpi_blocks': 0,
    'api_blocks': 0,
    'writeback_blocks': 0,
    'conditional_blocks': 0,
    'loop_blocks': 0,
    'ai_blocks': 0,
    'storage_blocks': 0,
    'historical_run_count': 10,
    'historical_avg_duration': 10.0,
    'historical_median_duration': 8.0,
    'historical_stddev_duration': 5.0,
    'historical_failure_rate': 0.05
}

# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/', methods=['GET'])
def index():
    """Service information endpoint."""
    return jsonify({
        'service': 'Workflow Execution Duration Predictor',
        'version': '1.0.0',
        'description': 'Predicts how long a workflow execution will take based on workflow structure and history',
        'endpoints': {
            'GET /': 'Service information',
            'GET /health': 'Health check',
            'GET /info': 'Model information',
            'POST /predict': 'Predict duration'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'duration-prediction',
        'model_loaded': model is not None
    })


@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint."""
    return jsonify({
        'model_type': model_info.get('model_type', 'XGBoost Regressor'),
        'params': model_info.get('params', {}),
        'num_boost_round': model_info.get('num_boost_round', 0),
        'metrics': {
            'train_rmse': round(model_info.get('metrics', {}).get('train_rmse', 0), 4),
            'val_rmse': round(model_info.get('metrics', {}).get('val_rmse', 0), 4),
            'test_rmse': round(model_info.get('metrics', {}).get('test_rmse', 0), 4),
            'train_r2': round(model_info.get('metrics', {}).get('train_r2', 0), 4),
            'val_r2': round(model_info.get('metrics', {}).get('val_r2', 0), 4),
            'test_r2': round(model_info.get('metrics', {}).get('test_r2', 0), 4)
        },
        'n_features': model_info.get('n_features', 0),
        'feature_columns': model_info.get('feature_columns', [])
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict workflow execution duration.

    Expected JSON input:
    {
        "start_hour": 14,
        "day_of_week": 2,
        "month": 1,
        "is_weekend": 0,
        "is_business_hours": 1,
        "total_blocks": 10,
        "sql_blocks": 3,
        "tableau_blocks": 2,
        "email_blocks": 1,
        "slack_blocks": 1,
        "parameter_blocks": 1,
        "code_blocks": 0,
        "plotly_blocks": 0,
        "kpi_blocks": 0,
        "api_blocks": 0,
        "writeback_blocks": 0,
        "conditional_blocks": 0,
        "loop_blocks": 0,
        "ai_blocks": 0,
        "storage_blocks": 0,
        "historical_run_count": 100,
        "historical_avg_duration": 15.5,
        "historical_median_duration": 12.0,
        "historical_stddev_duration": 8.0,
        "historical_failure_rate": 0.02
    }

    Returns:
    {
        "predicted_duration": 15.45,
        "duration_category": "normal",
        "duration_description": "Normal execution (5-30s)",
        "recommendation": "Standard workflow - monitor for anomalies"
    }
    """
    try:
        # Get input data
        workflow = request.get_json()

        if workflow is None:
            workflow = {}

        # Fill missing values with defaults
        for key, default_value in DEFAULT_FEATURES.items():
            if key not in workflow:
                workflow[key] = default_value

        # Transform with DictVectorizer
        X = dv.transform([workflow])

        # Create DMatrix with feature names
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)

        # Predict
        predicted_duration = float(model.predict(dmatrix)[0])

        # Ensure non-negative prediction
        predicted_duration = max(0.0, predicted_duration)

        # Get duration category
        category, description, recommendation = get_duration_category(predicted_duration)

        return jsonify({
            'predicted_duration': round(predicted_duration, 2),
            'duration_category': category,
            'duration_description': description,
            'recommendation': recommendation
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to process prediction request'
        }), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print(f'\n{"="*60}')
    print('WORKFLOW EXECUTION DURATION PREDICTOR')
    print(f'{"="*60}')
    print(f'\nStarting Flask server on {HOST}:{PORT}...')
    print(f'\nTest with:')
    print(f'  curl http://localhost:{PORT}/health')
    print(f'  curl -X POST http://localhost:{PORT}/predict \\')
    print(f'       -H "Content-Type: application/json" \\')
    print(f'       -d \'{{"total_blocks": 10, "sql_blocks": 3, "historical_avg_duration": 15}}\'')
    print(f'\n{"="*60}\n')

    app.run(host=HOST, port=PORT, debug=False)
