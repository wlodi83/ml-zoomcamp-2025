#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Report Failure Prediction API

Flask web service for predicting report execution failures.

Usage:
    python predict.py

Then send POST requests to http://localhost:9696/predict with JSON data.

Example request:
    {
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
    }
"""

import pickle
import xgboost as xgb
from flask import Flask, request, jsonify

# Configuration
MODEL_FILE = 'model.xgb'
DV_FILE = 'dv.pkl'
MODEL_INFO_FILE = 'model_info.pkl'
HOST = '0.0.0.0'
PORT = 9696

# Load model and DictVectorizer
print("Loading model files...")

# Load DictVectorizer
with open(DV_FILE, 'rb') as f:
    dv = pickle.load(f)
print(f"Loaded DictVectorizer from '{DV_FILE}'")

# Load XGBoost model
model = xgb.Booster()
model.load_model(MODEL_FILE)
print(f"Loaded XGBoost model from '{MODEL_FILE}'")

# Load model metadata
with open(MODEL_INFO_FILE, 'rb') as f:
    model_info = pickle.load(f)
print(f"Loaded model metadata from '{MODEL_INFO_FILE}'")

print(f"\nModel Info:")
print(f"  Type: {model_info['model_type']}")
print(f"  Num rounds: {model_info['num_boost_rounds']}")
print(f"  Test AUC: {model_info['test_auc']:.4f}")
print(f"  Features: {model_info['n_features']}")

# Initialize Flask app
app = Flask('report-failure-prediction')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict report failure probability.

    Expects JSON with report features.
    Returns JSON with failure probability and prediction.
    """
    # Get JSON data
    report_data = request.get_json()

    # Validate input
    required_features = model_info['feature_cols']
    missing_features = [f for f in required_features if f not in report_data]

    # check if the coming request has all the required features
    if missing_features:
        return jsonify({
            'error': f'Missing features: {missing_features}',
            'required_features': required_features
        }), 400

    # Prepare features
    # Fill missing values with 0 (same as training)
    report_dict = {}
    for feature in required_features:
        value = report_data.get(feature, 0)
        # Handle None values
        if value is None:
            value = 0
        report_dict[feature] = value

    # Transform with DictVectorizer
    X = dv.transform([report_dict])

    # Create DMatrix for prediction
    dmatrix = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())

    # Predict
    failure_probability = float(model.predict(dmatrix)[0])
    failure_prediction = failure_probability >= 0.5

    # Prepare response
    result = {
        'failure_probability': round(failure_probability, 4),
        'failure': bool(failure_prediction),
        'risk_level': get_risk_level(failure_probability)
    }

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': model_info['model_type'],
        'test_auc': model_info['test_auc']
    })


@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint."""
    return jsonify({
        'model_type': model_info['model_type'],
        'params': model_info['params'],
        'num_boost_rounds': model_info['num_boost_rounds'],
        'train_auc': model_info['train_auc'],
        'val_auc': model_info['val_auc'],
        'test_auc': model_info['test_auc'],
        'n_features': model_info['n_features'],
        'feature_cols': model_info['feature_cols']
    })


def get_risk_level(probability):
    """
    Classify risk level based on failure probability.

    Returns:
        str: 'low', 'medium', 'high', or 'critical'
    """
    if probability < 0.25:
        return 'low'
    elif probability < 0.5:
        return 'medium'
    elif probability < 0.75:
        return 'high'
    else:
        return 'critical'


if __name__ == '__main__':
    print(f"\nStarting Flask server on {HOST}:{PORT}")
    print(f"Endpoints:")
    print(f"  POST /predict - Predict report failure")
    print(f"  GET  /health  - Health check")
    print(f"  GET  /info    - Model information")
    print(f"\nExample request:")
    print(f"""
    curl -X POST http://localhost:{PORT}/predict \\
      -H "Content-Type: application/json" \\
      -d '{{
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
    }}'
    """)

    app.run(debug=True, host=HOST, port=PORT)
