#!/usr/bin/env python3
"""
User Churn Prediction - Flask Prediction Service
ML Zoomcamp Capstone 2 Project

This Flask application serves predictions for user churn probability.
It loads a trained XGBoost model and provides REST API endpoints.

Endpoints:
    GET  /           - Service info
    GET  /health     - Health check
    GET  /info       - Model information
    POST /predict    - Predict churn probability for a user

Usage:
    python predict.py                    # Development server
    gunicorn --bind 0.0.0.0:9696 predict:app  # Production server
"""

import pickle
import os
import warnings
from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

# Suppress XGBoost file format warning for .xgb extension
warnings.filterwarnings('ignore', message='.*Unknown file format.*')

# Configuration
MODEL_FILE = os.getenv('MODEL_FILE', 'model.xgb')
DV_FILE = os.getenv('DV_FILE', 'dv.pkl')
MODEL_INFO_FILE = os.getenv('MODEL_INFO_FILE', 'model_info.pkl')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 9696))

# Initialize Flask app
app = Flask(__name__)

# Load model artifacts
print(f'Loading model from {MODEL_FILE}...')
model = xgb.Booster()
model.load_model(MODEL_FILE)

print(f'Loading DictVectorizer from {DV_FILE}...')
with open(DV_FILE, 'rb') as f:
    dv = pickle.load(f)

print(f'Loading model info from {MODEL_INFO_FILE}...')
with open(MODEL_INFO_FILE, 'rb') as f:
    model_info = pickle.load(f)

print('Model loaded successfully!')


def get_risk_level(probability: float) -> tuple:
    """
    Determine churn risk level and recommendation based on probability.

    Returns:
        tuple: (risk_level, recommendation)
    """
    if probability >= 0.9:
        return 'critical', 'Immediate intervention required - user likely to churn'
    elif probability >= 0.7:
        return 'high', 'High risk - prioritize for customer success outreach'
    elif probability >= 0.5:
        return 'medium', 'Moderate risk - schedule check-in within 7 days'
    else:
        return 'low', 'Low risk - continue standard engagement'


@app.route('/', methods=['GET'])
def index():
    """Service info endpoint."""
    return jsonify({
        'service': 'User Churn Prediction API',
        'version': '1.0.0',
        'description': 'Predicts user churn probability for PushMetrics platform',
        'endpoints': {
            '/': 'Service info (this page)',
            '/health': 'Health check',
            '/info': 'Model information',
            '/predict': 'POST - Predict churn probability'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'user-churn-predictor',
        'model_loaded': model is not None
    })


@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint."""
    return jsonify({
        'model_type': model_info.get('model_type', 'XGBoost'),
        'params': model_info.get('params', {}),
        'num_boost_rounds': model_info.get('num_boost_rounds', 200),
        'train_auc': round(model_info.get('train_auc', 0), 4),
        'val_auc': round(model_info.get('val_auc', 0), 4),
        'test_auc': round(model_info.get('test_auc', 0), 4),
        'n_features': model_info.get('n_features', 0),
        'feature_columns': model_info.get('feature_columns', [])
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn probability for a user.

    Expected JSON payload with user features:
    {
        "login_count": 10,
        "days_since_last_login": 5.0,
        "login_frequency": 0.15,
        "workspace_count": 2,
        ...
    }

    Returns:
    {
        "churn_probability": 0.75,
        "will_churn": true,
        "risk_level": "high",
        "recommendation": "High risk - prioritize for customer success outreach"
    }
    """
    try:
        # Get user data from request
        user_data = request.get_json()

        if not user_data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide user features as JSON'
            }), 400

        # Define default values for missing features
        default_values = {
            'is_generic_email': 0,
            'account_active': 1,
            'login_count': 0,
            'fail_login_count': 0,
            'days_since_signup': 0,
            'days_since_last_login': 0,
            'signup_hour': 12,
            'signup_day_of_week': 1,
            'signup_month': 1,
            'last_login_hour': 12,
            'last_login_day_of_week': 1,
            'has_first_name': 0,
            'has_last_name': 0,
            'profile_completeness': 0,
            'workspace_count': 1,
            'organization_count': 1,
            'role_count': 1,
            'is_admin': 0,
            'is_guest': 0,
            'has_avatar': 0,
            'mfa_enabled': 0,
            'login_frequency': 0
        }

        # Fill in missing features with defaults
        for feature, default_val in default_values.items():
            if feature not in user_data:
                user_data[feature] = default_val

        # Transform features using DictVectorizer
        X = dv.transform([user_data])

        # Get feature names from DictVectorizer
        feature_names = list(dv.get_feature_names_out())

        # Create DMatrix with feature names and predict
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        probability = float(model.predict(dmatrix)[0])

        # Determine risk level and recommendation
        risk_level, recommendation = get_risk_level(probability)

        # Prepare response
        response = {
            'churn_probability': round(probability, 4),
            'will_churn': probability >= 0.5,
            'risk_level': risk_level,
            'recommendation': recommendation
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


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


if __name__ == '__main__':
    print(f'Starting Flask server on {HOST}:{PORT}...')
    app.run(host=HOST, port=PORT, debug=False)
