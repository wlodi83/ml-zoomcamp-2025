#!/usr/bin/env python
"""
Flask Prediction Service for Trial-to-Paid Conversion

Usage:
    python predict.py

Test:
    curl -X POST http://localhost:9696/predict \
      -H "Content-Type: application/json" \
      -d '{"hours_to_first_execution": 2.5, "reports_created": 3, "executions_total": 15, "execution_success_rate": 0.93, "activated_day1": 1}'
"""

import pickle
from flask import Flask, request, jsonify

# Configuration
MODEL_FILE = 'model.pkl'
VECTORIZER_FILE = 'dv.pkl'
PORT = 9696

# Load model and vectorizer
print("Loading model and vectorizer...")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print(f"✓ Loaded model from {MODEL_FILE}")

with open(VECTORIZER_FILE, 'rb') as f:
    dv = pickle.load(f)
print(f"✓ Loaded vectorizer from {VECTORIZER_FILE}")

# Create Flask app
app = Flask('trial-conversion')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict trial conversion probability

    Input (JSON):
        {
            "hours_to_first_execution": 2.5,
            "reports_created": 3,
            "executions_total": 15,
            "execution_success_rate": 0.93,
            "activated_day1": 1,
            "has_sql_block": 1,
            "has_bigquery": 1,
            "schedules_created": 1,
            ...  # other features
        }

    Output (JSON):
        {
            "conversion_probability": 0.87,
            "will_convert": true,
            "risk_level": "low",
            "recommendation": "Low churn risk - standard onboarding"
        }
    """
    # Get customer data
    customer = request.get_json()

    # Vectorize features
    X = dv.transform([customer])

    # Predict
    probability = float(model.predict_proba(X)[0, 1])

    # Determine risk level and recommendation
    if probability >= 0.5:
        risk_level = "low"
        recommendation = "High conversion likelihood - standard onboarding"
    elif probability >= 0.32:
        risk_level = "medium"
        recommendation = "Moderate risk - consider sales outreach or onboarding assistance"
    else:
        risk_level = "high"
        recommendation = "High churn risk - urgent intervention needed (sales call, dedicated support)"

    # Prepare response
    result = {
        'conversion_probability': round(probability, 4),
        'will_convert': probability >= 0.5,
        'risk_level': risk_level,
        'recommendation': recommendation
    }

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_FILE,
        'service': 'trial-conversion-predictor'
    })


@app.route('/', methods=['GET'])
def index():
    """Service info"""
    return jsonify({
        'service': 'Trial-to-Paid Conversion Predictor',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Predict conversion probability',
            '/health': 'GET - Health check',
            '/': 'GET - Service info'
        },
        'example_request': {
            'url': 'http://localhost:9696/predict',
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'body': {
                'hours_to_first_execution': 2.5,
                'reports_created': 3,
                'executions_total': 15,
                'execution_success_rate': 0.93,
                'activated_day1': 1,
                'has_sql_block': 1,
                'has_bigquery': 1,
                'schedules_created': 1
            }
        }
    })


if __name__ == '__main__':
    print("="*80)
    print("TRIAL CONVERSION PREDICTION SERVICE")
    print("="*80)
    print(f"Starting Flask app on http://0.0.0.0:{PORT}")
    print(f"\nTest endpoint:")
    print(f"  curl http://localhost:{PORT}/health")
    print(f"\nPredict endpoint:")
    print(f"  curl -X POST http://localhost:{PORT}/predict \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"hours_to_first_execution\": 2.5, \"reports_created\": 3, \"activated_day1\": 1}}'")
    print("="*80)
    print()

    app.run(debug=True, host='0.0.0.0', port=PORT)
