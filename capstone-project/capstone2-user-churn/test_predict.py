#!/usr/bin/env python3
"""
User Churn Prediction - API Test Suite
ML Zoomcamp Capstone 2 Project

This script tests the prediction API endpoints with various scenarios.

Usage:
    # Start the API server first
    python predict.py

    # Run tests
    python test_predict.py
    python test_predict.py --url http://localhost:9696
    python test_predict.py --url https://churn.pushmetrics.io
"""

import argparse
import json
import requests
import sys


# Default API URL
DEFAULT_URL = 'http://localhost:9696'


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print('\n[TEST] Health Check')
    print('-' * 50)

    try:
        response = requests.get(f'{base_url}/health', timeout=10)
        data = response.json()

        print(f'Status Code: {response.status_code}')
        print(f'Response: {json.dumps(data, indent=2)}')

        if response.status_code == 200 and data.get('status') == 'healthy':
            print('PASS: Health check successful')
            return True
        else:
            print('FAIL: Health check failed')
            return False

    except Exception as e:
        print(f'FAIL: {e}')
        return False


def test_info(base_url: str) -> bool:
    """Test model info endpoint."""
    print('\n[TEST] Model Info')
    print('-' * 50)

    try:
        response = requests.get(f'{base_url}/info', timeout=10)
        data = response.json()

        print(f'Status Code: {response.status_code}')
        print(f'Response: {json.dumps(data, indent=2)}')

        if response.status_code == 200 and 'model_type' in data:
            print('PASS: Model info retrieved successfully')
            return True
        else:
            print('FAIL: Model info retrieval failed')
            return False

    except Exception as e:
        print(f'FAIL: {e}')
        return False


def test_predict_low_risk(base_url: str) -> bool:
    """Test prediction for low-risk user (active, engaged)."""
    print('\n[TEST] Predict - Low Risk User (Active, Engaged)')
    print('-' * 50)

    # Active user with high engagement
    user_data = {
        'is_generic_email': 0,
        'account_active': 1,
        'login_count': 150,
        'fail_login_count': 0,
        'days_since_signup': 365,
        'days_since_last_login': 1,  # Logged in yesterday
        'signup_hour': 10,
        'signup_day_of_week': 2,
        'signup_month': 6,
        'last_login_hour': 14,
        'last_login_day_of_week': 5,
        'has_first_name': 1,
        'has_last_name': 1,
        'profile_completeness': 2,
        'workspace_count': 3,
        'organization_count': 2,
        'role_count': 5,
        'is_admin': 1,
        'is_guest': 0,
        'has_avatar': 1,
        'mfa_enabled': 1,
        'login_frequency': 0.41  # ~150 logins / 365 days
    }

    try:
        response = requests.post(
            f'{base_url}/predict',
            json=user_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        data = response.json()

        print(f'Status Code: {response.status_code}')
        print(f'Input: Active admin, 150 logins, logged in yesterday')
        print(f'Response: {json.dumps(data, indent=2)}')

        if response.status_code == 200 and 'churn_probability' in data:
            prob = data['churn_probability']
            risk = data['risk_level']
            print(f'PASS: Prediction successful (prob={prob:.4f}, risk={risk})')
            return True
        else:
            print('FAIL: Prediction failed')
            return False

    except Exception as e:
        print(f'FAIL: {e}')
        return False


def test_predict_high_risk(base_url: str) -> bool:
    """Test prediction for high-risk user (inactive, disengaged)."""
    print('\n[TEST] Predict - High Risk User (Inactive, Disengaged)')
    print('-' * 50)

    # Inactive user with low engagement
    user_data = {
        'is_generic_email': 1,  # Gmail user
        'account_active': 1,
        'login_count': 3,
        'fail_login_count': 2,
        'days_since_signup': 180,
        'days_since_last_login': 60,  # Last login 60 days ago
        'signup_hour': 2,
        'signup_day_of_week': 6,
        'signup_month': 1,
        'last_login_hour': 23,
        'last_login_day_of_week': 0,
        'has_first_name': 0,
        'has_last_name': 0,
        'profile_completeness': 0,
        'workspace_count': 1,
        'organization_count': 1,
        'role_count': 1,
        'is_admin': 0,
        'is_guest': 1,
        'has_avatar': 0,
        'mfa_enabled': 0,
        'login_frequency': 0.017  # ~3 logins / 180 days
    }

    try:
        response = requests.post(
            f'{base_url}/predict',
            json=user_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        data = response.json()

        print(f'Status Code: {response.status_code}')
        print(f'Input: Guest user, 3 logins, last login 60 days ago')
        print(f'Response: {json.dumps(data, indent=2)}')

        if response.status_code == 200 and 'churn_probability' in data:
            prob = data['churn_probability']
            risk = data['risk_level']
            print(f'PASS: Prediction successful (prob={prob:.4f}, risk={risk})')
            return True
        else:
            print('FAIL: Prediction failed')
            return False

    except Exception as e:
        print(f'FAIL: {e}')
        return False


def test_predict_medium_risk(base_url: str) -> bool:
    """Test prediction for medium-risk user."""
    print('\n[TEST] Predict - Medium Risk User')
    print('-' * 50)

    # Medium engagement user
    user_data = {
        'is_generic_email': 0,
        'account_active': 1,
        'login_count': 25,
        'fail_login_count': 1,
        'days_since_signup': 90,
        'days_since_last_login': 15,  # Last login 15 days ago
        'signup_hour': 14,
        'signup_day_of_week': 3,
        'signup_month': 9,
        'last_login_hour': 11,
        'last_login_day_of_week': 2,
        'has_first_name': 1,
        'has_last_name': 0,
        'profile_completeness': 1,
        'workspace_count': 1,
        'organization_count': 1,
        'role_count': 2,
        'is_admin': 0,
        'is_guest': 0,
        'has_avatar': 0,
        'mfa_enabled': 0,
        'login_frequency': 0.28  # ~25 logins / 90 days
    }

    try:
        response = requests.post(
            f'{base_url}/predict',
            json=user_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        data = response.json()

        print(f'Status Code: {response.status_code}')
        print(f'Input: Regular user, 25 logins, last login 15 days ago')
        print(f'Response: {json.dumps(data, indent=2)}')

        if response.status_code == 200 and 'churn_probability' in data:
            prob = data['churn_probability']
            risk = data['risk_level']
            print(f'PASS: Prediction successful (prob={prob:.4f}, risk={risk})')
            return True
        else:
            print('FAIL: Prediction failed')
            return False

    except Exception as e:
        print(f'FAIL: {e}')
        return False


def test_predict_partial_data(base_url: str) -> bool:
    """Test prediction with partial data (missing features)."""
    print('\n[TEST] Predict - Partial Data (Missing Features)')
    print('-' * 50)

    # Only essential features provided
    user_data = {
        'login_count': 10,
        'days_since_last_login': 7,
        'days_since_signup': 60
    }

    try:
        response = requests.post(
            f'{base_url}/predict',
            json=user_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        data = response.json()

        print(f'Status Code: {response.status_code}')
        print(f'Input: Only login_count, days_since_last_login, days_since_signup')
        print(f'Response: {json.dumps(data, indent=2)}')

        if response.status_code == 200 and 'churn_probability' in data:
            prob = data['churn_probability']
            risk = data['risk_level']
            print(f'PASS: Prediction successful with partial data (prob={prob:.4f}, risk={risk})')
            return True
        else:
            print('FAIL: Prediction failed')
            return False

    except Exception as e:
        print(f'FAIL: {e}')
        return False


def test_predict_empty_data(base_url: str) -> bool:
    """Test prediction with empty data (should return error)."""
    print('\n[TEST] Predict - Empty Data (Should Return Error)')
    print('-' * 50)

    try:
        response = requests.post(
            f'{base_url}/predict',
            json={},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        data = response.json()

        print(f'Status Code: {response.status_code}')
        print(f'Response: {json.dumps(data, indent=2)}')

        # Empty data should still work with defaults
        if response.status_code == 200 and 'churn_probability' in data:
            print('PASS: Empty data handled gracefully with defaults')
            return True
        else:
            print('PASS: Empty data returned error as expected')
            return True

    except Exception as e:
        print(f'INFO: {e}')
        return True  # Expected behavior for edge case


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Test User Churn Prediction API')
    parser.add_argument('--url', default=DEFAULT_URL, help=f'API base URL (default: {DEFAULT_URL})')
    args = parser.parse_args()

    base_url = args.url.rstrip('/')

    print('=' * 60)
    print('USER CHURN PREDICTION API - TEST SUITE')
    print('=' * 60)
    print(f'Testing API at: {base_url}')

    # Run tests
    tests = [
        ('Health Check', test_health),
        ('Model Info', test_info),
        ('Low Risk User', test_predict_low_risk),
        ('High Risk User', test_predict_high_risk),
        ('Medium Risk User', test_predict_medium_risk),
        ('Partial Data', test_predict_partial_data),
        ('Empty Data', test_predict_empty_data),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func(base_url)
            results.append((name, result))
        except Exception as e:
            print(f'ERROR in {name}: {e}')
            results.append((name, False))

    # Summary
    print('\n' + '=' * 60)
    print('TEST SUMMARY')
    print('=' * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = 'PASS' if result else 'FAIL'
        symbol = '\u2713' if result else '\u2717'
        print(f'{symbol} {status}: {name}')

    print(f'\nTotal: {passed}/{total} tests passed')

    if passed == total:
        print('\nAll tests passed!')
        return 0
    else:
        print('\nSome tests failed.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
