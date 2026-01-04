#!/usr/bin/env python
"""
Test script for the prediction service

Usage:
    # Start the service first: python predict.py
    # Then run this test: python test_predict.py
"""

import requests
import json

# Service URL
URL = 'http://localhost:9696/predict'

# Test cases
test_cases = [
    {
        'name': 'High conversion probability (activated quickly, good usage)',
        'customer': {
            'users_invited': 35,
            'is_generic_email': 0,
            'trial_start_hour': 10,
            'trial_start_day_of_week': 1,
            'trial_start_is_business_hours': 1,
            'trial_duration_days': 2,
            'accepted_terms': 0,
            'hours_to_first_execution': 1.0,
            'hours_to_first_report': 0.0,
            'reports_created': 20,
        }
    },
    {
        'name': 'Medium conversion probability (moderate activation, some usage)',
        'customer': {
            'users_invited': 10,
            'is_generic_email': 0,
            'trial_start_hour': 14,
            'trial_start_day_of_week': 3,
            'trial_start_is_business_hours': 1,
            'trial_duration_days': 10,
            'accepted_terms': 1,
            'hours_to_first_execution': 24.0,
            'hours_to_first_report': 30.0,
            'reports_created': 3,
            'reports_week1': 1,
            'executions_total': 10,
            'executions_week1': 4,
            'execution_success_rate': 0.75,
            'execution_growth_rate': 0.05,
            'activated_day1': 1,
            'activated_day2': 1,
            'activated_week1': 0,
            'has_sql_block': 1,
            'has_bigquery': 0,
            'schedules_created': 1,
            'schedules_active': 0,
            'sql_connections_count': 1,
            'blocks_created': 5
        }
    },
    {
        'name': 'Low conversion probability (never activated, no usage)',
        'customer': {
            'users_invited': 0,
            'is_generic_email': 1,
            'trial_start_hour': 2,
            'trial_start_day_of_week': 4,
            'trial_start_is_business_hours': 0,
            'trial_duration_days': 14,
            'accepted_terms': 0,
            'hours_to_first_execution': 999.0,
            'hours_to_first_report': 999.0,
            'reports_created': 0,
            'reports_week1': 0,
            'executions_total': 0,
            'executions_week1': 0,
            'execution_success_rate': 0.0,
            'execution_growth_rate': 0.0,
            'activated_day1': 0,
            'activated_day2': 0,
            'activated_week1': 0,
            'has_sql_block': 0,
            'has_bigquery': 0,
            'schedules_created': 0,
            'schedules_active': 0,
            'sql_connections_count': 0,
            'blocks_created': 0
        }
    }
]

def test_prediction_service():
    """Test the prediction service with multiple test cases"""
    print("="*80)
    print("TESTING TRIAL CONVERSION PREDICTION SERVICE")
    print("="*80)
    print()

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 80)

        # Send request
        response = requests.post(URL, json=test_case['customer'])

        if response.status_code == 200:
            result = response.json()

            print(f"✓ Request successful")
            print(f"\nInput features:")
            for key, value in test_case['customer'].items():
                print(f"  {key:30s}: {value}")

            print(f"\nPrediction:")
            print(f"  Conversion probability: {result['conversion_probability']:.2%}")
            print(f"  Will convert:          {result['will_convert']}")
            print(f"  Risk level:            {result['risk_level']}")
            print(f"  Recommendation:        {result['recommendation']}")

        else:
            print(f"✗ Request failed with status code: {response.status_code}")
            print(f"  Response: {response.text}")

        print()
        print()


def test_health_endpoint():
    """Test the health check endpoint"""
    print("="*80)
    print("TESTING HEALTH ENDPOINT")
    print("="*80)

    health_url = 'http://localhost:9696/health'
    response = requests.get(health_url)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Health check successful")
        print(f"  Status: {result['status']}")
        print(f"  Model:  {result['model']}")
        print(f"  Service: {result['service']}")
    else:
        print(f"✗ Health check failed")

    print()


if __name__ == '__main__':
    try:
        # Test health endpoint
        test_health_endpoint()

        # Test prediction service
        test_prediction_service()

        print("="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to prediction service")
        print("Make sure the service is running: python predict.py")
    except Exception as e:
        print(f"Error: {e}")
