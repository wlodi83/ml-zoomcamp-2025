#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Report Failure Prediction API

Usage:
    python test_api.py
"""

import requests
import json

# API configuration
BASE_URL = "http://localhost:9696"

# Test data examples
test_cases = [
    {
        "name": "Low Risk Report (Simple, Good History)",
        "data": {
            "num_blocks": 3,
            "num_sql_blocks": 1,
            "num_writeback_blocks": 0,
            "num_viz_blocks": 1,
            "num_tableau_blocks": 0,
            "num_email_blocks": 1,
            "num_slack_blocks": 0,
            "num_api_blocks": 0,
            "num_sftp_blocks": 0,
            "num_storage_blocks": 0,
            "num_control_blocks": 0,
            "num_parameters": 0,
            "num_databases": 1,
            "historical_failure_count": 1,
            "historical_executions": 500,
            "historical_failure_rate": 0.002,
            "avg_historical_duration": 5.2,
            "hours_since_last_success": 2.0,
            "hour_of_day": 10,
            "day_of_week": 2,
            "is_weekend": 0,
            "is_business_hours": 1,
            "is_rerun": 0,
            "is_scheduled": 1,
            "database_types": "bigquery"
        }
    },
    {
        "name": "High Risk Report (Complex, Bad History)",
        "data": {
            "num_blocks": 25,
            "num_sql_blocks": 8,
            "num_writeback_blocks": 2,
            "num_viz_blocks": 5,
            "num_tableau_blocks": 2,
            "num_email_blocks": 2,
            "num_slack_blocks": 1,
            "num_api_blocks": 1,
            "num_sftp_blocks": 1,
            "num_storage_blocks": 2,
            "num_control_blocks": 4,
            "num_parameters": 3,
            "num_databases": 5,
            "historical_failure_count": 150,
            "historical_executions": 200,
            "historical_failure_rate": 0.75,
            "avg_historical_duration": 120.5,
            "hours_since_last_success": 168.0,
            "hour_of_day": 3,
            "day_of_week": 0,
            "is_weekend": 1,
            "is_business_hours": 0,
            "is_rerun": 1,
            "is_scheduled": 0,
            "database_types": "snowflake,bigquery,clickhouse"
        }
    },
    {
        "name": "Medium Risk Report",
        "data": {
            "num_blocks": 10,
            "num_sql_blocks": 3,
            "num_writeback_blocks": 1,
            "num_viz_blocks": 2,
            "num_tableau_blocks": 0,
            "num_email_blocks": 1,
            "num_slack_blocks": 1,
            "num_api_blocks": 0,
            "num_sftp_blocks": 0,
            "num_storage_blocks": 1,
            "num_control_blocks": 1,
            "num_parameters": 2,
            "num_databases": 2,
            "historical_failure_count": 25,
            "historical_executions": 100,
            "historical_failure_rate": 0.25,
            "avg_historical_duration": 45.2,
            "hours_since_last_success": 24.0,
            "hour_of_day": 14,
            "day_of_week": 3,
            "is_weekend": 0,
            "is_business_hours": 1,
            "is_rerun": 0,
            "is_scheduled": 1,
            "database_types": "postgres"
        }
    }
]


def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()

        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Model: {data['model']}")
        print(f"Test AUC: {data['test_auc']:.4f}")
        print("✓ Health check passed")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_info():
    """Test info endpoint."""
    print("\n" + "="*60)
    print("Testing /info endpoint")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/info")
        response.raise_for_status()

        data = response.json()
        print(f"Model Type: {data['model_type']}")
        print(f"Num Boost Rounds: {data['num_boost_rounds']}")
        print(f"Train AUC: {data['train_auc']:.4f}")
        print(f"Validation AUC: {data['val_auc']:.4f}")
        print(f"Test AUC: {data['test_auc']:.4f}")
        print(f"Number of Features: {data['n_features']}")
        print("✓ Info endpoint passed")
        return True
    except Exception as e:
        print(f"✗ Info endpoint failed: {e}")
        return False


def test_predict(test_case):
    """Test predict endpoint with a test case."""
    print("\n" + "="*60)
    print(f"Testing: {test_case['name']}")
    print("="*60)

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=test_case['data']
        )
        response.raise_for_status()

        result = response.json()

        print(f"Input features:")
        print(f"  Blocks: {test_case['data']['num_blocks']}")
        print(f"  SQL blocks: {test_case['data']['num_sql_blocks']}")
        print(f"  Databases: {test_case['data']['num_databases']}")
        print(f"  Historical failures: {test_case['data']['historical_failure_count']}/{test_case['data']['historical_executions']}")
        print(f"  Historical failure rate: {test_case['data']['historical_failure_rate']*100:.1f}%")
        print(f"  Database types: {test_case['data']['database_types']}")

        print(f"\nPrediction:")
        print(f"  Failure Probability: {result['failure_probability']*100:.2f}%")
        print(f"  Will Fail: {result['failure']}")
        print(f"  Risk Level: {result['risk_level'].upper()}")

        print(f"✓ Prediction successful")
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("REPORT FAILURE PREDICTION API - TEST SUITE")
    print("="*60)
    print(f"Base URL: {BASE_URL}")

    results = []

    # Test health endpoint
    results.append(("Health Check", test_health()))

    # Test info endpoint
    results.append(("Model Info", test_info()))

    # Test predictions
    for test_case in test_cases:
        results.append((test_case['name'], test_predict(test_case)))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    return passed == total


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
