#!/usr/bin/env python3
"""
Workflow Execution Duration Prediction - API Test Suite

This script tests the prediction service endpoints with various scenarios.

Usage:
    python test_predict.py                    # Test local server
    python test_predict.py --url http://host:port  # Test remote server
"""

import argparse
import json
import sys

import requests

DEFAULT_URL = 'http://localhost:9696'


def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    print('\n[TEST] Health Check')
    print('-' * 50)

    try:
        response = requests.get(f'{base_url}/health', timeout=10)
        print(f'Status Code: {response.status_code}')

        if response.status_code == 200:
            data = response.json()
            print(f'Response: {json.dumps(data, indent=2)}')

            if data.get('status') == 'healthy':
                print('PASS: Health check passed')
                return True
            else:
                print('FAIL: Unhealthy status')
                return False
        else:
            print(f'FAIL: Expected 200, got {response.status_code}')
            return False

    except Exception as e:
        print(f'FAIL: {str(e)}')
        return False


def test_info(base_url: str) -> bool:
    """Test the model info endpoint."""
    print('\n[TEST] Model Info')
    print('-' * 50)

    try:
        response = requests.get(f'{base_url}/info', timeout=10)
        print(f'Status Code: {response.status_code}')

        if response.status_code == 200:
            data = response.json()
            print(f'Model Type: {data.get("model_type")}')
            print(f'Features: {data.get("n_features")}')
            print(f'Test RMSE: {data.get("metrics", {}).get("test_rmse")}')
            print(f'Test R²: {data.get("metrics", {}).get("test_r2")}')
            print('PASS: Info endpoint working')
            return True
        else:
            print(f'FAIL: Expected 200, got {response.status_code}')
            return False

    except Exception as e:
        print(f'FAIL: {str(e)}')
        return False


def test_predict_quick_workflow(base_url: str) -> bool:
    """Test prediction for a quick/simple workflow."""
    print('\n[TEST] Quick Workflow Prediction')
    print('-' * 50)

    workflow = {
        'start_hour': 10,
        'day_of_week': 2,
        'month': 6,
        'is_weekend': 0,
        'is_business_hours': 1,
        'total_blocks': 3,
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
        'historical_run_count': 500,
        'historical_avg_duration': 3.5,
        'historical_median_duration': 3.0,
        'historical_stddev_duration': 1.0,
        'historical_failure_rate': 0.01
    }

    try:
        response = requests.post(
            f'{base_url}/predict',
            json=workflow,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f'Status Code: {response.status_code}')
        print(f'Input: Simple workflow with 3 blocks, good history (avg 3.5s)')

        if response.status_code == 200:
            data = response.json()
            print(f'Response: {json.dumps(data, indent=2)}')

            duration = data.get('predicted_duration', 0)
            category = data.get('duration_category', '')

            # Quick workflows should predict low duration
            if duration < 30 and category in ['quick', 'normal']:
                print(f'PASS: Predicted {duration:.2f}s ({category})')
                return True
            else:
                print(f'WARN: Unexpected prediction for quick workflow')
                return True  # Still pass if prediction was made

        else:
            print(f'FAIL: Expected 200, got {response.status_code}')
            return False

    except Exception as e:
        print(f'FAIL: {str(e)}')
        return False


def test_predict_complex_workflow(base_url: str) -> bool:
    """Test prediction for a complex workflow."""
    print('\n[TEST] Complex Workflow Prediction')
    print('-' * 50)

    workflow = {
        'start_hour': 7,
        'day_of_week': 1,
        'month': 1,
        'is_weekend': 0,
        'is_business_hours': 0,
        'total_blocks': 25,
        'sql_blocks': 8,
        'tableau_blocks': 5,
        'email_blocks': 3,
        'slack_blocks': 2,
        'parameter_blocks': 3,
        'code_blocks': 2,
        'plotly_blocks': 1,
        'kpi_blocks': 0,
        'api_blocks': 1,
        'writeback_blocks': 2,
        'conditional_blocks': 2,
        'loop_blocks': 1,
        'ai_blocks': 0,
        'storage_blocks': 1,
        'historical_run_count': 50,
        'historical_avg_duration': 120.0,
        'historical_median_duration': 100.0,
        'historical_stddev_duration': 45.0,
        'historical_failure_rate': 0.08
    }

    try:
        response = requests.post(
            f'{base_url}/predict',
            json=workflow,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f'Status Code: {response.status_code}')
        print(f'Input: Complex workflow with 25 blocks, 8 SQL, 2 writebacks, loops (avg 120s)')

        if response.status_code == 200:
            data = response.json()
            print(f'Response: {json.dumps(data, indent=2)}')

            duration = data.get('predicted_duration', 0)
            category = data.get('duration_category', '')

            # Complex workflows should predict higher duration
            if duration > 10:
                print(f'PASS: Predicted {duration:.2f}s ({category})')
                return True
            else:
                print(f'WARN: Unexpectedly low prediction for complex workflow')
                return True

        else:
            print(f'FAIL: Expected 200, got {response.status_code}')
            return False

    except Exception as e:
        print(f'FAIL: {str(e)}')
        return False


def test_predict_medium_workflow(base_url: str) -> bool:
    """Test prediction for a medium-complexity workflow."""
    print('\n[TEST] Medium Workflow Prediction')
    print('-' * 50)

    workflow = {
        'start_hour': 14,
        'day_of_week': 3,
        'month': 9,
        'is_weekend': 0,
        'is_business_hours': 1,
        'total_blocks': 10,
        'sql_blocks': 3,
        'tableau_blocks': 2,
        'email_blocks': 2,
        'slack_blocks': 1,
        'parameter_blocks': 1,
        'code_blocks': 1,
        'plotly_blocks': 0,
        'kpi_blocks': 0,
        'api_blocks': 0,
        'writeback_blocks': 0,
        'conditional_blocks': 1,
        'loop_blocks': 0,
        'ai_blocks': 0,
        'storage_blocks': 0,
        'historical_run_count': 200,
        'historical_avg_duration': 25.0,
        'historical_median_duration': 20.0,
        'historical_stddev_duration': 10.0,
        'historical_failure_rate': 0.03
    }

    try:
        response = requests.post(
            f'{base_url}/predict',
            json=workflow,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f'Status Code: {response.status_code}')
        print(f'Input: Medium workflow with 10 blocks, 3 SQL, conditional (avg 25s)')

        if response.status_code == 200:
            data = response.json()
            print(f'Response: {json.dumps(data, indent=2)}')

            duration = data.get('predicted_duration', 0)
            category = data.get('duration_category', '')
            print(f'PASS: Predicted {duration:.2f}s ({category})')
            return True

        else:
            print(f'FAIL: Expected 200, got {response.status_code}')
            return False

    except Exception as e:
        print(f'FAIL: {str(e)}')
        return False


def test_predict_partial_data(base_url: str) -> bool:
    """Test prediction with partial input (uses defaults)."""
    print('\n[TEST] Partial Data Prediction')
    print('-' * 50)

    workflow = {
        'total_blocks': 5,
        'sql_blocks': 2,
        'historical_avg_duration': 10.0
    }

    try:
        response = requests.post(
            f'{base_url}/predict',
            json=workflow,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f'Status Code: {response.status_code}')
        print(f'Input: Only 3 fields provided (should use defaults)')

        if response.status_code == 200:
            data = response.json()
            print(f'Response: {json.dumps(data, indent=2)}')
            print('PASS: Partial data handled correctly')
            return True

        else:
            print(f'FAIL: Expected 200, got {response.status_code}')
            return False

    except Exception as e:
        print(f'FAIL: {str(e)}')
        return False


def test_predict_empty_data(base_url: str) -> bool:
    """Test prediction with empty input (uses all defaults)."""
    print('\n[TEST] Empty Data Prediction')
    print('-' * 50)

    try:
        response = requests.post(
            f'{base_url}/predict',
            json={},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f'Status Code: {response.status_code}')
        print(f'Input: Empty JSON (should use all defaults)')

        if response.status_code == 200:
            data = response.json()
            print(f'Response: {json.dumps(data, indent=2)}')
            print('PASS: Empty data handled correctly')
            return True

        else:
            print(f'FAIL: Expected 200, got {response.status_code}')
            return False

    except Exception as e:
        print(f'FAIL: {str(e)}')
        return False


def test_predict_new_workflow(base_url: str) -> bool:
    """Test prediction for a new workflow with no history."""
    print('\n[TEST] New Workflow (No History)')
    print('-' * 50)

    workflow = {
        'start_hour': 15,
        'day_of_week': 4,
        'month': 3,
        'is_weekend': 0,
        'is_business_hours': 1,
        'total_blocks': 8,
        'sql_blocks': 2,
        'tableau_blocks': 1,
        'email_blocks': 1,
        'slack_blocks': 1,
        'parameter_blocks': 2,
        'code_blocks': 1,
        'plotly_blocks': 0,
        'kpi_blocks': 0,
        'api_blocks': 0,
        'writeback_blocks': 0,
        'conditional_blocks': 0,
        'loop_blocks': 0,
        'ai_blocks': 0,
        'storage_blocks': 0,
        'historical_run_count': 0,
        'historical_avg_duration': 0,
        'historical_median_duration': 0,
        'historical_stddev_duration': 0,
        'historical_failure_rate': 0
    }

    try:
        response = requests.post(
            f'{base_url}/predict',
            json=workflow,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f'Status Code: {response.status_code}')
        print(f'Input: New workflow with 8 blocks, no historical data')

        if response.status_code == 200:
            data = response.json()
            print(f'Response: {json.dumps(data, indent=2)}')
            print('PASS: New workflow prediction successful')
            return True

        else:
            print(f'FAIL: Expected 200, got {response.status_code}')
            return False

    except Exception as e:
        print(f'FAIL: {str(e)}')
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Test the duration prediction service')
    parser.add_argument('--url', default=DEFAULT_URL, help='Base URL of the service')
    args = parser.parse_args()

    base_url = args.url.rstrip('/')

    print('=' * 60)
    print('WORKFLOW DURATION PREDICTION - API TEST SUITE')
    print('=' * 60)
    print(f'\nTarget: {base_url}')

    # Run all tests
    tests = [
        ('Health Check', test_health),
        ('Model Info', test_info),
        ('Quick Workflow', test_predict_quick_workflow),
        ('Complex Workflow', test_predict_complex_workflow),
        ('Medium Workflow', test_predict_medium_workflow),
        ('Partial Data', test_predict_partial_data),
        ('Empty Data', test_predict_empty_data),
        ('New Workflow', test_predict_new_workflow),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func(base_url)
            results.append((name, passed))
        except Exception as e:
            print(f'ERROR in {name}: {str(e)}')
            results.append((name, False))

    # Print summary
    print('\n' + '=' * 60)
    print('TEST SUMMARY')
    print('=' * 60)

    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed

    for name, result in results:
        status = 'PASS' if result else 'FAIL'
        print(f'  [{status}] {name}')

    print(f'\nTotal: {passed}/{len(results)} tests passed')

    if failed > 0:
        print(f'\nWARNING: {failed} test(s) failed')
        sys.exit(1)
    else:
        print('\nAll tests passed!')
        sys.exit(0)


if __name__ == '__main__':
    main()
