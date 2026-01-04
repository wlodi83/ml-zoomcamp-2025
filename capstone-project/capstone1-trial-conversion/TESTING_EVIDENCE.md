# Testing Evidence

This document captures the live deployment checks for the Trial-to-Paid Conversion Predictor.
Fill in the placeholders after you verify the live endpoints.

## Deployment Details

- **Date**: 2026-01-04
- **Environment**: AWS EKS (eu-central-1)
- **Public URL**: https://trial-conversion.pushmetrics.io
- **Image**: 133737826969.dkr.ecr.eu-central-1.amazonaws.com/trial-to-paid:latest
- **Git Commit**: `<commit-sha>`

## Health Check

Command:
```bash
curl https://trial-conversion.pushmetrics.io/health
```

Observed response:
```json
{
  "model": "model.pkl",
  "service": "trial-conversion-predictor",
  "status": "healthy"
}
```

## Service Info

Command:
```bash
curl https://trial-conversion.pushmetrics.io/
```

Observed response:
```json
{
  "endpoints": {
    "/": "GET - Service info",
    "/health": "GET - Health check",
    "/predict": "POST - Predict conversion probability"
  },
  "example_request": {
    "body": {
      "activated_day1": 1,
      "execution_success_rate": 0.93,
      "executions_total": 15,
      "has_bigquery": 1,
      "has_sql_block": 1,
      "hours_to_first_execution": 2.5,
      "reports_created": 3,
      "schedules_created": 1
    },
    "headers": {
      "Content-Type": "application/json"
    },
    "method": "POST",
    "url": "http://localhost:9696/predict"
  },
  "service": "Trial-to-Paid Conversion Predictor",
  "version": "1.0.0"
}
```

## Prediction

Command:
```bash
curl -X POST https://trial-conversion.pushmetrics.io/predict \
  -H "Content-Type: application/json" \
  -d '{
    "users_invited": 5,
    "is_generic_email": 0,
    "trial_start_hour": 10,
    "trial_start_day_of_week": 2,
    "hours_to_first_execution": 2.5,
    "execution_growth_rate": 0.15,
    "reports_created": 5,
    "executions_total": 25,
    "execution_success_rate": 0.92,
    "activated_day1": 1,
    "activated_day2": 1,
    "reports_week1": 1,
    "executions_week1": 5,
    "schedules_active": 1,
    "has_sql_block": 1,
    "has_bigquery": 1,
    "schedules_created": 2,
    "sql_connections_count": 1,
    "blocks_created": 15
  }'
```

Observed response:
```json
{
  "conversion_probability": 0.4779,
  "will_convert": false,
  "risk_level": "medium",
  "recommendation": "Moderate risk - consider sales outreach or onboarding assistance"
}
```
