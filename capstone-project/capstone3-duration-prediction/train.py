#!/usr/bin/env python3
"""
Workflow Execution Duration Prediction - Model Training Script

This script trains an XGBoost regression model to predict workflow execution
duration based on workflow structure, timing, and historical performance features.

Usage:
    python train.py

Outputs:
    - model.xgb: Trained XGBoost model
    - dv.pkl: DictVectorizer for feature transformation
    - model_info.pkl: Model metadata and metrics
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_FILE = 'data.csv'
MODEL_FILE = 'model.xgb'
DV_FILE = 'dv.pkl'
MODEL_INFO_FILE = 'model_info.pkl'
RANDOM_STATE = 42

# Feature columns
FEATURE_COLUMNS = [
    # Timing features
    'start_hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours',
    # Block composition features
    'total_blocks', 'sql_blocks', 'tableau_blocks', 'email_blocks', 'slack_blocks',
    'parameter_blocks', 'code_blocks', 'plotly_blocks', 'kpi_blocks', 'api_blocks',
    'writeback_blocks', 'conditional_blocks', 'loop_blocks', 'ai_blocks', 'storage_blocks',
    # Historical performance features
    'historical_run_count', 'historical_avg_duration', 'historical_median_duration',
    'historical_stddev_duration', 'historical_failure_rate'
]

TARGET_COLUMN = 'duration_seconds'

# XGBoost hyperparameters (tuned)
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 8,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': RANDOM_STATE,
    'verbosity': 1
}

NUM_BOOST_ROUNDS = 300
EARLY_STOPPING_ROUNDS = 30


# =============================================================================
# Functions
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f'Loaded {len(df):,} samples from {filepath}')
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target for modeling."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Fill missing values with 0
    X = X.fillna(0)

    print(f'Prepared {len(FEATURE_COLUMNS)} features')
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Split data into train, validation, and test sets (60/20/20)."""
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Second split: 75% train, 25% val (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE
    )

    print(f'Data split:')
    print(f'  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)')
    print(f'  Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)')
    print(f'  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)')

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train: pd.DataFrame, X_val: pd.DataFrame,
                y_train: pd.Series, y_val: pd.Series) -> tuple:
    """Train XGBoost model with DictVectorizer."""

    # Convert to dictionary format
    train_dicts = X_train.to_dict(orient='records')
    val_dicts = X_val.to_dict(orient='records')

    # Create and fit DictVectorizer
    dv = DictVectorizer(sparse=False)
    X_train_vec = dv.fit_transform(train_dicts)
    X_val_vec = dv.transform(val_dicts)

    feature_names = list(dv.get_feature_names_out())

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_vec, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_vec, label=y_val, feature_names=feature_names)

    watchlist = [(dtrain, 'train'), (dval, 'val')]

    # Train model
    print(f'\nTraining XGBoost model...')
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50
    )

    print(f'\nBest iteration: {model.best_iteration}')

    return model, dv


def evaluate_model(model: xgb.Booster, dv: DictVectorizer,
                   X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> dict:
    """Evaluate model on all data splits."""

    feature_names = list(dv.get_feature_names_out())

    # Transform data
    X_train_vec = dv.transform(X_train.to_dict(orient='records'))
    X_val_vec = dv.transform(X_val.to_dict(orient='records'))
    X_test_vec = dv.transform(X_test.to_dict(orient='records'))

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_vec, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_vec, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test_vec, feature_names=feature_names)

    # Predictions
    y_train_pred = model.predict(dtrain)
    y_val_pred = model.predict(dval)
    y_test_pred = model.predict(dtest)

    # Calculate metrics
    metrics = {}

    for name, y_true, y_pred in [('train', y_train, y_train_pred),
                                   ('val', y_val, y_val_pred),
                                   ('test', y_test, y_test_pred)]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics[f'{name}_rmse'] = rmse
        metrics[f'{name}_mae'] = mae
        metrics[f'{name}_r2'] = r2

    print(f'\nModel Performance:')
    print(f'  Train - RMSE: {metrics["train_rmse"]:.4f}, MAE: {metrics["train_mae"]:.4f}, R²: {metrics["train_r2"]:.4f}')
    print(f'  Val   - RMSE: {metrics["val_rmse"]:.4f}, MAE: {metrics["val_mae"]:.4f}, R²: {metrics["val_r2"]:.4f}')
    print(f'  Test  - RMSE: {metrics["test_rmse"]:.4f}, MAE: {metrics["test_mae"]:.4f}, R²: {metrics["test_r2"]:.4f}')

    return metrics


def save_artifacts(model: xgb.Booster, dv: DictVectorizer, metrics: dict, best_iteration: int):
    """Save model artifacts to disk."""

    # Save DictVectorizer
    with open(DV_FILE, 'wb') as f:
        pickle.dump(dv, f)
    print(f'\n✓ Saved: {DV_FILE}')

    # Save XGBoost model
    model.save_model(MODEL_FILE)
    print(f'✓ Saved: {MODEL_FILE}')

    # Save model info
    model_info = {
        'model_type': 'XGBoost Regressor',
        'params': XGB_PARAMS,
        'num_boost_round': best_iteration,
        'metrics': metrics,
        'feature_columns': FEATURE_COLUMNS,
        'n_features': len(FEATURE_COLUMNS)
    }

    with open(MODEL_INFO_FILE, 'wb') as f:
        pickle.dump(model_info, f)
    print(f'✓ Saved: {MODEL_INFO_FILE}')


def main():
    """Main training pipeline."""
    print('=' * 70)
    print('WORKFLOW EXECUTION DURATION PREDICTION - MODEL TRAINING')
    print('=' * 70)

    # Check if data file exists
    if not Path(DATA_FILE).exists():
        print(f'\nError: Data file {DATA_FILE} not found!')
        print('Please ensure data.csv is in the current directory.')
        return

    # Load data
    print(f'\nLoading data from {DATA_FILE}...')
    df = load_data(DATA_FILE)

    # Prepare features
    print(f'\nPreparing features...')
    X, y = prepare_features(df)

    # Split data
    print(f'\nSplitting data...')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train model
    model, dv = train_model(X_train, X_val, y_train, y_val)

    # Evaluate model
    metrics = evaluate_model(model, dv, X_train, X_val, X_test,
                            y_train, y_val, y_test)

    # Save artifacts
    print(f'\nSaving model artifacts...')
    save_artifacts(model, dv, metrics, model.best_iteration)

    print('\n' + '=' * 70)
    print('TRAINING COMPLETE!')
    print('=' * 70)
    print(f'\nFinal Test Performance:')
    print(f'  RMSE: {metrics["test_rmse"]:.4f} seconds')
    print(f'  MAE:  {metrics["test_mae"]:.4f} seconds')
    print(f'  R²:   {metrics["test_r2"]:.4f}')


if __name__ == '__main__':
    main()
