#!/usr/bin/env python3
"""
User Churn Prediction - Model Training Script
ML Zoomcamp Capstone 2 Project

This script trains an XGBoost model to predict user churn based on
behavioral and profile features extracted from PushMetrics production data.

Usage:
    python train.py

Output:
    - model.xgb: Trained XGBoost model
    - dv.pkl: DictVectorizer for feature transformation
    - model_info.pkl: Model metadata and performance metrics
"""

import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Suppress XGBoost file format warning for .xgb extension
warnings.filterwarnings('ignore', message='.*Saving model in the UBJSON format.*')


# Configuration
DATA_FILE = 'data.csv'
MODEL_FILE = 'model.xgb'
DV_FILE = 'dv.pkl'
MODEL_INFO_FILE = 'model_info.pkl'
RANDOM_STATE = 42

# Feature columns to use
FEATURE_COLUMNS = [
    'is_generic_email',
    'account_active',
    'login_count',
    'fail_login_count',
    'days_since_signup',
    'days_since_last_login',
    'signup_hour',
    'signup_day_of_week',
    'signup_month',
    'last_login_hour',
    'last_login_day_of_week',
    'has_first_name',
    'has_last_name',
    'profile_completeness',
    'workspace_count',
    'organization_count',
    'role_count',
    'is_admin',
    'is_guest',
    'has_avatar',
    'mfa_enabled',
    'login_frequency'
]

TARGET_COLUMN = 'churned'

# XGBoost parameters (tuned)
XGB_PARAMS = {
    'eta': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': RANDOM_STATE,
    'verbosity': 1
}
NUM_BOOST_ROUNDS = 200


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return the dataset."""
    print(f'Loading data from {filepath}...')
    df = pd.read_csv(filepath)
    print(f'Loaded {len(df)} samples')
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target from dataframe."""
    print(f'Preparing {len(FEATURE_COLUMNS)} features...')

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Handle missing values
    X = X.fillna(0)

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Split data into train/validation/test sets (60/20/20)."""
    print('Splitting data...')

    # First split: 80% train+val, 20% test
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Second split: 75% train, 25% val (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_full_train
    )

    print(f'  Train: {len(X_train)} samples ({y_train.mean()*100:.1f}% churned)')
    print(f'  Val:   {len(X_val)} samples ({y_val.mean()*100:.1f}% churned)')
    print(f'  Test:  {len(X_test)} samples ({y_test.mean()*100:.1f}% churned)')

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train: pd.DataFrame, X_val: pd.DataFrame,
                y_train: pd.Series, y_val: pd.Series) -> tuple:
    """Train XGBoost model with DictVectorizer."""
    print('\nTraining XGBoost model...')

    # Convert to dictionaries for DictVectorizer
    train_dicts = X_train.to_dict(orient='records')
    val_dicts = X_val.to_dict(orient='records')

    # Fit DictVectorizer
    dv = DictVectorizer(sparse=False)
    X_train_dv = dv.fit_transform(train_dicts)
    X_val_dv = dv.transform(val_dicts)

    # Get feature names from DictVectorizer
    feature_names = list(dv.get_feature_names_out())

    # Create DMatrix with feature names
    dtrain = xgb.DMatrix(X_train_dv, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_dv, label=y_val, feature_names=feature_names)

    watchlist = [(dtrain, 'train'), (dval, 'val')]

    # Train model
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=watchlist,
        verbose_eval=50
    )

    return model, dv


def evaluate_model(model: xgb.Booster, dv: DictVectorizer,
                   X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> dict:
    """Evaluate model on all datasets."""
    print('\nEvaluating model...')

    # Transform data
    X_train_dv = dv.transform(X_train.to_dict(orient='records'))
    X_val_dv = dv.transform(X_val.to_dict(orient='records'))
    X_test_dv = dv.transform(X_test.to_dict(orient='records'))

    # Get feature names from DictVectorizer
    feature_names = list(dv.get_feature_names_out())

    # Create DMatrix with feature names
    dtrain = xgb.DMatrix(X_train_dv, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_dv, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test_dv, feature_names=feature_names)

    # Predict
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    test_pred = model.predict(dtest)

    # Calculate AUC
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    print(f'\nAUC Scores:')
    print(f'  Train: {train_auc:.4f}')
    print(f'  Val:   {val_auc:.4f}')
    print(f'  Test:  {test_auc:.4f}')
    print(f'  Overfitting (train-val): {train_auc - val_auc:.4f}')

    return {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc
    }


def save_artifacts(model: xgb.Booster, dv: DictVectorizer, metrics: dict):
    """Save model artifacts to disk."""
    print('\nSaving model artifacts...')

    # Save DictVectorizer
    with open(DV_FILE, 'wb') as f:
        pickle.dump(dv, f)
    print(f'  Saved DictVectorizer to {DV_FILE}')

    # Save XGBoost model
    model.save_model(MODEL_FILE)
    print(f'  Saved XGBoost model to {MODEL_FILE}')

    # Save model info
    model_info = {
        'model_type': 'XGBoost',
        'params': XGB_PARAMS,
        'num_boost_rounds': NUM_BOOST_ROUNDS,
        'train_auc': metrics['train_auc'],
        'val_auc': metrics['val_auc'],
        'test_auc': metrics['test_auc'],
        'feature_columns': FEATURE_COLUMNS,
        'n_features': len(FEATURE_COLUMNS)
    }

    with open(MODEL_INFO_FILE, 'wb') as f:
        pickle.dump(model_info, f)
    print(f'  Saved model info to {MODEL_INFO_FILE}')


def main():
    """Main training pipeline."""
    print('=' * 70)
    print('USER CHURN PREDICTION - MODEL TRAINING')
    print('=' * 70)

    # Load data
    df = load_data(DATA_FILE)

    # Prepare features
    X, y = prepare_features(df)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train model
    model, dv = train_model(X_train, X_val, y_train, y_val)

    # Evaluate model
    metrics = evaluate_model(model, dv, X_train, X_val, X_test,
                            y_train, y_val, y_test)

    # Save artifacts
    save_artifacts(model, dv, metrics)

    print('\n' + '=' * 70)
    print('Training complete!')
    print('=' * 70)


if __name__ == '__main__':
    main()
