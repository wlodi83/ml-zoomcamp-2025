#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Report Failure Prediction Model

This script trains an XGBoost model to predict report execution failures.

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Configuration
DATA_FILE = 'report_executions.csv'
MODEL_FILE = 'model.xgb'
DV_FILE = 'dv.pkl'
MODEL_INFO_FILE = 'model_info.pkl'

# Feature columns to exclude
EXCLUDE_COLS = [
    'execution_id',      # identifier
    'report_hash',       # identifier
    'workspace_hash',    # identifier
    'duration_seconds',  # only known after execution
    'execution_date',    # for train/test split
    'failed'             # target variable
]

# XGBoost hyperparameters
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 10,
    'min_child_weight': 1,
    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}

NUM_BOOST_ROUNDS = 200


def load_data(file_path):
    """Load data from CSV file."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows")
    return df


def split_data(df):
    """
    Split data into train/val/test sets.

    Train: 60%, Validation: 20%, Test: 20%
    """
    print("\nSplitting data...")

    # Convert to datetime and sort
    df['execution_date'] = pd.to_datetime(df['execution_date'])
    df_sorted = df.sort_values('execution_date').reset_index(drop=True)

    n = len(df_sorted)

    # Split: first 80% for full_train, 20% for test
    full_train, df_test = train_test_split(df_sorted, test_size=0.2, random_state=1)

    # Split full_train: 75% for train (60% of total), 25% for val (20% of total)
    df_train, df_val = train_test_split(full_train, test_size=0.25, random_state=1)

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"Training set:   {len(df_train):,} rows ({len(df_train)/n*100:.1f}%)")
    print(f"  Date range: {df_train['execution_date'].min()} to {df_train['execution_date'].max()}")
    print(f"  Failure rate: {df_train['failed'].mean()*100:.2f}%")

    print(f"Validation set: {len(df_val):,} rows ({len(df_val)/n*100:.1f}%)")
    print(f"  Date range: {df_val['execution_date'].min()} to {df_val['execution_date'].max()}")
    print(f"  Failure rate: {df_val['failed'].mean()*100:.2f}%")

    print(f"Test set:       {len(df_test):,} rows ({len(df_test)/n*100:.1f}%)")
    print(f"  Date range: {df_test['execution_date'].min()} to {df_test['execution_date'].max()}")
    print(f"  Failure rate: {df_test['failed'].mean()*100:.2f}%")

    return df_train, df_val, df_test


def prepare_features(df, feature_cols):
    """
    Prepare features for modeling.

    Returns:
        List of dictionaries (for DictVectorizer)
    """
    df_features = df[feature_cols].copy()

    # Fill missing values
    df_features = df_features.fillna(0)

    # Convert to list of dicts
    return df_features.to_dict(orient='records')


def train_model(X_train, y_train, X_val, y_val, params, num_rounds, feature_names=None):
    """
    Train XGBoost model.

    Returns:
        Trained model, training AUC, validation AUC
    """
    print("\nTraining XGBoost model...")
    print(f"Parameters: {params}")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    # Train
    watchlist = [(dtrain, 'train'), (dval, 'val')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=watchlist,
        verbose_eval=5
    )

    # Evaluate
    y_train_pred = model.predict(dtrain)
    y_val_pred = model.predict(dval)

    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)

    print(f"\nTraining complete!")
    print(f"  Training AUC:    {train_auc:.4f}")
    print(f"  Validation AUC:  {val_auc:.4f}")
    print(f"  Difference:      {abs(train_auc - val_auc):.4f}")

    return model, train_auc, val_auc


def evaluate_on_test(model, X_test, y_test, feature_names=None):
    """Evaluate model on test set."""
    print("\nEvaluating on test set...")

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    y_test_pred = model.predict(dtest)
    test_auc = roc_auc_score(y_test, y_test_pred)

    print(f"  Test AUC: {test_auc:.4f}")

    return test_auc


def save_model(model, dv, model_info):
    """Save model, DictVectorizer, and metadata."""
    print("\nSaving model files...")

    # Save DictVectorizer
    with open(DV_FILE, 'wb') as f:
        pickle.dump(dv, f)
    print(f"  Saved DictVectorizer to '{DV_FILE}'")

    # Save XGBoost model
    model.save_model(MODEL_FILE)
    print(f"  Saved XGBoost model to '{MODEL_FILE}'")

    # Save model metadata
    with open(MODEL_INFO_FILE, 'wb') as f:
        pickle.dump(model_info, f)
    print(f"  Saved model metadata to '{MODEL_INFO_FILE}'")

    print("\nAll files saved successfully!")


def main():
    """Main training pipeline."""
    print("="*60)
    print("REPORT FAILURE PREDICTION - MODEL TRAINING")
    print("="*60)

    # 1. Load data
    df = load_data(DATA_FILE)

    # 2. Split data
    df_train, df_val, df_test = split_data(df)

    # 3. Prepare features
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    print(f"\nUsing {len(feature_cols)} features")

    X_train_dicts = prepare_features(df_train, feature_cols)
    X_val_dicts = prepare_features(df_val, feature_cols)
    X_test_dicts = prepare_features(df_test, feature_cols)

    y_train = df_train['failed'].values
    y_val = df_val['failed'].values
    y_test = df_test['failed'].values

    # 4. Apply DictVectorizer
    print("\nApplying DictVectorizer...")
    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(X_train_dicts)
    X_val = dv.transform(X_val_dicts)
    X_test = dv.transform(X_test_dicts)

    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Total features after one-hot encoding: {len(dv.get_feature_names_out())}")

    # 5. Train model
    feature_names = dv.get_feature_names_out()
    model, train_auc, val_auc = train_model(
        X_train, y_train,
        X_val, y_val,
        XGB_PARAMS,
        NUM_BOOST_ROUNDS,
        feature_names=feature_names
    )

    # 6. Evaluate on test set
    test_auc = evaluate_on_test(model, X_test, y_test, feature_names=feature_names)

    # 7. Save model
    # Identify numerical and categorical features
    numerical_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in feature_cols if col not in numerical_features]

    model_info = {
        'model_type': 'XGBoost',
        'params': XGB_PARAMS,
        'num_boost_rounds': NUM_BOOST_ROUNDS,
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'test_auc': float(test_auc),
        'feature_cols': feature_cols,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'n_features': len(dv.get_feature_names_out())
    }

    save_model(model, dv, model_info)

    # 8. Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model type:      XGBoost")
    print(f"Num rounds:      {NUM_BOOST_ROUNDS}")
    print(f"Training AUC:    {train_auc:.4f}")
    print(f"Validation AUC:  {val_auc:.4f}")
    print(f"Test AUC:        {test_auc:.4f}")
    print(f"\nModel files saved:")
    print(f"  - {MODEL_FILE}")
    print(f"  - {DV_FILE}")
    print(f"  - {MODEL_INFO_FILE}")
    print("="*60)


if __name__ == '__main__':
    main()
