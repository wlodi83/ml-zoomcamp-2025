#!/usr/bin/env python
"""
Train Random Forest Model for Trial-to-Paid Conversion Prediction

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Configuration
DATA_FILE = 'data.csv'
MODEL_FILE = 'model.pkl'
VECTORIZER_FILE = 'dv.pkl'
RANDOM_STATE = 42


def load_data(filename):
    """Load dataset"""
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    print(f"✓ Loaded {len(df)} samples")
    return df


def prepare_features(df):
    """Prepare features for training"""
    print("\nPreparing features...")

    # Exclude non-feature columns (identifiers, dates, target, outcome)
    exclude_cols = [
        'customer_id', 'organization_id',
        'trial_start', 'trial_end', 'trial_subscription_id',
        'outcome', 'converted_to_paid'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Select features and target
    X = df[feature_cols].copy()
    y = df['converted_to_paid'].copy()

    # Fill missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    print(f"✓ Prepared {X.shape[1]} features")
    return X, y


def split_data(X, y):
    """Split data into train/val/test sets"""
    print("\nSplitting data...")

    # Split: 60% train, 20% val, 20% test
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_full_train
    )

    print(f"✓ Train: {len(X_train)} samples ({y_train.mean():.1%} converted)")
    print(f"✓ Val:   {len(X_val)} samples ({y_val.mean():.1%} converted)")
    print(f"✓ Test:  {len(X_test)} samples ({y_test.mean():.1%} converted)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize_features(X_train, X_val, X_test):
    """Vectorize features using DictVectorizer"""
    print("\nVectorizing features...")

    # Convert to dict format
    train_dicts = X_train.to_dict(orient='records')
    val_dicts = X_val.to_dict(orient='records')
    test_dicts = X_test.to_dict(orient='records')

    # Fit vectorizer
    dv = DictVectorizer(sparse=False)
    X_train_vec = dv.fit_transform(train_dicts)
    X_val_vec = dv.transform(val_dicts)
    X_test_vec = dv.transform(test_dicts)

    print(f"✓ Vectorized to {X_train_vec.shape[1]} features")

    return X_train_vec, X_val_vec, X_test_vec, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    """Train Random Forest model"""
    print("\nTraining Random Forest model...")

    # Best params from notebook output
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, dv):
    """Evaluate model performance"""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # Predictions
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_val = model.predict_proba(X_val)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    # AUC scores
    auc_train = roc_auc_score(y_train, y_pred_train)
    auc_val = roc_auc_score(y_val, y_pred_val)
    auc_test = roc_auc_score(y_test, y_pred_test)

    print(f"\nAUC Scores:")
    print(f"  Train: {auc_train:.4f}")
    print(f"  Val:   {auc_val:.4f}")
    print(f"  Test:  {auc_test:.4f}")

    # Classification report (threshold = 0.5)
    y_pred_binary = (y_pred_test >= 0.5).astype(int)

    print(f"\nClassification Report (Test Set, threshold=0.5):")
    print(classification_report(y_test, y_pred_binary, target_names=['Churned', 'Converted']))

    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_binary)
    print(f"                Predicted")
    print(f"                Churned  Converted")
    print(f"Actual Churned    {cm[0][0]:4d}      {cm[0][1]:4d}")
    print(f"       Converted  {cm[1][0]:4d}      {cm[1][1]:4d}")

    return auc_test


def save_model(model, dv):
    """Save model and vectorizer"""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    # Save Random Forest model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved model to: {MODEL_FILE}")

    # Save DictVectorizer
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(dv, f)
    print(f"✓ Saved vectorizer to: {VECTORIZER_FILE}")


def main():
    """Main training pipeline"""
    print("="*80)
    print("TRIAL-TO-PAID CONVERSION PREDICTION - MODEL TRAINING")
    print("="*80)

    # 1. Load data
    df = load_data(DATA_FILE)

    # 2. Prepare features
    X, y = prepare_features(df)

    # 3. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 4. Vectorize features
    X_train_vec, X_val_vec, X_test_vec, dv = vectorize_features(X_train, X_val, X_test)

    # 5. Train model
    model = train_model(X_train_vec, y_train, X_val_vec, y_val, dv)

    # 6. Evaluate model
    auc_test = evaluate_model(model, X_train_vec, y_train, X_val_vec, y_val, X_test_vec, y_test, dv)

    # 7. Save model
    save_model(model, dv)

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"✓ Model trained successfully")
    print(f"✓ Test AUC: {auc_test:.4f}")
    print(f"✓ Model artifacts saved")
    print(f"\nNext steps:")
    print(f"  1. Review model performance")
    print(f"  2. Test prediction service: python predict.py")
    print(f"  3. Build Docker image: docker build -t trial-conversion .")
    print(f"  4. Deploy to production")


if __name__ == '__main__':
    main()
