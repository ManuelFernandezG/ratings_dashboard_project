# -*- coding: utf-8 -*-
"""
src/modeling/train_models.py

Trains Logistic Regression and XGBoost models for credit risk prediction.
Saves the trained models and their results for later evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os
import json

# --- 1. CONFIGURATION ---

# Define the directory to save trained models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Define the features (must match the columns created in build_features.py)
FEATURE_COLUMNS = [
    "LEVERAGE",
    "PROFIT_MARGIN",
    "LIQUIDITY_RATIO",
    "HIGH_YIELD_OAS",
    "IG_OAS",
    "TED_SPREAD",
    "VIX",
    "FED_FUNDS",
]
TARGET_COLUMN = "TARGET_HIGH_RISK"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- 2. DATA PREPARATION ---


def load_and_split_data(path: str = "data/processed/features_and_target.csv"):
    """Loads processed data and splits it into training and testing sets."""
    print("Loading data and splitting into train/test sets...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "features_and_target.csv not found. Please run src/preprocessing/build_features.py first."
        )

    # Ensure all required features are present
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    # Split while maintaining the ratio of the target variable (stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# --- 3. MODEL TRAINING FUNCTIONS ---


def train_logistic_regression(X_train, y_train):
    """Trains and saves the Logistic Regression model (Interpretable Baseline)."""
    print("\nTraining Logistic Regression...")
    # Use 'balanced' class_weight to help address the imbalanced nature of credit risk data
    model = LogisticRegression(
        random_state=RANDOM_STATE, 
        solver="liblinear", 
        max_iter=1000, 
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, "log_reg_model.pkl"))
    print("Saved Logistic Regression model to models/log_reg_model.pkl")

    # Save coefficients for interpretability
    coefficients = dict(zip(FEATURE_COLUMNS, model.coef_[0].tolist()))
    with open(os.path.join(MODEL_DIR, "log_reg_coefficients.json"), "w") as f:
        json.dump(coefficients, f, indent=4)
    print("Saved Logistic Regression coefficients to models/log_reg_coefficients.json")
    return model


def train_xgboost(X_train, y_train):
    """Trains and saves the XGBoost Classifier (High Performance)."""
    print("\nTraining XGBoost Classifier...")
    
    # Calculate scale_pos_weight to handle class imbalance (more negatives than positives)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, 
        n_jobs=-1 # Use all cores
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    print("Saved XGBoost model to models/xgboost_model.pkl")
    return model


# --- 4. MAIN EXECUTION ---


def main():
    """Executes the data loading and model training pipeline."""
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Train and save models
    train_logistic_regression(X_train, y_train)
    train_xgboost(X_train, y_train)
    
    # Save test data for consistent evaluation across all metrics
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test
    # NOTE: This file is essential for the evaluation and SHAP analysis steps!
    test_data.to_csv(os.path.join(MODEL_DIR, "test_data_for_evaluation.csv"), index=False)
    print("\nSaved test data for consistent evaluation to models/test_data_for_evaluation.csv")


if __name__ == "__main__":
    main()