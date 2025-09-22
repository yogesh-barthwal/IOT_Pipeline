"""
train_model.py

Loads processed Parquet data, trains a RandomForest classifier to predict 'failure',
and saves the trained model + scaler to disk.

Usage:
    python src/train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier #Random Forest is a solid, reliable, and interpretable choice for 
                                                    # predicting machine failures with tabular sensor data.
                                                    # It balances performance, speed, and simplicity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# --- Paths setup ---
ROOT = os.path.join(os.path.dirname(__file__), '..')  # project root
PROC_DIR = os.path.join(ROOT, 'data', 'processed')    # processed data folder
PARQUET_PATH = os.path.join(PROC_DIR, 'processed.parquet')  # processed parquet file
MODELS_DIR = os.path.join(ROOT, 'models')            # folder to save models
os.makedirs(MODELS_DIR, exist_ok=True)               # ensure models dir exists

# Paths for saved model and scaler
MODEL_PATH = os.path.join(MODELS_DIR, 'latest_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Features to use for training
FEATURES = [
    'temperature', 'vibration', 'pressure', 'current', 'usage_hours',
    'temp_mean_5', 'temp_std_5', 'vib_mean_5', 'vib_std_5', 'current_mean_5', 'current_std_5'
]

# --- Function to load processed data ---
def load_data():
    """Load the processed parquet file into a pandas DataFrame."""
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError("Processed parquet not found. Run ETL first.")
    df = pd.read_parquet(PARQUET_PATH)
    # Could optionally filter to last N rows per machine if dataset is huge
    return df

# --- Main training function ---
def train_and_save():
    """Train RandomForest on processed data and save model + scaler."""
    df = load_data()

    # Drop rows where target 'failure' is missing
    df = df.dropna(subset=['failure'])

    # Ensure all expected features exist (fill missing with 0.0)- If any feature column is missing (e.g., due to ETL not generating it or sensor not available),
    # create it and fill with 0.0
    
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    # Features and target
    X = df[FEATURES].astype(float)
    y = df['failure'].astype(int)

    # Split data into train/test (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Scale features to zero mean, unit variance
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Initialize RandomForest classifier
    # class_weight='balanced' handles class imbalance
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # use all cores
    )

    print("Training model...")
    clf.fit(X_train_s, y_train)  # train model

    # Evaluate model
    preds = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)[:, 1]  # probability of failure class

    print("Classification report on holdout set:")
    print(classification_report(y_test, preds, digits=4))
    try:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        pass  # in case ROC-AUC fails (e.g., single class in y_test)

    # Save model and scaler to disk
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")

    return MODEL_PATH

# --- Entry point ---
if __name__ == "__main__":
    train_and_save()
