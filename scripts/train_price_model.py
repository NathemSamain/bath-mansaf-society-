#!/usr/bin/env python3
"""
ML model training for stock price direction prediction.

Reads features from Supabase, trains baseline models, and saves the best one.

Usage:
    python scripts/train_price_model.py

Environment variables (read from .env file):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Feature columns for model training
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "volume_change_1d",
    "moving_average_5d",
    "moving_average_20d",
    "rsi_14",
    "macd",
    "macd_signal",
]

TARGET_COLUMN = "target_next_day_direction"

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_env_var(name: str) -> str:
    """Get environment variable or raise error."""
    value = os.environ.get(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


def init_supabase() -> Client:
    """Initialize Supabase client with service role key."""
    load_dotenv()  # Load from .env file

    url = get_env_var("SUPABASE_URL")
    key = get_env_var("SUPABASE_SERVICE_ROLE_KEY")
    client = create_client(url, key)
    logger.info("Connected to Supabase")
    return client


def fetch_features(supabase: Client) -> pd.DataFrame:
    """
    Fetch all feature rows from database with stock ticker mapping.
    Uses pagination to load all rows (Supabase defaults to 1000 max per request).
    """
    try:
        all_rows = []
        batch_size = 1000
        offset = 0

        while True:
            # Fetch features with pagination using range
            response = (
                supabase.table("features_daily")
                .select(
                    "date, stock_id, return_1d, return_5d, return_20d, "
                    "volatility_5d, volatility_20d, volume_change_1d, "
                    "moving_average_5d, moving_average_20d, rsi_14, macd, macd_signal, "
                    "target_next_day_direction"
                )
                .order("date", desc=False)
                .range(offset, offset + batch_size - 1)
                .execute()
            )

            if not response.data:
                break

            all_rows.extend(response.data)

            # Check if we got fewer rows than batch size (end of data)
            if len(response.data) < batch_size:
                break

            offset += batch_size

        if not all_rows:
            logger.error("No rows found in features_daily table")
            sys.exit(1)

        df = pd.DataFrame(all_rows)
        logger.info(f"Loaded {len(df)} total rows from features_daily")

        # Fetch stock tickers for mapping
        stocks_response = supabase.table("stocks").select("id, ticker").execute()

        if stocks_response.data:
            stock_map = {s["id"]: s["ticker"] for s in stocks_response.data}
            df["ticker"] = df["stock_id"].map(stock_map)

        return df

    except Exception as e:
        logger.error(f"Failed to fetch features from database: {e}")
        sys.exit(1)


def validate_columns(df: pd.DataFrame) -> None:
    """Check that all required columns exist."""
    required = FEATURE_COLUMNS + [TARGET_COLUMN, "date", "stock_id"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)

    logger.info(f"All {len(required)} required columns present")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing features or targets.
    """
    initial_count = len(df)

    # Check for missing values in features and target
    check_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df_clean = df.dropna(subset=check_cols)

    removed = initial_count - len(df_clean)

    if removed > 0:
        logger.info(f"Removed {removed} rows with missing values ({removed/initial_count*100:.1f}%)")

    if len(df_clean) == 0:
        logger.error("No valid rows remaining after cleaning")
        sys.exit(1)

    logger.info(f"Clean dataset: {len(df_clean)} rows")

    return df_clean.sort_values("date").reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: oldest 70% train, next 15% val, newest 15% test.
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(f"Time-based split:")
    logger.info(f"  Training:   {len(train_df)} rows ({TRAIN_RATIO*100:.0f}%)")
    logger.info(f"  Validation: {len(val_df)} rows ({VAL_RATIO*100:.0f}%)")
    logger.info(f"  Test:       {len(test_df)} rows ({TEST_RATIO*100:.0f}%)")

    if len(train_df) < 100:
        logger.warning(f"Training set is small ({len(train_df)} rows). Model may not generalize well.")

    return train_df, val_df, test_df


def prepare_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values.astype(int)
    return X, y


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Train a logistic regression model with feature scaling."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a random forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)

    metrics = {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
    }

    cm = confusion_matrix(y, y_pred)

    logger.info(f"\n{name} Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  Confusion Matrix:\n{cm}")

    return metrics


def save_model(model, metadata: dict) -> None:
    """Save model and metadata to disk."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "price_direction_model.joblib"
    metadata_path = models_dir / "price_direction_model_metadata.json"

    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to: {metadata_path}")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)


def train_model():
    """Main model training workflow."""
    logger.info("=" * 60)
    logger.info("Stock Price Direction Model Training")
    logger.info("=" * 60)

    # Initialize
    supabase = init_supabase()

    # Fetch data
    df = fetch_features(supabase)

    # Validate columns
    validate_columns(df)

    # Clean and sort
    df_clean = clean_data(df)

    # Split data
    train_df, val_df, test_df = split_data(df_clean)

    # Prepare matrices
    X_train, y_train = prepare_matrices(train_df)
    X_val, y_val = prepare_matrices(val_df)
    X_test, y_test = prepare_matrices(test_df)

    logger.info(f"\nFeature matrix shapes:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_val:   {X_val.shape}")
    logger.info(f"  X_test:  {X_test.shape}")

    # Train models
    logger.info("\n" + "-" * 60)
    logger.info("Training Baseline Models")
    logger.info("-" * 60)

    # Logistic Regression
    logger.info("\nTraining Logistic Regression...")
    try:
        lr_model = train_logistic_regression(X_train, y_train)
        lr_val_metrics = evaluate_model(lr_model, X_val, y_val, "Logistic Regression (Validation)")
    except Exception as e:
        logger.error(f"Logistic Regression training failed: {e}")
        lr_model = None
        lr_val_metrics = {"f1": 0}

    # Random Forest
    logger.info("\nTraining Random Forest...")
    try:
        rf_model = train_random_forest(X_train, y_train)
        rf_val_metrics = evaluate_model(rf_model, X_val, y_val, "Random Forest (Validation)")
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
        rf_model = None
        rf_val_metrics = {"f1": 0}

    # Select best model
    logger.info("\n" + "-" * 60)
    logger.info("Model Selection")
    logger.info("-" * 60)

    lr_f1 = lr_val_metrics.get("f1", 0)
    rf_f1 = rf_val_metrics.get("f1", 0)

    if lr_model is None and rf_model is None:
        logger.error("All models failed to train")
        sys.exit(1)

    if rf_f1 >= lr_f1 and rf_model is not None:
        best_model = rf_model
        best_name = "Random Forest"
        best_val_metrics = rf_val_metrics
        logger.info(f"\nBest model: Random Forest (F1 = {rf_f1:.4f})")
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_val_metrics = lr_val_metrics
        logger.info(f"\nBest model: Logistic Regression (F1 = {lr_f1:.4f})")

    # Evaluate on test set
    logger.info("\n" + "-" * 60)
    logger.info(f"Final Evaluation on Test Set: {best_name}")
    logger.info("-" * 60)

    test_metrics = evaluate_model(best_model, X_test, y_test, f"{best_name} (Test)")

    # Save model and metadata
    metadata = {
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    save_model(best_model, metadata)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Best model: {best_name}")
    logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"Model saved to: models/price_direction_model.joblib")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
