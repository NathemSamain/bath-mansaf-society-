#!/usr/bin/env python3
"""
Generate daily stock price direction predictions using trained model.

Loads the trained model, reads latest features for each stock, and inserts
predictions into the predictions table.

Usage:
    python scripts/generate_daily_predictions.py

Environment variables (read from .env file):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import sys
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import joblib
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "price_direction_model.joblib"
METADATA_FILE = MODELS_DIR / "price_direction_model_metadata.json"


def get_env_var(name: str) -> str:
    """Get environment variable or raise error."""
    value = os.environ.get(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


def init_supabase() -> Client:
    """Initialize Supabase client with service role key."""
    load_dotenv()

    url = get_env_var("SUPABASE_URL")
    key = get_env_var("SUPABASE_SERVICE_ROLE_KEY")
    client = create_client(url, key)
    logger.info("Connected to Supabase")
    return client


def load_model():
    """Load trained model from disk."""
    if not MODEL_FILE.exists():
        logger.error(f"Model file not found: {MODEL_FILE}")
        sys.exit(1)

    try:
        model = joblib.load(MODEL_FILE)
        logger.info(f"Model loaded from: {MODEL_FILE}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def load_metadata() -> dict:
    """Load model metadata from disk."""
    if not METADATA_FILE.exists():
        logger.error(f"Metadata file not found: {METADATA_FILE}")
        sys.exit(1)

    try:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded: {metadata.get('model_name', 'Unknown')}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        sys.exit(1)


def fetch_active_stocks(supabase: Client) -> list[dict]:
    """Fetch all active stocks from database."""
    try:
        response = supabase.table("stocks").select("id, ticker").eq("is_active", True).execute()

        if not response.data:
            logger.warning("No active stocks found")
            return []

        logger.info(f"Found {len(response.data)} active stocks")
        return response.data

    except Exception as e:
        logger.error(f"Failed to fetch stocks: {e}")
        sys.exit(1)


def fetch_latest_features(supabase: Client, stock_id: str, ticker: str) -> dict | None:
    """
    Fetch the latest feature row for a specific stock.
    Returns the row with feature_date included.
    """
    try:
        response = (
            supabase.table("features_daily")
            .select("*")
            .eq("stock_id", stock_id)
            .order("date", desc=True)
            .limit(1)
            .execute()
        )

        if not response.data:
            logger.warning(f"No features found for {ticker}")
            return None

        return response.data[0]

    except Exception as e:
        logger.error(f"Failed to fetch features for {ticker}: {e}")
        return None


def prepare_features(feature_row: dict, feature_columns: list[str], ticker: str) -> np.ndarray | None:
    """
    Extract and validate features from a feature row.
    """
    features = []

    for col in feature_columns:
        if col not in feature_row:
            logger.warning(f"Missing feature column '{col}' for {ticker}")
            return None

        value = feature_row[col]
        if value is None:
            logger.warning(f"Feature '{col}' is null for {ticker}")
            return None

        features.append(float(value))

    return np.array(features).reshape(1, -1)


def generate_prediction(model, features: np.ndarray) -> dict:
    """
    Generate prediction using the trained model.
    Returns dict with direction, probabilities, and confidence.
    """
    # Get prediction
    predicted_direction = model.predict(features)[0]

    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        # probabilities[0] = P(class 0), probabilities[1] = P(class 1)
        probability_down = float(probabilities[0])
        probability_up = float(probabilities[1])
    else:
        # Fallback if no probabilities available
        probability_up = 0.5
        probability_down = 0.5

    # Confidence is the max probability
    confidence_score = max(probability_up, probability_down)

    return {
        "predicted_direction": int(predicted_direction),
        "probability_up": round(probability_up, 6),
        "probability_down": round(probability_down, 6),
        "confidence_score": round(confidence_score, 6),
    }


def insert_prediction(
    supabase: Client,
    stock_id: str,
    feature_date: date,
    target_date: date,
    prediction: dict,
    model_name: str,
) -> bool:
    """
    Insert or update prediction in database.

    Uses prediction_date = feature_date (the date of data used for prediction).
    Uses target_date = next calendar day (TODO: should be next trading day).

    Note: The predictions table does not have a feature_date column,
    so we use prediction_date to store the feature_date.
    """
    record = {
        "stock_id": stock_id,
        "prediction_date": str(feature_date),  # Date of features used
        "target_date": str(target_date),       # Date being predicted
        "predicted_direction": prediction["predicted_direction"],
        "probability_up": prediction["probability_up"],
        "probability_down": prediction["probability_down"],
        "confidence_score": prediction["confidence_score"],
        "model_id": None,  # No model registry, set to NULL
        "explanation": {
            "model_name": model_name,
            "source": "price_only",
            "feature_date": str(feature_date),  # Store in explanation for reference
        },
    }

    try:
        # Check if prediction already exists for this stock/date combination
        existing = (
            supabase.table("predictions")
            .select("id")
            .eq("stock_id", stock_id)
            .eq("prediction_date", str(feature_date))
            .eq("target_date", str(target_date))
            .is_("model_id", "null")
            .execute()
        )

        if existing.data:
            # Update existing prediction
            pred_id = existing.data[0]["id"]
            supabase.table("predictions").update(record).eq("id", pred_id).execute()
            logger.info(f"  Updated existing prediction")
        else:
            # Insert new prediction
            supabase.table("predictions").insert(record).execute()
            logger.info(f"  Inserted new prediction")

        return True

    except Exception as e:
        logger.error(f"Failed to insert prediction: {e}")
        return False


def generate_daily_predictions():
    """Main prediction workflow."""
    logger.info("=" * 60)
    logger.info("Daily Prediction Generation")
    logger.info("=" * 60)

    # Initialize
    supabase = init_supabase()

    # Load model and metadata
    model = load_model()
    metadata = load_metadata()

    model_name = metadata.get("model_name", "Unknown")
    feature_columns = metadata.get("feature_columns", [])

    if not feature_columns:
        logger.error("No feature columns found in metadata")
        sys.exit(1)

    logger.info(f"Model: {model_name}")
    logger.info(f"Features: {len(feature_columns)} columns")

    # Fetch active stocks
    stocks = fetch_active_stocks(supabase)

    if not stocks:
        logger.error("No active stocks to process")
        sys.exit(1)

    logger.info("-" * 60)

    # Process each stock
    predictions_inserted = 0
    stocks_skipped = 0

    for i, stock in enumerate(stocks, 1):
        ticker = stock["ticker"]
        stock_id = stock["id"]

        logger.info(f"\n[{i}/{len(stocks)}] {ticker}")

        # Fetch latest features
        features_row = fetch_latest_features(supabase, stock_id, ticker)

        if not features_row:
            stocks_skipped += 1
            continue

        # Extract the feature date
        feature_date_str = features_row.get("date")
        if not feature_date_str:
            logger.warning(f"No date found in features for {ticker}")
            stocks_skipped += 1
            continue

        # Parse feature date
        try:
            feature_date = date.fromisoformat(str(feature_date_str))
        except ValueError as e:
            logger.error(f"Invalid feature date '{feature_date_str}' for {ticker}: {e}")
            stocks_skipped += 1
            continue

        # TODO: Calculate next trading day instead of next calendar day.
        # This should skip weekends and market holidays.
        # For now, use next calendar day.
        target_date = feature_date + timedelta(days=1)

        # Log feature date clearly since predictions table doesn't have this column
        logger.info(f"  Feature date: {feature_date} (data used for prediction)")
        logger.info(f"  Target date: {target_date} (date being predicted)")

        # Prepare feature vector
        features = prepare_features(features_row, feature_columns, ticker)

        if features is None:
            stocks_skipped += 1
            continue

        # Generate prediction
        prediction = generate_prediction(model, features)

        direction_str = "UP" if prediction["predicted_direction"] == 1 else "DOWN"
        logger.info(f"  Prediction: {direction_str}")
        logger.info(f"  Probability UP:   {prediction['probability_up']:.4f}")
        logger.info(f"  Probability DOWN: {prediction['probability_down']:.4f}")
        logger.info(f"  Confidence:       {prediction['confidence_score']:.4f}")

        # Insert prediction using feature_date as prediction_date
        success = insert_prediction(
            supabase,
            stock_id,
            feature_date,
            target_date,
            prediction,
            model_name,
        )

        if success:
            predictions_inserted += 1
        else:
            stocks_skipped += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Prediction Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Stocks processed: {len(stocks)}")
    logger.info(f"Predictions inserted/updated: {predictions_inserted}")
    logger.info(f"Stocks skipped: {stocks_skipped}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        generate_daily_predictions()
    except KeyboardInterrupt:
        logger.info("\nPrediction generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
