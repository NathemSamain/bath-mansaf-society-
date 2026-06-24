#!/usr/bin/env python3
"""
ML model training for stock price direction prediction.

Reads features from Supabase, trains baseline models, and saves the best one.

This version adds trustworthy-evaluation upgrades on top of the original
price-only baseline:
  1. StockNet-style label thresholding using `target_next_day_return`
     (drop near-flat "noise" days so the model learns real moves).
  2. Matthews Correlation Coefficient (MCC) + ROC-AUC alongside Acc/Prec/Rec/F1.
  3. Naive baselines: Always-UP, Persistence, Random (class-balanced).
  4. Lag-mimic diagnostic (does the model just copy yesterday's direction?).
  5. Probability calibration via CalibratedClassifierCV.
  6. A single comparison table across models + baselines.
  7. Model selection by validation MCC (F1 as tie-breaker).
  8. Scale-free price features (MA/MACD ratios instead of dollar levels).
  9. Date-based train/val/test split (keeps all tickers of a day together).
 10. Market-wide context features (SPY/QQQ/VIX), joined by date when available.

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
from datetime import datetime, timezone
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Raw feature columns fetched from features_daily.
RAW_FETCH_COLUMNS = [
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

# Already scale-free, used directly.
SCALE_FREE_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "volume_change_1d",
    "rsi_14",
]

# Price-scale features (moving averages, MACD) are in dollars and not
# comparable across a $20 and a $500 stock. We replace them with scale-free
# ratios so the model learns shape, not price level (suggestion #1).
DERIVED_FEATURES = [
    "ma_5_vs_20",       # moving_average_5d / moving_average_20d - 1
    "macd_norm",        # macd / moving_average_20d
    "macd_signal_norm", # macd_signal / moving_average_20d
    "macd_hist_norm",   # (macd - macd_signal) / moving_average_20d
]

# Market-wide context (suggestion #6), fetched from Yahoo Finance and joined by
# date. All values are as-of the same day as the stock's own features (no
# lookahead). If the fetch fails, training proceeds with price-only features.
MARKET_FEATURES = [
    "mkt_spy_ret_1d",
    "mkt_spy_ret_5d",
    "mkt_qqq_ret_1d",
    "mkt_vix",
    "mkt_vix_chg_1d",
]
MARKET_TICKERS = {"SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX"}

# Stage 2 sentiment features (from news_articles -> news_sentiment, aggregated
# into features_daily by build_sentiment_features.py). Rows with no news are
# treated as neutral (filled with 0) rather than dropped.
SENTIMENT_FEATURES = ["news_count", "avg_sentiment_score"]

# Active feature set used for training. Set at runtime in train_model() once we
# know whether market features are available.
FEATURE_COLUMNS = SCALE_FREE_FEATURES + DERIVED_FEATURES

# Continuous next-day return (stored in PERCENT, e.g. 1.25 == +1.25%).
# We build the training label from this column instead of trusting the
# pre-computed direction, so we can drop near-flat "noise" days.
RETURN_COLUMN = "target_next_day_return"
LABEL_COLUMN = "label"

# StockNet-style label thresholding (in percent). Moves inside the band
# [DOWN, UP] are treated as noise and dropped from training/evaluation.
LABEL_THRESHOLD_UP = 0.5      # return > +0.5%  -> 1 (UP)
LABEL_THRESHOLD_DOWN = -0.5   # return < -0.5%  -> 0 (DOWN)

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Lag-mimic diagnostic thresholds.
# If a model's accuracy is within this distance of the persistence baseline
# AND it agrees with "yesterday's direction" more than this fraction of the
# time, we flag it as a likely lag-mimic (Radfar's false-positive trap).
PERSISTENCE_ACC_CLOSE = 0.02      # within 2 percentage points
LAG_MIMIC_AGREEMENT_WARN = 0.90   # agrees with prev day > 90% of the time

# If a model predicts the same class for more than this fraction of rows it has
# effectively collapsed to the majority class and learned nothing useful.
CLASS_COLLAPSE_WARN = 0.90

# Calibration method. Sigmoid (Platt) is monotonic and robust on small
# validation sets, so it mainly fixes probabilities without thrashing labels.
CALIBRATION_METHOD = "sigmoid"


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
            # Fetch features with pagination using range. We pull the
            # continuous next-day return so we can build thresholded labels.
            response = (
                supabase.table("features_daily")
                .select(
                    "date, stock_id, return_1d, return_5d, return_20d, "
                    "volatility_5d, volatility_20d, volume_change_1d, "
                    "moving_average_5d, moving_average_20d, rsi_14, macd, macd_signal, "
                    "news_count, avg_sentiment_score, "
                    "target_next_day_return"
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

        # Ensure numeric types (Supabase numeric columns can arrive as strings)
        numeric_cols = RAW_FETCH_COLUMNS + SENTIMENT_FEATURES + [RETURN_COLUMN]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch features from database: {e}")
        sys.exit(1)


def derive_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace price-scale features (moving averages, MACD) with scale-free ratios
    so they are comparable across stocks at different price levels.

    A $20 and a $500 stock have wildly different absolute MAs/MACD; the ratios
    below capture the same trend/momentum shape independent of price level.
    """
    df = df.copy()

    ma20 = df["moving_average_20d"].astype(float)
    # Guard against divide-by-zero; non-positive MAs become NaN and are dropped
    # later in cleaning (real equity MAs are always positive).
    ma20 = ma20.where(ma20 > 0, np.nan)

    df["ma_5_vs_20"] = df["moving_average_5d"] / ma20 - 1.0
    df["macd_norm"] = df["macd"] / ma20
    df["macd_signal_norm"] = df["macd_signal"] / ma20
    df["macd_hist_norm"] = (df["macd"] - df["macd_signal"]) / ma20

    logger.info(f"Derived {len(DERIVED_FEATURES)} scale-free price features")
    return df


def fetch_market_features(min_date: str, max_date: str) -> pd.DataFrame | None:
    """
    Fetch market-wide context (SPY, QQQ, VIX) from Yahoo Finance and build
    per-date features aligned to the stock features (same-day, no lookahead).

    Returns a DataFrame with a 'date' column (YYYY-MM-DD) plus MARKET_FEATURES,
    or None if yfinance/network is unavailable (training then proceeds
    price-only).
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; skipping market features.")
        return None

    try:
        # Pad the start so 5-day returns are defined at the earliest stock date.
        start = (pd.to_datetime(min_date) - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
        end = (pd.to_datetime(max_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        def close_series(ticker: str) -> pd.Series | None:
            raw = yf.download(
                ticker, start=start, end=end, interval="1d",
                auto_adjust=False, actions=False, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            return raw[col].astype(float)

        spy = close_series(MARKET_TICKERS["SPY"])
        qqq = close_series(MARKET_TICKERS["QQQ"])
        vix = close_series(MARKET_TICKERS["VIX"])

        if spy is None or qqq is None or vix is None:
            logger.warning("Market data download empty; skipping market features.")
            return None

        market = pd.DataFrame(index=spy.index)
        market["mkt_spy_ret_1d"] = spy.pct_change(1) * 100
        market["mkt_spy_ret_5d"] = spy.pct_change(5) * 100
        market["mkt_qqq_ret_1d"] = qqq.pct_change(1) * 100
        market["mkt_vix"] = vix
        market["mkt_vix_chg_1d"] = vix.pct_change(1) * 100

        market = market.reset_index()
        market["date"] = pd.to_datetime(market.iloc[:, 0]).dt.strftime("%Y-%m-%d")
        market = market[["date"] + MARKET_FEATURES]

        logger.info(
            f"Fetched market features (SPY/QQQ/VIX) for {len(market)} trading days"
        )
        return market

    except Exception as e:
        logger.warning(f"Failed to fetch market features ({e}); proceeding price-only.")
        return None


def validate_columns(df: pd.DataFrame) -> None:
    """Check that all required raw columns exist."""
    required = RAW_FETCH_COLUMNS + [RETURN_COLUMN, "date", "stock_id", "ticker"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)

    logger.info(f"All {len(required)} required columns present")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing features or missing next-day return.
    """
    initial_count = len(df)

    # Need every feature and the continuous return to build a label.
    check_cols = FEATURE_COLUMNS + [RETURN_COLUMN]
    df_clean = df.dropna(subset=check_cols)

    removed = initial_count - len(df_clean)

    if removed > 0:
        logger.info(
            f"Removed {removed} rows with missing values "
            f"({removed / initial_count * 100:.1f}%)"
        )

    if len(df_clean) == 0:
        logger.error("No valid rows remaining after cleaning")
        sys.exit(1)

    logger.info(f"Rows after cleaning (missing values dropped): {len(df_clean)}")

    return df_clean.sort_values(["ticker", "date"]).reset_index(drop=True)


def apply_label_thresholding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a binary label from `target_next_day_return` using StockNet-style
    thresholding, and drop near-flat "noise" days.

      return >  LABEL_THRESHOLD_UP   -> 1 (UP)
      return <  LABEL_THRESHOLD_DOWN -> 0 (DOWN)
      otherwise (inside the band)    -> dropped as noise
    """
    before = len(df)

    ret = df[RETURN_COLUMN]
    df = df.copy()
    df[LABEL_COLUMN] = np.nan
    df.loc[ret > LABEL_THRESHOLD_UP, LABEL_COLUMN] = 1
    df.loc[ret < LABEL_THRESHOLD_DOWN, LABEL_COLUMN] = 0

    noise_count = int(df[LABEL_COLUMN].isna().sum())
    df = df.dropna(subset=[LABEL_COLUMN]).copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    n_up = int((df[LABEL_COLUMN] == 1).sum())
    n_down = int((df[LABEL_COLUMN] == 0).sum())
    final = len(df)
    up_ratio = round(n_up / final, 4) if final else 0.0

    logger.info(
        f"Label thresholding (UP > {LABEL_THRESHOLD_UP}%, "
        f"DOWN < {LABEL_THRESHOLD_DOWN}%):"
    )
    logger.info(f"  Rows removed as noise (flat band): {noise_count}")
    logger.info(f"  Final rows used for training/eval: {final}")
    logger.info(
        f"  Class balance: UP={n_up} ({up_ratio * 100:.1f}%), "
        f"DOWN={n_down} ({(1 - up_ratio) * 100:.1f}%)"
    )

    if final == 0:
        logger.error("No rows remaining after label thresholding")
        sys.exit(1)

    return df.reset_index(drop=True)


def add_previous_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `prev_direction`: the actual (thresholded) label of the previous
    available trading day for the same ticker. Used for both the persistence
    baseline and the lag-mimic diagnostic.

    The first row per ticker has no prior day and is left as NaN.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["prev_direction"] = df.groupby("ticker")[LABEL_COLUMN].shift(1)

    n_missing = int(df["prev_direction"].isna().sum())
    logger.info(
        f"Computed previous-day direction per ticker "
        f"({n_missing} rows have no prior day)"
    )
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split on UNIQUE DATES (not raw rows): oldest 70% of trading days
    -> train, next 15% -> validation, newest 15% -> test.

    Splitting by date keeps every ticker from the same day in the same split,
    which is the correct grouping for pooled cross-sectional data and avoids
    cutting a single day across the train/test boundary (suggestion #4).
    """
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    unique_dates = np.sort(df["date"].unique())
    n_dates = len(unique_dates)
    train_end = int(n_dates * TRAIN_RATIO)
    val_end = int(n_dates * (TRAIN_RATIO + VAL_RATIO))

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    train_df = df[df["date"].isin(train_dates)]
    val_df = df[df["date"].isin(val_dates)]
    test_df = df[df["date"].isin(test_dates)]

    logger.info("Time-based split (by unique date):")
    logger.info(
        f"  Training:   {len(train_df):>6} rows / {len(train_dates):>4} days "
        f"({TRAIN_RATIO * 100:.0f}%)"
    )
    logger.info(
        f"  Validation: {len(val_df):>6} rows / {len(val_dates):>4} days "
        f"({VAL_RATIO * 100:.0f}%)"
    )
    logger.info(
        f"  Test:       {len(test_df):>6} rows / {len(test_dates):>4} days "
        f"({TEST_RATIO * 100:.0f}%)"
    )

    if len(train_df) < 100:
        logger.warning(
            f"Training set is small ({len(train_df)} rows). "
            f"Model may not generalize well."
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and binary label vector y."""
    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].values.astype(int)
    return X, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the full metric set, including MCC. (AUC added separately.)"""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "auc": None,
    }


def positive_scores(model, X: np.ndarray):
    """Probability of the UP class, for ROC-AUC (falls back to decision_function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def compute_auc(y_true: np.ndarray, scores) -> float | None:
    """
    ROC-AUC from probability/ranking scores. Measures whether the model's
    *ranking* of UP-likelihood has signal, independent of the 0.5 threshold.
    AUC ~ 0.50 means no ranking signal (so no threshold tuning can help).
    Returns None for label-only predictors (baselines) or when undefined.
    """
    if scores is None:
        return None
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return round(float(roc_auc_score(y_true, scores)), 4)
    except Exception:
        return None


def log_metrics(name: str, split: str, m: dict) -> None:
    """Log a metric dict in a consistent format."""
    logger.info(f"\n{name} ({split}) Results:")
    logger.info(f"  Accuracy:  {m['accuracy']:.4f}")
    logger.info(f"  Precision: {m['precision']:.4f}")
    logger.info(f"  Recall:    {m['recall']:.4f}")
    logger.info(f"  F1 Score:  {m['f1']:.4f}")
    logger.info(f"  MCC:       {m['mcc']:.4f}")
    if m.get("auc") is not None:
        logger.info(f"  ROC-AUC:   {m['auc']:.4f}")


def lag_mimic_agreement(y_pred: np.ndarray, prev_direction: np.ndarray) -> float:
    """
    Fraction of predictions that match the previous trading day's actual
    direction. High values suggest the model is just echoing yesterday.
    Rows with no prior day (NaN) are excluded.
    """
    prev = np.asarray(prev_direction, dtype=float)
    mask = ~np.isnan(prev)
    if mask.sum() == 0:
        return float("nan")
    agree = np.mean(y_pred[mask] == prev[mask].astype(int))
    return round(float(agree), 4)


def check_lag_warning(
    name: str, model_acc: float, persistence_acc: float, lag_agree: float
) -> None:
    """Warn if a model looks like it is mimicking yesterday's direction."""
    if np.isnan(lag_agree):
        return
    logger.info(f"  Lag mimic agreement: {lag_agree * 100:.0f}%")
    close_to_persistence = abs(model_acc - persistence_acc) < PERSISTENCE_ACC_CLOSE
    if close_to_persistence and lag_agree > LAG_MIMIC_AGREEMENT_WARN:
        logger.warning(
            f"WARNING: {name} may be copying yesterday's direction rather "
            f"than learning useful predictive structure."
        )


def check_class_collapse(name: str, y_pred: np.ndarray) -> dict:
    """
    Detect a model that predicts (almost) one class for everything. This is the
    degenerate "always UP/DOWN" failure: it can post a deceptively high F1 by
    riding the class imbalance while having no real predictive skill (MCC ~ 0).
    Returns the dominant-class share so it can be recorded in metadata.
    """
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_pred)
    up_share = float(np.mean(y_pred == 1)) if n else 0.0
    dominant_class = 1 if up_share >= 0.5 else 0
    dominant_share = max(up_share, 1 - up_share)

    logger.info(
        f"  Predicted class balance: UP={up_share * 100:.0f}% / "
        f"DOWN={(1 - up_share) * 100:.0f}%"
    )
    if dominant_share > CLASS_COLLAPSE_WARN:
        label = "UP" if dominant_class == 1 else "DOWN"
        logger.warning(
            f"WARNING: {name} collapsed to the majority class "
            f"(predicts {label} {dominant_share * 100:.0f}% of the time) - it is "
            f"likely riding class imbalance, not learning. Trust MCC over F1 here."
        )

    return {
        "dominant_class": dominant_class,
        "dominant_class_share": round(dominant_share, 4),
        "collapsed": bool(dominant_share > CLASS_COLLAPSE_WARN),
    }


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------

def baseline_predictions(
    split_df: pd.DataFrame,
    p_up: float,
    train_majority: int,
    rng: np.random.Generator,
) -> dict:
    """
    Build predictions for the three naive baselines on a split:
      A. Always-UP   - always predicts 1
      B. Persistence - predicts previous trading day's actual direction
      C. Random      - random 0/1 using the training class balance
    """
    n = len(split_df)

    always_up = np.ones(n, dtype=int)

    persistence = (
        split_df["prev_direction"].fillna(train_majority).astype(int).values
    )

    random_pred = rng.choice([0, 1], size=n, p=[1 - p_up, p_up]).astype(int)

    return {
        "Always-UP": always_up,
        "Persistence": persistence,
        "Random": random_pred,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Train a logistic regression model with feature scaling."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_SEED
        )),
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a random forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def calibrate_model(fitted_estimator, X_val: np.ndarray, y_val: np.ndarray):
    """
    Wrap an already-fitted estimator in CalibratedClassifierCV using the
    validation set (cv='prefit'). Returns (estimator, calibrated_flag).
    Falls back to the uncalibrated estimator if calibration fails.
    """
    # Need both classes present in the validation set to calibrate.
    if len(np.unique(y_val)) < 2:
        logger.warning(
            "Validation set has a single class; skipping calibration."
        )
        return fitted_estimator, False

    try:
        # sklearn >= 1.6 replaced cv="prefit" with FrozenEstimator. Support
        # both so calibration works across the version range in requirements.
        try:
            from sklearn.frozen import FrozenEstimator
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(fitted_estimator),
                method=CALIBRATION_METHOD,
            )
        except ImportError:
            calibrated = CalibratedClassifierCV(
                estimator=fitted_estimator,
                method=CALIBRATION_METHOD,
                cv="prefit",
            )
        calibrated.fit(X_val, y_val)
        return calibrated, True
    except Exception as e:
        logger.warning(f"Calibration failed ({e}); using uncalibrated model.")
        return fitted_estimator, False


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(rows: list[dict]) -> None:
    """
    Print a comparison table.
    Each row: {"model": str, "split": str, "metrics": dict}
    """
    header = (
        f"{'Model':<22} | {'Split':<11} | {'Accuracy':>8} | "
        f"{'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'MCC':>7} | {'AUC':>7}"
    )
    sep = "-" * len(header)

    logger.info("\n" + "=" * len(header))
    logger.info("Comparison Table")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info(sep)
    for r in rows:
        m = r["metrics"]
        auc = m.get("auc")
        auc_str = f"{auc:.4f}" if auc is not None else "n/a"
        logger.info(
            f"{r['model']:<22} | {r['split']:<11} | "
            f"{m['accuracy']:>8.4f} | {m['precision']:>9.4f} | "
            f"{m['recall']:>7.4f} | {m['f1']:>6.4f} | {m['mcc']:>7.4f} | {auc_str:>7}"
        )
    logger.info("=" * len(header))


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
    rows_loaded = len(df)

    # Validate raw columns
    validate_columns(df)

    # Normalize price-scale features into scale-free ratios (suggestion #1)
    df = derive_normalized_features(df)

    # Stage 2 sentiment: fill missing (no news) with neutral 0 so those rows are
    # kept rather than dropped during cleaning.
    for col in SENTIMENT_FEATURES:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
    sentiment_coverage = float((df["news_count"] > 0).mean())

    # Add market-wide context joined by date (suggestion #6). Degrades
    # gracefully to price-only if the market data can't be fetched.
    global FEATURE_COLUMNS
    active_features = list(SCALE_FREE_FEATURES + DERIVED_FEATURES)
    market_df = fetch_market_features(df["date"].min(), df["date"].max())
    market_features_used = False
    if market_df is not None:
        df = df.merge(market_df, on="date", how="left")
        active_features += MARKET_FEATURES
        market_features_used = True
        logger.info(f"Added {len(MARKET_FEATURES)} market-wide features")
    else:
        logger.warning("Proceeding with price-only features (no market context)")
    active_features += SENTIMENT_FEATURES
    logger.info(
        f"Added {len(SENTIMENT_FEATURES)} sentiment features "
        f"(news present on {sentiment_coverage * 100:.1f}% of rows)"
    )
    FEATURE_COLUMNS = active_features
    logger.info(f"Active feature set: {len(FEATURE_COLUMNS)} columns")

    # Clean (drop missing features / returns)
    df_clean = clean_data(df)
    rows_after_cleaning = len(df_clean)

    # StockNet-style label thresholding (drop noise band)
    df_labeled = apply_label_thresholding(df_clean)
    rows_after_thresholding = len(df_labeled)

    n_up = int((df_labeled[LABEL_COLUMN] == 1).sum())
    n_down = int((df_labeled[LABEL_COLUMN] == 0).sum())
    class_balance = {
        "up": n_up,
        "down": n_down,
        "up_ratio": round(n_up / rows_after_thresholding, 4),
    }

    # Previous-day direction for persistence baseline + lag-mimic diagnostic
    df_labeled = add_previous_direction(df_labeled)

    # Time-based split
    train_df, val_df, test_df = split_data(df_labeled)

    # Prepare matrices
    X_train, y_train = prepare_matrices(train_df)
    X_val, y_val = prepare_matrices(val_df)
    X_test, y_test = prepare_matrices(test_df)

    logger.info("\nFeature matrix shapes:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_val:   {X_val.shape}")
    logger.info(f"  X_test:  {X_test.shape}")

    # Training class balance drives the Random baseline and persistence fill.
    p_up = float(y_train.mean()) if len(y_train) else 0.5
    train_majority = int(round(p_up))
    rng = np.random.default_rng(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Naive baselines
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 60)
    logger.info("Naive Baselines")
    logger.info("-" * 60)

    val_baseline_preds = baseline_predictions(val_df, p_up, train_majority, rng)
    test_baseline_preds = baseline_predictions(test_df, p_up, train_majority, rng)

    baseline_metrics = {"validation": {}, "test": {}}
    for name, preds in val_baseline_preds.items():
        m = compute_metrics(y_val, preds)
        baseline_metrics["validation"][name] = m
        log_metrics(name, "Validation", m)
    for name, preds in test_baseline_preds.items():
        baseline_metrics["test"][name] = compute_metrics(y_test, preds)

    persistence_val_acc = baseline_metrics["validation"]["Persistence"]["accuracy"]
    persistence_test_acc = baseline_metrics["test"]["Persistence"]["accuracy"]

    val_prev = val_df["prev_direction"].values
    test_prev = test_df["prev_direction"].values

    # ------------------------------------------------------------------
    # Train + calibrate models
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 60)
    logger.info("Training Baseline Models")
    logger.info("-" * 60)

    models = {}  # name -> {"estimator", "calibrated", "val_metrics", "lag"}

    # Logistic Regression
    logger.info("\nTraining Logistic Regression...")
    try:
        lr_raw = train_logistic_regression(X_train, y_train)
        lr_model, lr_calibrated = calibrate_model(lr_raw, X_val, y_val)
        lr_val_pred = lr_model.predict(X_val)
        lr_val_metrics = compute_metrics(y_val, lr_val_pred)
        lr_val_metrics["auc"] = compute_auc(y_val, positive_scores(lr_model, X_val))
        log_metrics("Logistic Regression", "Validation", lr_val_metrics)
        lr_lag = lag_mimic_agreement(lr_val_pred, val_prev)
        check_lag_warning(
            "Logistic Regression", lr_val_metrics["accuracy"],
            persistence_val_acc, lr_lag,
        )
        lr_collapse = check_class_collapse("Logistic Regression", lr_val_pred)
        models["Logistic Regression"] = {
            "estimator": lr_model,
            "calibrated": lr_calibrated,
            "val_metrics": lr_val_metrics,
            "lag": lr_lag,
            "collapse": lr_collapse,
        }
    except Exception as e:
        logger.error(f"Logistic Regression training failed: {e}")

    # Random Forest
    logger.info("\nTraining Random Forest...")
    try:
        rf_raw = train_random_forest(X_train, y_train)
        rf_model, rf_calibrated = calibrate_model(rf_raw, X_val, y_val)
        rf_val_pred = rf_model.predict(X_val)
        rf_val_metrics = compute_metrics(y_val, rf_val_pred)
        rf_val_metrics["auc"] = compute_auc(y_val, positive_scores(rf_model, X_val))
        log_metrics("Random Forest", "Validation", rf_val_metrics)
        rf_lag = lag_mimic_agreement(rf_val_pred, val_prev)
        check_lag_warning(
            "Random Forest", rf_val_metrics["accuracy"],
            persistence_val_acc, rf_lag,
        )
        rf_collapse = check_class_collapse("Random Forest", rf_val_pred)
        models["Random Forest"] = {
            "estimator": rf_model,
            "calibrated": rf_calibrated,
            "val_metrics": rf_val_metrics,
            "lag": rf_lag,
            "collapse": rf_collapse,
        }
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")

    if not models:
        logger.error("All models failed to train")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Model selection: validation MCC first, F1 as tie-breaker
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 60)
    logger.info("Model Selection (by validation MCC, F1 tie-breaker)")
    logger.info("-" * 60)

    best_name = max(
        models,
        key=lambda n: (
            models[n]["val_metrics"]["mcc"],
            models[n]["val_metrics"]["f1"],
        ),
    )
    best = models[best_name]
    best_model = best["estimator"]
    best_val_metrics = best["val_metrics"]
    best_calibrated = best["calibrated"]

    logger.info(
        f"\nBest model: {best_name} "
        f"(val MCC = {best_val_metrics['mcc']:.4f}, "
        f"val F1 = {best_val_metrics['f1']:.4f}, "
        f"calibrated = {best_calibrated})"
    )

    # ------------------------------------------------------------------
    # Final evaluation on the held-out test set
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 60)
    logger.info(f"Final Evaluation on Test Set: {best_name}")
    logger.info("-" * 60)

    test_pred = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_pred)
    test_metrics["auc"] = compute_auc(y_test, positive_scores(best_model, X_test))
    log_metrics(best_name, "Test", test_metrics)
    logger.info(f"  Confusion Matrix:\n{confusion_matrix(y_test, test_pred)}")

    best_test_lag = lag_mimic_agreement(test_pred, test_prev)
    check_lag_warning(
        best_name, test_metrics["accuracy"], persistence_test_acc, best_test_lag
    )
    best_test_collapse = check_class_collapse(best_name, test_pred)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    table_rows = [
        {"model": "Always-UP", "split": "Validation",
         "metrics": baseline_metrics["validation"]["Always-UP"]},
        {"model": "Persistence", "split": "Validation",
         "metrics": baseline_metrics["validation"]["Persistence"]},
        {"model": "Random", "split": "Validation",
         "metrics": baseline_metrics["validation"]["Random"]},
    ]
    if "Logistic Regression" in models:
        table_rows.append({
            "model": "Logistic Regression", "split": "Validation",
            "metrics": models["Logistic Regression"]["val_metrics"],
        })
    if "Random Forest" in models:
        table_rows.append({
            "model": "Random Forest", "split": "Validation",
            "metrics": models["Random Forest"]["val_metrics"],
        })
    # Test split: naive baselines + the selected best model, so we can see
    # directly whether the model beats the baselines out of sample.
    table_rows.append({"model": "Always-UP", "split": "Test",
                       "metrics": baseline_metrics["test"]["Always-UP"]})
    table_rows.append({"model": "Persistence", "split": "Test",
                       "metrics": baseline_metrics["test"]["Persistence"]})
    table_rows.append({"model": "Random", "split": "Test",
                       "metrics": baseline_metrics["test"]["Random"]})
    table_rows.append({
        "model": f"Best ({best_name})", "split": "Test",
        "metrics": test_metrics,
    })
    print_comparison_table(table_rows)

    # ------------------------------------------------------------------
    # Metadata + save
    # ------------------------------------------------------------------
    lag_mimic_diagnostic = {
        "warning_threshold_agreement": LAG_MIMIC_AGREEMENT_WARN,
        "persistence_closeness_threshold": PERSISTENCE_ACC_CLOSE,
        "best_model_test_agreement": best_test_lag,
        "persistence_test_accuracy": persistence_test_acc,
    }
    for name, info in models.items():
        key = name.lower().replace(" ", "_") + "_validation_agreement"
        lag_mimic_diagnostic[key] = info["lag"]

    class_collapse_diagnostic = {
        "warning_threshold": CLASS_COLLAPSE_WARN,
        "best_model_test": best_test_collapse,
    }
    for name, info in models.items():
        key = name.lower().replace(" ", "_") + "_validation"
        class_collapse_diagnostic[key] = info["collapse"]

    metadata = {
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "label_threshold_up": LABEL_THRESHOLD_UP,
        "label_threshold_down": LABEL_THRESHOLD_DOWN,
        "rows_loaded": rows_loaded,
        "rows_after_cleaning": rows_after_cleaning,
        "rows_after_thresholding": rows_after_thresholding,
        "class_balance": class_balance,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "market_features_used": market_features_used,
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "lag_mimic_diagnostic": lag_mimic_diagnostic,
        "class_collapse_diagnostic": class_collapse_diagnostic,
        "roc_auc": {
            "best_validation": best_val_metrics.get("auc"),
            "best_test": test_metrics.get("auc"),
            "note": "AUC ~ 0.50 means no ranking signal (threshold tuning cannot help)",
        },
        "calibrated": best_calibrated,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    save_model(best_model, metadata)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Best model: {best_name} (calibrated = {best_calibrated})")
    logger.info(f"Test MCC: {test_metrics['mcc']:.4f}")
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
