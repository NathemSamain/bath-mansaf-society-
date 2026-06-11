#!/usr/bin/env python3
"""
Feature engineering pipeline for daily stock prices.

Reads historical prices from Supabase, calculates technical indicators and
ML targets, and stores in features_daily table.

Usage:
    python scripts/build_features_daily.py

Environment variables required:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import sys
import logging
from typing import Optional
from datetime import date

import numpy as np
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_env_var(name: str) -> str:
    """Get environment variable or raise error."""
    value = os.environ.get(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


def init_supabase() -> Client:
    """Initialize Supabase client with service role key."""
    url = get_env_var("SUPABASE_URL")
    key = get_env_var("SUPABASE_SERVICE_ROLE_KEY")
    client = create_client(url, key)
    logger.info("Connected to Supabase")
    return client


def fetch_active_tickers(supabase: Client) -> list[dict]:
    """Fetch all active stock tickers from database."""
    try:
        response = supabase.table("stocks").select("id, ticker").eq("is_active", True).execute()
        if not response.data:
            logger.warning("No active tickers found in stocks table")
            return []
        logger.info(f"Found {len(response.data)} active tickers")
        return response.data
    except Exception as e:
        logger.error(f"Failed to fetch tickers from database: {e}")
        sys.exit(1)


def fetch_prices_for_stock(supabase: Client, stock_id: str, ticker: str) -> list[dict]:
    """
    Fetch all historical prices for a stock, ordered by date ascending.
    Returns list of dicts with date, close, volume fields.
    """
    try:
        response = (
            supabase.table("prices_daily")
            .select("date, close, volume")
            .eq("stock_id", stock_id)
            .order("date", desc=False)
            .execute()
        )

        if not response.data:
            logger.warning(f"No price data found for {ticker}")
            return []

        return response.data

    except Exception as e:
        logger.error(f"Failed to fetch prices for {ticker}: {e}")
        return []


def calculate_returns(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate 1-day, 5-day, and 20-day returns.

    Returns are percentage change in price.
    """
    n = len(prices)

    # 1-day return
    return_1d = np.full(n, np.nan)
    return_1d[1:] = ((prices[1:] - prices[:-1]) / prices[:-1]) * 100

    # 5-day return
    return_5d = np.full(n, np.nan)
    return_5d[5:] = ((prices[5:] - prices[:-5]) / prices[:-5]) * 100

    # 20-day return
    return_20d = np.full(n, np.nan)
    return_20d[20:] = ((prices[20:] - prices[:-20]) / prices[:-20]) * 100

    return return_1d, return_5d, return_20d


def calculate_volatility(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling volatility (standard deviation of returns) over window.
    """
    n = len(prices)
    volatility = np.full(n, np.nan)

    # Calculate daily returns
    daily_returns = np.full(n, np.nan)
    daily_returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

    # Rolling std
    for i in range(window, n):
        window_returns = daily_returns[i - window + 1 : i + 1]
        volatility[i] = np.std(window_returns, ddof=1) * 100  # As percentage

    return volatility


def calculate_volume_change(volumes: np.ndarray) -> np.ndarray:
    """Calculate 1-day volume change percentage."""
    n = len(volumes)
    volume_change = np.full(n, np.nan)

    for i in range(1, n):
        if volumes[i - 1] > 0:
            volume_change[i] = ((volumes[i] - volumes[i - 1]) / volumes[i - 1]) * 100

    return volume_change


def calculate_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate simple moving average over window."""
    n = len(prices)
    ma = np.full(n, np.nan)

    for i in range(window - 1, n):
        ma[i] = np.mean(prices[i - window + 1 : i + 1])

    return ma


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    n = len(prices)
    rsi = np.full(n, np.nan)

    if n < period + 1:
        return rsi

    # Calculate price changes
    deltas = np.diff(prices)

    # Gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # First average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Smoothed averages
    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_macd(
    prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence) and signal line.

    MACD = EMA(fast) - EMA(slow)
    Signal = EMA of MACD over signal period
    """
    n = len(prices)
    macd = np.full(n, np.nan)
    signal_line = np.full(n, np.nan)

    if n < slow:
        return macd, signal_line

    # Calculate EMAs using smoothing factor alpha = 2 / (period + 1)
    def ema(data, period):
        alpha = 2.0 / (period + 1)
        ema_vals = np.full(len(data), np.nan)
        ema_vals[period - 1] = np.mean(data[:period])

        for i in range(period, len(data)):
            ema_vals[i] = alpha * data[i] + (1 - alpha) * ema_vals[i - 1]

        return ema_vals

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)

    # MACD line
    valid_mask = ~np.isnan(ema_fast) & ~np.isnan(ema_slow)
    macd[valid_mask] = ema_fast[valid_mask] - ema_slow[valid_mask]

    # Signal line (EMA of MACD)
    alpha_signal = 2.0 / (signal + 1)

    # Find first valid MACD index
    first_valid = slow - 1

    # Initialize signal as SMA of first 'signal' MACD values
    if n >= first_valid + signal:
        signal_line[first_valid + signal - 1] = np.mean(
            macd[first_valid : first_valid + signal]
        )

        for i in range(first_valid + signal, n):
            if not np.isnan(macd[i - 1]):
                signal_line[i] = (
                    alpha_signal * macd[i] + (1 - alpha_signal) * signal_line[i - 1]
                )

    return macd, signal_line


def calculate_targets(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate supervised learning targets.

    target_next_day_direction: 1 if next day close > current close, else 0
    target_next_day_return: percentage return from current to next day close
    """
    n = len(prices)
    direction = np.full(n, np.nan, dtype=float)
    ret = np.full(n, np.nan, dtype=float)

    for i in range(n - 1):
        if prices[i + 1] > prices[i]:
            direction[i] = 1.0
        else:
            direction[i] = 0.0
        ret[i] = ((prices[i + 1] - prices[i]) / prices[i]) * 100

    return direction, ret


def build_features_for_stock(
    stock_id: str, ticker: str, prices_data: list[dict]
) -> list[dict]:
    """
    Build feature rows for a single stock from price history.

    Returns list of dicts ready for upsert.
    """
    if len(prices_data) < 21:
        logger.warning(
            f"Insufficient price history for {ticker}: {len(prices_data)} rows (need 21+)"
        )
        return []

    # Extract arrays
    dates = [row["date"] for row in prices_data]
    closes = np.array([float(row["close"]) for row in prices_data])
    volumes = np.array([float(row["volume"]) for row in prices_data])

    n = len(prices_data)
    logger.info(f"Building features for {ticker}: {n} price rows")

    # Calculate all features
    return_1d, return_5d, return_20d = calculate_returns(closes)
    volatility_5d = calculate_volatility(closes, 5)
    volatility_20d = calculate_volatility(closes, 20)
    volume_change_1d = calculate_volume_change(volumes)
    ma_5d = calculate_moving_average(closes, 5)
    ma_20d = calculate_moving_average(closes, 20)
    rsi_14 = calculate_rsi(closes, 14)
    macd, macd_signal = calculate_macd(closes)
    target_direction, target_return = calculate_targets(closes)

    # Build feature records
    records = []

    for i in range(n):
        # Only create record if we have enough data for core features
        # Require at least: return_1d, volatility_5d, ma_5d, rsi_14
        if i < 14:  # First valid RSI is at index 14
            continue

        record = {
            "stock_id": stock_id,
            "date": dates[i],
            "return_1d": round(return_1d[i], 6) if not np.isnan(return_1d[i]) else None,
            "return_5d": round(return_5d[i], 6) if not np.isnan(return_5d[i]) else None,
            "return_20d": round(return_20d[i], 6) if not np.isnan(return_20d[i]) else None,
            "volatility_5d": round(volatility_5d[i], 6) if not np.isnan(volatility_5d[i]) else None,
            "volatility_20d": round(volatility_20d[i], 6) if not np.isnan(volatility_20d[i]) else None,
            "volume_change_1d": round(volume_change_1d[i], 6) if not np.isnan(volume_change_1d[i]) else None,
            "moving_average_5d": round(ma_5d[i], 6) if not np.isnan(ma_5d[i]) else None,
            "moving_average_20d": round(ma_20d[i], 6) if not np.isnan(ma_20d[i]) else None,
            "rsi_14": round(rsi_14[i], 4) if not np.isnan(rsi_14[i]) else None,
            "macd": round(macd[i], 6) if not np.isnan(macd[i]) else None,
            "macd_signal": round(macd_signal[i], 6) if not np.isnan(macd_signal[i]) else None,
            "news_count": 0,  # Placeholder for future news integration
            "avg_positive_sentiment": None,
            "avg_negative_sentiment": None,
            "avg_neutral_sentiment": None,
            "avg_sentiment_score": None,
            "target_next_day_direction": int(target_direction[i]) if not np.isnan(target_direction[i]) else None,
            "target_next_day_return": round(target_return[i], 6) if not np.isnan(target_return[i]) else None,
        }
        records.append(record)

    logger.info(f"  Generated {len(records)} feature rows for {ticker}")
    return records


def upsert_features(supabase: Client, records: list[dict], ticker: str) -> int:
    """Upsert feature records into database using batch inserts."""
    if not records:
        logger.warning(f"No feature records to upsert for {ticker}")
        return 0

    total_upserted = 0
    batch_size = 100

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            response = (
                supabase.table("features_daily")
                .upsert(batch, on_conflict="stock_id,date")
                .execute()
            )
            total_upserted += len(batch)
            logger.info(f"  Upserted batch {i // batch_size + 1}: {len(batch)} records")
        except Exception as e:
            logger.error(f"Failed to upsert feature batch for {ticker}: {e}")
            continue

    return total_upserted


def build_features():
    """Main feature engineering workflow."""
    logger.info("=" * 60)
    logger.info("Starting daily feature engineering pipeline")
    logger.info("=" * 60)

    # Initialize client
    supabase = init_supabase()

    # Fetch active tickers
    tickers = fetch_active_tickers(supabase)
    if not tickers:
        logger.error("No active tickers to process")
        sys.exit(1)

    # Process each stock
    total_features = 0
    success_count = 0

    for i, stock in enumerate(tickers, 1):
        ticker = stock["ticker"]
        stock_id = stock["id"]

        logger.info(f"\n[{i}/{len(tickers)}] Processing {ticker}...")

        # Fetch historical prices
        prices_data = fetch_prices_for_stock(supabase, stock_id, ticker)

        if not prices_data:
            logger.warning(f"Skipping {ticker}: no price data")
            continue

        # Build features
        feature_records = build_features_for_stock(stock_id, ticker, prices_data)

        if not feature_records:
            logger.warning(f"Skipping {ticker}: insufficient data for features")
            continue

        # Upsert to database
        upserted = upsert_features(supabase, feature_records, ticker)
        total_features += upserted
        success_count += 1

        logger.info(f"  ✓ {ticker}: {upserted} feature rows upserted")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Feature engineering complete")
    logger.info(f"Stocks processed: {success_count}/{len(tickers)}")
    logger.info(f"Total feature rows: {total_features}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        build_features()
    except KeyboardInterrupt:
        logger.info("\nFeature engineering interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
