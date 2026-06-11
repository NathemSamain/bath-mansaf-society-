#!/usr/bin/env python3
"""
Historical daily price ingestion from Alpha Vantage to Supabase.

Usage:
    python scripts/ingest_prices_daily.py

Environment variables required:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    ALPHA_VANTAGE_API_KEY
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional

import requests
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
RATE_LIMIT_DELAY = 12  # Alpha Vantage free tier: ~5 calls per minute


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


def fetch_alpha_vantage_prices(ticker: str, api_key: str) -> Optional[dict]:
    """
    Fetch daily adjusted time series from Alpha Vantage.

    Returns dict of {date: {open, high, low, close, adjusted_close, volume}}
    or None on error.
    """
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
    }

    try:
        logger.info(f"Fetching prices for {ticker} from Alpha Vantage...")
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for API error messages
        if "Error Message" in data:
            logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
            return None

        if "Note" in data:
            logger.warning(f"Alpha Vantage rate limit note: {data['Note']}")
            return None

        if "Information" in data:
            logger.warning(f"Alpha Vantage info message: {data['Information']}")
            return None

        time_series = data.get("Time Series (Daily)")
        if not time_series:
            logger.warning(f"No time series data returned for {ticker}")
            return None

        logger.info(f"Received {len(time_series)} daily records for {ticker}")
        return time_series

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {ticker} from Alpha Vantage")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching {ticker}: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Failed to parse Alpha Vantage response for {ticker}: {e}")
        return None


def parse_price_data(
    stock_id: str, ticker: str, time_series: dict
) -> list[dict]:
    """Parse Alpha Vantage time series into database records."""
    records = []

    for date_str, values in time_series.items():
        try:
            record = {
                "stock_id": stock_id,
                "date": date_str,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "adjusted_close": float(values.get("5. adjusted close", 0)),
                "volume": int(values.get("6. volume", 0)),
                "source": "Alpha Vantage",
            }
            records.append(record)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Skipping invalid record for {ticker} on {date_str}: {e}")
            continue

    return records


def upsert_prices(supabase: Client, records: list[dict], ticker: str) -> int:
    """Upsert price records into database using batch inserts."""
    if not records:
        logger.warning(f"No records to upsert for {ticker}")
        return 0

    total_upserted = 0
    batch_size = 100

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            response = supabase.table("prices_daily").upsert(
                batch, on_conflict="stock_id,date"
            ).execute()
            total_upserted += len(batch)
            logger.info(f"  Upserted batch {i // batch_size + 1}: {len(batch)} records")
        except Exception as e:
            logger.error(f"Failed to upsert batch for {ticker}: {e}")
            continue

    return total_upserted


def ingest_prices():
    """Main ingestion workflow."""
    logger.info("=" * 60)
    logger.info("Starting daily price ingestion")
    logger.info("=" * 60)

    # Validate environment
    api_key = get_env_var("ALPHA_VANTAGE_API_KEY")

    # Initialize clients
    supabase = init_supabase()

    # Fetch active tickers
    tickers = fetch_active_tickers(supabase)
    if not tickers:
        logger.error("No active tickers to process")
        sys.exit(1)

    # Process each ticker
    total_inserted = 0
    success_count = 0

    for i, stock in enumerate(tickers, 1):
        ticker = stock["ticker"]
        stock_id = stock["id"]

        logger.info(f"\n[{i}/{len(tickers)}] Processing {ticker}...")

        # Fetch from Alpha Vantage
        time_series = fetch_alpha_vantage_prices(ticker, api_key)

        if not time_series:
            logger.warning(f"Skipping {ticker} due to API error or empty response")
            continue

        # Parse data
        records = parse_price_data(stock_id, ticker, time_series)

        if not records:
            logger.warning(f"No valid records parsed for {ticker}")
            continue

        # Upsert to database
        upserted = upsert_prices(supabase, records, ticker)
        total_inserted += upserted
        success_count += 1

        logger.info(f"  ✓ {ticker}: {upserted} records upserted")

        # Rate limit delay (skip after last ticker)
        if i < len(tickers):
            logger.info(f"  Waiting {RATE_LIMIT_DELAY}s to respect rate limit...")
            time.sleep(RATE_LIMIT_DELAY)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion complete")
    logger.info(f"Tickers processed: {success_count}/{len(tickers)}")
    logger.info(f"Total records upserted: {total_inserted}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        ingest_prices()
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
