#!/usr/bin/env python3
"""
Historical daily price ingestion from Yahoo Finance (yfinance) to Supabase.

This is a free, no-API-key alternative to `ingest_prices_daily.py` (Alpha
Vantage). It pulls REAL daily OHLCV for every active ticker in the `stocks`
table and REPLACES any existing rows for those tickers (e.g. synthetic
"generated" seed data) with real prices tagged `source = 'yfinance'`.

The `prices_daily` schema is unchanged:
    stock_id, date, open, high, low, close, adjusted_close, volume, source

Usage:
    python scripts/ingest_prices_yfinance.py

Optional environment overrides (otherwise sensible defaults are used):
    YF_PERIOD       e.g. "5y", "10y", "max"   (default "5y")
    YF_START_DATE   e.g. "2015-01-01"         (overrides YF_PERIOD if set)

Required environment variables (read from .env file):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import sys
import time
import logging

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCE_LABEL = "yfinance"
DEFAULT_PERIOD = "5y"      # how much history to pull when no start date given
UPSERT_BATCH_SIZE = 500    # rows per Supabase request
INTER_TICKER_DELAY = 1.0   # polite pause between tickers (seconds)


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


def fetch_active_tickers(supabase: Client) -> list[dict]:
    """Fetch all active stock tickers from database."""
    try:
        response = (
            supabase.table("stocks")
            .select("id, ticker")
            .eq("is_active", True)
            .execute()
        )
        if not response.data:
            logger.warning("No active tickers found in stocks table")
            return []
        logger.info(f"Found {len(response.data)} active tickers")
        return response.data
    except Exception as e:
        logger.error(f"Failed to fetch tickers from database: {e}")
        sys.exit(1)


def download_prices(ticker: str) -> pd.DataFrame | None:
    """
    Download daily OHLCV history for a ticker from Yahoo Finance.

    Returns a DataFrame with single-level columns
    (Open, High, Low, Close, Adj Close, Volume) indexed by date, or None.
    """
    start_date = os.environ.get("YF_START_DATE")
    period = os.environ.get("YF_PERIOD", DEFAULT_PERIOD)

    try:
        logger.info(f"Downloading {ticker} from Yahoo Finance...")
        kwargs = dict(
            interval="1d",
            auto_adjust=False,   # keep raw OHLC + a separate 'Adj Close'
            actions=False,
            progress=False,
            threads=False,
        )
        if start_date:
            df = yf.download(ticker, start=start_date, **kwargs)
        else:
            df = yf.download(ticker, period=period, **kwargs)

        if df is None or df.empty:
            logger.warning(f"No data returned for {ticker}")
            return None

        # yfinance may return MultiIndex columns (field, ticker). Flatten to field.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        logger.info(f"Received {len(df)} daily records for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Failed to download {ticker}: {e}")
        return None


def parse_price_data(stock_id: str, ticker: str, df: pd.DataFrame) -> list[dict]:
    """Parse a yfinance DataFrame into prices_daily records."""
    records = []

    # 'Adj Close' is preferred for adjusted_close; fall back to 'Close'.
    has_adj = "Adj Close" in df.columns

    for idx, row in df.iterrows():
        try:
            close = row["Close"]
            # Skip rows with no valid close (occasional NaN from the source).
            if pd.isna(close):
                continue

            adj = row["Adj Close"] if has_adj and not pd.isna(row["Adj Close"]) else close
            volume = row.get("Volume", 0)

            record = {
                "stock_id": stock_id,
                "date": pd.Timestamp(idx).strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(close),
                "adjusted_close": float(adj),
                "volume": int(volume) if not pd.isna(volume) else 0,
                "source": SOURCE_LABEL,
            }
            records.append(record)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Skipping invalid record for {ticker} on {idx}: {e}")
            continue

    return records


def delete_existing_prices(supabase: Client, stock_id: str, ticker: str) -> None:
    """
    Remove all existing price rows for a ticker so real data fully replaces any
    prior (e.g. synthetic) rows, including stale dates the new feed won't cover.
    """
    try:
        supabase.table("prices_daily").delete().eq("stock_id", stock_id).execute()
        logger.info(f"  Cleared existing prices_daily rows for {ticker}")
    except Exception as e:
        logger.error(f"Failed to clear existing prices for {ticker}: {e}")


def upsert_prices(supabase: Client, records: list[dict], ticker: str) -> int:
    """Upsert price records into database using batch inserts."""
    if not records:
        logger.warning(f"No records to upsert for {ticker}")
        return 0

    total_upserted = 0

    for i in range(0, len(records), UPSERT_BATCH_SIZE):
        batch = records[i : i + UPSERT_BATCH_SIZE]
        try:
            supabase.table("prices_daily").upsert(
                batch, on_conflict="stock_id,date"
            ).execute()
            total_upserted += len(batch)
            logger.info(
                f"  Upserted batch {i // UPSERT_BATCH_SIZE + 1}: {len(batch)} records"
            )
        except Exception as e:
            logger.error(f"Failed to upsert batch for {ticker}: {e}")
            continue

    return total_upserted


def ingest_prices():
    """Main ingestion workflow."""
    logger.info("=" * 60)
    logger.info("Starting yfinance price ingestion (real OHLCV)")
    logger.info("=" * 60)

    supabase = init_supabase()

    tickers = fetch_active_tickers(supabase)
    if not tickers:
        logger.error("No active tickers to process")
        sys.exit(1)

    total_inserted = 0
    success_count = 0

    for i, stock in enumerate(tickers, 1):
        ticker = stock["ticker"]
        stock_id = stock["id"]

        logger.info(f"\n[{i}/{len(tickers)}] Processing {ticker}...")

        df = download_prices(ticker)
        if df is None:
            logger.warning(f"Skipping {ticker} due to download error or empty response")
            continue

        records = parse_price_data(stock_id, ticker, df)
        if not records:
            logger.warning(f"No valid records parsed for {ticker}")
            continue

        # Replace: clear old rows (synthetic or otherwise) then insert real data.
        delete_existing_prices(supabase, stock_id, ticker)
        upserted = upsert_prices(supabase, records, ticker)
        total_inserted += upserted
        success_count += 1

        logger.info(f"  OK {ticker}: {upserted} real records ingested")

        if i < len(tickers):
            time.sleep(INTER_TICKER_DELAY)

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion complete")
    logger.info(f"Tickers processed: {success_count}/{len(tickers)}")
    logger.info(f"Total records ingested: {total_inserted}")
    logger.info(f"Source label: {SOURCE_LABEL}")
    logger.info("=" * 60)
    logger.info("Next: rerun scripts/build_features_daily.py, then train_price_model.py")


if __name__ == "__main__":
    try:
        ingest_prices()
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
