#!/usr/bin/env python3
"""
Aggregate stored news sentiment into the `features_daily` sentiment columns,
using a leakage-safe, StockNet-style alignment rule.

Alignment (no lookahead): a feature row for trading day D predicts day D+1 and
is "locked" at D's close (~16:00 ET). So an article is attached to the first
trading day whose close falls on/after the article's timestamp:

    article time t (ET) <= 16:00 on a trading day  -> that trading day
    article after 16:00 ET (or on a non-trading day) -> the NEXT trading day

This guarantees row D only sees news from (close[D-1], close[D]] — i.e. news a
trader could actually have known before the D -> D+1 prediction.

For each (stock, trading day) we write:
    news_count          - number of articles attached to that day
    avg_sentiment_score - mean of available per-article sentiment scores

Other sentiment columns (avg_positive/negative/neutral) are left for a future
FinBERT pass. Run this AFTER build_features_daily.py (which sets the price
features) and after ingest_news.py.

Usage:
    python scripts/build_sentiment_features.py

Environment variables (read from .env):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import sys
import bisect
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 16          # 16:00 ET
UPSERT_BATCH_SIZE = 500


def get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


def init_supabase() -> Client:
    load_dotenv()
    url = get_env_var("SUPABASE_URL")
    key = get_env_var("SUPABASE_SERVICE_ROLE_KEY")
    client = create_client(url, key)
    logger.info("Connected to Supabase")
    return client


def fetch_active_tickers(supabase: Client) -> list[dict]:
    resp = (
        supabase.table("stocks").select("id, ticker").eq("is_active", True).execute()
    )
    if not resp.data:
        logger.error("No active tickers found")
        sys.exit(1)
    return resp.data


def _paginate(query_builder) -> list[dict]:
    """Run a Supabase select with range pagination (1000-row cap per request)."""
    rows, offset, batch = [], 0, 1000
    while True:
        resp = query_builder(offset, batch).execute()
        if not resp.data:
            break
        rows.extend(resp.data)
        if len(resp.data) < batch:
            break
        offset += batch
    return rows


def feature_dates(supabase: Client, stock_id: str) -> list[str]:
    """Sorted list of this stock's trading dates present in features_daily."""
    rows = _paginate(
        lambda off, b: supabase.table("features_daily")
        .select("date").eq("stock_id", stock_id).order("date").range(off, off + b - 1)
    )
    return sorted({r["date"] for r in rows})


def fetch_news_with_sentiment(supabase: Client, stock_id: str) -> pd.DataFrame:
    """All articles for a stock joined to their sentiment score."""
    articles = _paginate(
        lambda off, b: supabase.table("news_articles")
        .select("id, published_at").eq("stock_id", stock_id).range(off, off + b - 1)
    )
    if not articles:
        return pd.DataFrame(columns=["published_at", "sentiment_score"])

    art_df = pd.DataFrame(articles).rename(columns={"id": "news_id"})
    ids = art_df["news_id"].tolist()

    sent_rows = []
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        resp = (
            supabase.table("news_sentiment")
            .select("news_id, sentiment_score")
            .in_("news_id", chunk)
            .execute()
        )
        sent_rows.extend(resp.data or [])

    sent_df = pd.DataFrame(sent_rows) if sent_rows else pd.DataFrame(
        columns=["news_id", "sentiment_score"]
    )
    return art_df.merge(sent_df, on="news_id", how="left")


def assign_trading_date(published_at: str, trading_dates: list[str]) -> str | None:
    """
    Map an article timestamp to the trading day whose close first follows it.
    Returns an ISO date string present in trading_dates, or None if past the end.
    """
    ts = datetime.fromisoformat(published_at).astimezone(MARKET_TZ)
    candidate = ts.date()
    if ts.hour >= MARKET_CLOSE_HOUR:
        candidate = candidate + timedelta(days=1)
    candidate_iso = candidate.isoformat()

    idx = bisect.bisect_left(trading_dates, candidate_iso)
    if idx >= len(trading_dates):
        return None
    return trading_dates[idx]


def aggregate_stock(news_df: pd.DataFrame, trading_dates: list[str]) -> pd.DataFrame:
    """Per trading day: news_count and avg_sentiment_score for one stock."""
    if news_df.empty or not trading_dates:
        return pd.DataFrame(columns=["date", "news_count", "avg_sentiment_score"])

    df = news_df.copy()
    df["trading_date"] = df["published_at"].apply(
        lambda p: assign_trading_date(p, trading_dates)
    )
    df = df.dropna(subset=["trading_date"])
    if df.empty:
        return pd.DataFrame(columns=["date", "news_count", "avg_sentiment_score"])

    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    grouped = df.groupby("trading_date").agg(
        news_count=("news_id", "count"),
        avg_sentiment_score=("sentiment_score", "mean"),
    ).reset_index().rename(columns={"trading_date": "date"})

    # Round and replace NaN means (days where no article had a score) with None.
    grouped["avg_sentiment_score"] = grouped["avg_sentiment_score"].round(4)
    grouped = grouped.astype({"news_count": int})
    grouped["avg_sentiment_score"] = grouped["avg_sentiment_score"].where(
        grouped["avg_sentiment_score"].notna(), None
    )
    return grouped


def upsert_sentiment(supabase: Client, stock_id: str, agg: pd.DataFrame) -> int:
    """
    Write news_count / avg_sentiment_score into features_daily by (stock_id,date).
    Upsert only touches these columns; existing price features are preserved
    because every (stock_id, date) here already exists in features_daily.
    """
    if agg.empty:
        return 0
    rows = [
        {
            "stock_id": stock_id,
            "date": r["date"],
            "news_count": int(r["news_count"]),
            "avg_sentiment_score": (
                None if pd.isna(r["avg_sentiment_score"]) else r["avg_sentiment_score"]
            ),
        }
        for _, r in agg.iterrows()
    ]
    written = 0
    for i in range(0, len(rows), UPSERT_BATCH_SIZE):
        batch = rows[i:i + UPSERT_BATCH_SIZE]
        supabase.table("features_daily").upsert(
            batch, on_conflict="stock_id,date"
        ).execute()
        written += len(batch)
    return written


def build_sentiment_features():
    logger.info("=" * 60)
    logger.info("Building sentiment features (leakage-safe alignment)")
    logger.info("=" * 60)

    supabase = init_supabase()
    tickers = fetch_active_tickers(supabase)

    total_days = 0
    for i, stock in enumerate(tickers, 1):
        ticker, stock_id = stock["ticker"], stock["id"]
        dates = feature_dates(supabase, stock_id)
        news_df = fetch_news_with_sentiment(supabase, stock_id)

        if news_df.empty:
            logger.info(f"[{i}/{len(tickers)}] {ticker}: no news stored, skipping")
            continue

        agg = aggregate_stock(news_df, dates)
        written = upsert_sentiment(supabase, stock_id, agg)
        total_days += written
        logger.info(
            f"[{i}/{len(tickers)}] {ticker}: {len(news_df)} articles -> "
            f"{written} trading days updated"
        )

    logger.info("\n" + "=" * 60)
    logger.info(f"Done. Updated sentiment on {total_days} (stock, day) rows.")
    logger.info("Next: add sentiment to FEATURE_COLUMNS and rerun train_price_model.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        build_sentiment_features()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
