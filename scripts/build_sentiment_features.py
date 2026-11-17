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

Source: Finnhub articles (provider='finnhub') scored by FinBERT
(model_name='finbert'). Alpha Vantage articles/sentiment are ignored so the two
providers are never blended or double-counted on overlapping days.

For each (stock, trading day) inside the ticker's news window (first to last
trading day with news) we write:
    news_count             - number of articles attached to that day
    avg_sentiment_score    - mean FinBERT sentiment_score (positive - negative)
    avg_positive_sentiment - mean FinBERT positive probability
    avg_negative_sentiment - mean FinBERT negative probability
    avg_neutral_sentiment  - mean FinBERT neutral probability

A trading day inside the window with no articles is a REAL zero-news day: it
gets news_count=0, avg_sentiment_score=0 and neutral probabilities (0/0/1), so
it is kept rather than looking like missing data. Outside the window the
sentiment columns are reset to the table defaults (news_count=0, averages
NULL), which also clears values left by earlier aggregation runs.

Run this AFTER build_features_daily.py (which sets the price features), after
ingest_finnhub_news.py and after score_finbert_sentiment.py.

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
for _name in ("httpx", "httpx2"):
    logging.getLogger(_name).setLevel(logging.WARNING)

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 16          # 16:00 ET
UPSERT_BATCH_SIZE = 500
PROVIDER = "finnhub"            # news_articles.provider to aggregate
SENTIMENT_MODEL = "finbert"     # news_sentiment.model_name to aggregate
PROB_COLUMNS = {                # news_sentiment column -> features_daily column
    "positive_score": "avg_positive_sentiment",
    "negative_score": "avg_negative_sentiment",
    "neutral_score": "avg_neutral_sentiment",
}
SENTIMENT_COLUMNS = ["avg_sentiment_score"] + list(PROB_COLUMNS.values())
# Values for a trading day in the news window with no articles.
ZERO_NEWS_VALUES = {
    "news_count": 0,
    "avg_sentiment_score": 0.0,
    "avg_positive_sentiment": 0.0,
    "avg_negative_sentiment": 0.0,
    "avg_neutral_sentiment": 1.0,
}


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
    """
    Finnhub articles for a stock with their FinBERT scores, via an embedded
    join filtered to SENTIMENT_MODEL (one request per page of articles).
    """
    articles = _paginate(
        lambda off, b: supabase.table("news_articles")
        .select(
            "id, published_at, news_sentiment(sentiment_score, "
            + ", ".join(PROB_COLUMNS) + ")"
        )
        .eq("stock_id", stock_id)
        .eq("provider", PROVIDER)
        .eq("news_sentiment.model_name", SENTIMENT_MODEL)
        # (provider, stock_id, published_at) index; id breaks timestamp ties so
        # offset pagination is stable.
        .order("published_at")
        .order("id")
        .range(off, off + b - 1)
    )
    columns = ["news_id", "published_at", "sentiment_score", *PROB_COLUMNS]
    if not articles:
        return pd.DataFrame(columns=columns)

    rows = []
    for a in articles:
        scores = (a.get("news_sentiment") or [{}])[0]
        rows.append({
            "news_id": a["id"],
            "published_at": a["published_at"],
            "sentiment_score": scores.get("sentiment_score"),
            **{c: scores.get(c) for c in PROB_COLUMNS},
        })
    return pd.DataFrame(rows, columns=columns)


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
    """
    Per trading day inside the ticker's news window: news_count and mean
    FinBERT scores. Days in the window with no articles get ZERO_NEWS_VALUES.
    """
    out_cols = ["date", "news_count"] + SENTIMENT_COLUMNS
    if news_df.empty or not trading_dates:
        return pd.DataFrame(columns=out_cols)

    df = news_df.copy()
    df["trading_date"] = df["published_at"].apply(
        lambda p: assign_trading_date(p, trading_dates)
    )
    df = df.dropna(subset=["trading_date"])
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    for col in ["sentiment_score", *PROB_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby("trading_date").agg(
        news_count=("news_id", "count"),
        avg_sentiment_score=("sentiment_score", "mean"),
        **{dst: (src, "mean") for src, dst in PROB_COLUMNS.items()},
    )

    # Every trading day between the first and last day with news is in the
    # window; days without articles are real zero-news days, not missing.
    window = [d for d in trading_dates if grouped.index.min() <= d <= grouped.index.max()]
    grouped = grouped.reindex(window)
    no_news = grouped["news_count"].isna()
    for col, value in ZERO_NEWS_VALUES.items():
        grouped.loc[no_news, col] = value
    grouped["news_count"] = grouped["news_count"].astype(int)
    grouped[SENTIMENT_COLUMNS] = grouped[SENTIMENT_COLUMNS].round(4)

    return grouped.rename_axis("date").reset_index()[out_cols]


def clear_outside_window(supabase: Client, stock_id: str, start: str, end: str) -> None:
    """
    Reset sentiment columns outside [start, end] to the table defaults, so no
    stale values (e.g. from an earlier Alpha Vantage aggregation) remain.
    """
    reset = {"news_count": 0, **{c: None for c in SENTIMENT_COLUMNS}}
    for bound in ("lt", "gt"):
        query = supabase.table("features_daily").update(reset).eq("stock_id", stock_id)
        getattr(query, bound)("date", start if bound == "lt" else end).execute()


def upsert_sentiment(supabase: Client, stock_id: str, agg: pd.DataFrame) -> int:
    """
    Write news_count and the sentiment averages into features_daily by
    (stock_id, date). Upsert only touches these columns; existing price
    features are preserved because every (stock_id, date) here already exists
    in features_daily.
    """
    if agg.empty:
        return 0
    rows = [
        {
            "stock_id": stock_id,
            "date": r["date"],
            "news_count": int(r["news_count"]),
            **{c: (None if pd.isna(r[c]) else float(r[c])) for c in SENTIMENT_COLUMNS},
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

    total_days, summary = 0, []
    for i, stock in enumerate(tickers, 1):
        ticker, stock_id = stock["ticker"], stock["id"]
        dates = feature_dates(supabase, stock_id)
        news_df = fetch_news_with_sentiment(supabase, stock_id)
        agg = aggregate_stock(news_df, dates)

        if agg.empty:
            # No Finnhub news for this ticker: reset every row to the defaults.
            clear_outside_window(supabase, stock_id, "9999-12-31", "9999-12-31")
            logger.info(f"[{i}/{len(tickers)}] {ticker}: no {PROVIDER} news, cleared")
            continue

        unscored = int(news_df["sentiment_score"].isna().sum())
        if unscored:
            logger.warning(
                f"  {ticker}: {unscored} articles have no {SENTIMENT_MODEL} score "
                f"(counted in news_count, excluded from averages)"
            )

        start, end = agg["date"].min(), agg["date"].max()
        clear_outside_window(supabase, stock_id, start, end)
        written = upsert_sentiment(supabase, stock_id, agg)
        total_days += written
        news_days = int((agg["news_count"] > 0).sum())
        summary.append((ticker, news_days, written - news_days, start, end))
        logger.info(
            f"[{i}/{len(tickers)}] {ticker}: {len(news_df)} articles -> "
            f"{written} trading days in window {start} -> {end}"
        )

    logger.info("=" * 60)
    logger.info(f"{'Ticker':<7} {'News days':>9} {'Zero-news':>9}  {'First':<10}  {'Last':<10}")
    logger.info("-" * 60)
    for ticker, news_days, zero_days, start, end in sorted(summary):
        logger.info(f"{ticker:<7} {news_days:>9} {zero_days:>9}  {start:<10}  {end:<10}")
    logger.info("-" * 60)
    logger.info(f"Done. Updated sentiment on {total_days} (stock, day) rows.")
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
