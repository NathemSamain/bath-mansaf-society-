#!/usr/bin/env python3
"""
News ingestion from Finnhub company_news into `news_articles` (articles only).

Finnhub's free tier keeps roughly one rolling year of company news (history
starts ~2025-10-28) and silently truncates each response to the ~245 NEWEST
articles in the requested range. A monthly window therefore only returns its
last few days, so this script requests ONE DAY at a time.

Design notes:
  * Provider marker: rows are written with provider = 'finnhub' (the existing
    `source` column holds the publisher, e.g. "Yahoo"), so Finnhub and Alpha
    Vantage articles stay distinguishable.
  * Leakage-safe timestamps: published_at comes from Finnhub's Unix `datetime`
    field as a UTC timestamp; build_sentiment_features.py aligns on it. The
    `date` column is generated from published_at by the database.
  * Resumable: each ticker restarts from the UTC day of its latest
    finnhub-sourced published_at (that day is re-fetched; dedup absorbs the
    overlap), so a re-run after a rate-limit stop continues instead of
    restarting.
  * Dedup: upsert on the (stock_id, url) unique constraint with
    ignore_duplicates, so overlapping fetches never duplicate rows and never
    overwrite an existing (e.g. Alpha Vantage) row with the same URL.
  * Rate limit: ~60 calls/min on the free tier -> CALL_DELAY between calls,
    exponential backoff on HTTP 429. If 429s persist, the script stops cleanly;
    just run it again.
  * news_sentiment is NOT written; scoring is a separate step.

Requires migration 20261026120000_news_articles_provider.sql.

Usage:
    python scripts/ingest_finnhub_news.py

Optional environment overrides:
    FINNHUB_START_DATE  default 2025-10-28
    FINNHUB_END_DATE    default 2026-10-22 (inclusive)

Environment variables (read from .env):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    FINNHUB_API_KEY
"""

import os
import sys
import time
import logging
from datetime import date, datetime, timedelta, timezone

import finnhub
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROVIDER = "finnhub"
DEFAULT_START_DATE = "2025-10-28"   # start of Finnhub free-tier history
DEFAULT_END_DATE = "2026-10-22"     # inclusive
CALL_DELAY = 1.1                    # seconds between calls (~60/min free tier)
MAX_RETRIES = 6                     # per call, for 429s / transient errors
BACKOFF_BASE = 15                   # seconds; doubles per retry, capped below
BACKOFF_MAX = 120
FLUSH_EVERY_DAYS = 7                # upsert buffered articles this often
UPSERT_BATCH_SIZE = 500
# A single day returning about this many articles may itself be truncated.
TRUNCATION_WARN = 240


class RateLimitExhausted(Exception):
    """Raised when 429s persist past MAX_RETRIES; the run stops cleanly."""


def get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


def init_supabase() -> Client:
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


def resume_date(supabase: Client, stock_id: str, start: date) -> date:
    """
    UTC day of this ticker's latest finnhub-sourced article, or `start` if none.
    That day is re-fetched because it may have been only partly stored.
    """
    resp = (
        supabase.table("news_articles")
        .select("published_at")
        .eq("stock_id", stock_id)
        .eq("provider", PROVIDER)
        .order("published_at", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return start
    latest = datetime.fromisoformat(resp.data[0]["published_at"])
    return max(start, latest.astimezone(timezone.utc).date())


def fetch_day(client: finnhub.Client, ticker: str, day: date) -> list[dict]:
    """company_news for a single day, with backoff on 429 / transient errors."""
    day_iso = day.isoformat()
    for attempt in range(MAX_RETRIES):
        try:
            return client.company_news(ticker, _from=day_iso, to=day_iso) or []
        except finnhub.FinnhubAPIException as e:
            if e.status_code in (401, 403):
                logger.error(f"Finnhub rejected the API key ({e.status_code}): {e}")
                sys.exit(1)
            if e.status_code != 429 and e.status_code < 500:
                logger.error(f"{ticker} {day_iso}: Finnhub error {e.status_code}: {e}")
                return []
            reason = "rate limited (429)" if e.status_code == 429 else f"HTTP {e.status_code}"
        except (
            finnhub.FinnhubRequestException,
            requests.exceptions.RequestException,
            ValueError,
        ) as e:
            reason = f"request failed ({e})"
        wait = min(BACKOFF_BASE * 2 ** attempt, BACKOFF_MAX)
        logger.warning(
            f"{ticker} {day_iso}: {reason}; retry {attempt + 1}/{MAX_RETRIES} in {wait}s"
        )
        time.sleep(wait)
    raise RateLimitExhausted(f"{ticker} {day_iso}: giving up after {MAX_RETRIES} retries")


def parse_articles(stock_id: str, items: list[dict]) -> list[dict]:
    """Map Finnhub items to news_articles rows (date is DB-generated)."""
    rows = []
    for item in items:
        url = item.get("url")
        headline = (item.get("headline") or "").strip()
        ts = item.get("datetime")
        if not url or not headline or not ts:
            continue
        rows.append({
            "stock_id": stock_id,
            "published_at": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(),
            "headline": headline[:1000],
            "summary": item.get("summary") or None,
            "source": item.get("source") or None,
            "url": url,
            "provider": PROVIDER,
        })
    return rows


def store_articles(supabase: Client, rows: list[dict]) -> int:
    """Upsert rows, skipping any (stock_id, url) already stored."""
    # Postgres rejects an upsert batch that hits the same key twice.
    unique = list({(r["stock_id"], r["url"]): r for r in rows}.values())
    for i in range(0, len(unique), UPSERT_BATCH_SIZE):
        supabase.table("news_articles").upsert(
            unique[i:i + UPSERT_BATCH_SIZE],
            on_conflict="stock_id,url",
            ignore_duplicates=True,
        ).execute()
    return len(unique)


def ingest_ticker(
    supabase: Client, client: finnhub.Client, stock: dict, start: date, end: date
) -> int:
    """Fetch [resume point, end] one day at a time; returns articles sent."""
    ticker, stock_id = stock["ticker"], stock["id"]
    day = resume_date(supabase, stock_id, start)
    if day > end:
        logger.info(f"  {ticker}: already complete through {end}")
        return 0
    logger.info(f"  {ticker}: fetching {day} -> {end}")

    buffer, sent, days_buffered = [], 0, 0
    try:
        while day <= end:
            items = fetch_day(client, ticker, day)
            if len(items) >= TRUNCATION_WARN:
                logger.warning(
                    f"  {ticker} {day}: {len(items)} articles, possibly truncated"
                )
            buffer.extend(parse_articles(stock_id, items))
            days_buffered += 1
            if days_buffered >= FLUSH_EVERY_DAYS:
                sent += store_articles(supabase, buffer)
                logger.info(f"  {ticker}: stored through {day} ({sent} articles sent)")
                buffer, days_buffered = [], 0
            day += timedelta(days=1)
            time.sleep(CALL_DELAY)
    finally:
        # Persist what we have so a resume starts from the latest stored day.
        if buffer:
            sent += store_articles(supabase, buffer)
    return sent


def log_summary(supabase: Client, tickers: list[dict]) -> None:
    """Per-ticker finnhub article count and min/max published date."""
    logger.info("\n" + "=" * 60)
    logger.info(f"{'Ticker':<7} {'Articles':>8}  {'First':<10}  {'Last':<10}")
    logger.info("-" * 60)
    total = 0
    for stock in sorted(tickers, key=lambda s: s["ticker"]):
        base = lambda: (
            supabase.table("news_articles").select("published_at", count="exact")
            .eq("stock_id", stock["id"]).eq("provider", PROVIDER)
        )
        first = base().order("published_at").limit(1).execute()
        last = base().order("published_at", desc=True).limit(1).execute()
        count = first.count or 0
        total += count
        fmt = lambda r: r.data[0]["published_at"][:10] if r.data else "-"
        logger.info(f"{stock['ticker']:<7} {count:>8}  {fmt(first):<10}  {fmt(last):<10}")
    logger.info("-" * 60)
    logger.info(f"{'TOTAL':<7} {total:>8}")
    logger.info("=" * 60)


def ingest_finnhub_news():
    logger.info("=" * 60)
    logger.info("Finnhub company news ingestion (articles only)")
    logger.info("=" * 60)

    load_dotenv(".env")
    supabase = init_supabase()
    client = finnhub.Client(api_key=get_env_var("FINNHUB_API_KEY"))

    start = date.fromisoformat(os.environ.get("FINNHUB_START_DATE", DEFAULT_START_DATE))
    end = date.fromisoformat(os.environ.get("FINNHUB_END_DATE", DEFAULT_END_DATE))
    logger.info(f"Window: {start} -> {end} (one call per ticker per day)")

    tickers = fetch_active_tickers(supabase)
    try:
        for i, stock in enumerate(tickers, 1):
            logger.info(f"\n[{i}/{len(tickers)}] {stock['ticker']}")
            sent = ingest_ticker(supabase, client, stock, start, end)
            logger.info(f"  {stock['ticker']}: done ({sent} articles sent to upsert)")
    except RateLimitExhausted as e:
        logger.warning(f"Stopping: {e}. Re-run to resume where this left off.")
    finally:
        log_summary(supabase, tickers)


if __name__ == "__main__":
    try:
        ingest_finnhub_news()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user; re-run to resume")
        sys.exit(0)
