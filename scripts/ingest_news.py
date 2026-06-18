#!/usr/bin/env python3
"""
News ingestion for Stage 2 (sentiment) using Alpha Vantage NEWS_SENTIMENT.

Pulls news for every active ticker into `news_articles`, and the per-ticker
sentiment score AV provides into `news_sentiment` (model_name='AlphaVantage').
Later stages can add a Finnhub provider and/or recompute sentiment with FinBERT
by writing new `news_sentiment` rows (model_name='FinBERT') for the same
articles — the schema already supports multiple sentiment rows per article.

Design notes:
  * Resumable: re-running continues forward from the latest stored article per
    ticker, and de-duplicates by URL, so an interrupted/rate-limited run is safe
    to repeat.
  * Free-tier aware: Alpha Vantage allows ~25 requests/day and ~5/min. With 30
    tickers and pagination this will NOT finish in one day on the free tier —
    the script stops cleanly when it hits the daily cap; just run it again the
    next day to continue.

Usage:
    python scripts/ingest_news.py

Environment variables (read from .env):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    ALPHA_VANTAGE_API_KEY
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from supabase import create_client, Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
NEWS_FUNCTION = "NEWS_SENTIMENT"
RATE_LIMIT_DELAY = 15          # seconds between calls (free tier ~5/min)
PAGE_LIMIT = 1000              # max articles per request
DEFAULT_TIME_FROM = "20220301T0000"  # AV news history starts ~2022
SENTIMENT_MODEL = "AlphaVantage"


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
    logger.info(f"Found {len(resp.data)} active tickers")
    return resp.data


def latest_stored_time_from(supabase: Client, stock_id: str) -> str:
    """
    Resume point: the AV time_from string just after the newest article we
    already stored for this ticker (so we fetch forward, not re-fetch history).
    """
    resp = (
        supabase.table("news_articles")
        .select("published_at")
        .eq("stock_id", stock_id)
        .order("published_at", desc=True)
        .limit(1)
        .execute()
    )
    if resp.data:
        ts = datetime.fromisoformat(resp.data[0]["published_at"])
        return ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")
    return DEFAULT_TIME_FROM


def existing_urls(supabase: Client, stock_id: str) -> set:
    """All URLs already stored for this ticker, for de-duplication."""
    urls, offset, batch = set(), 0, 1000
    while True:
        resp = (
            supabase.table("news_articles")
            .select("url")
            .eq("stock_id", stock_id)
            .range(offset, offset + batch - 1)
            .execute()
        )
        if not resp.data:
            break
        urls.update(r["url"] for r in resp.data if r.get("url"))
        if len(resp.data) < batch:
            break
        offset += batch
    return urls


def fetch_av_news(ticker: str, api_key: str, time_from: str) -> tuple[list, str]:
    """
    Fetch one page of AV NEWS_SENTIMENT for a ticker.
    Returns (feed_items, status) where status is 'ok', 'empty', or 'rate_limit'.
    """
    params = {
        "function": NEWS_FUNCTION,
        "tickers": ticker,
        "time_from": time_from,
        "limit": PAGE_LIMIT,
        "sort": "EARLIEST",
        "apikey": api_key,
    }
    try:
        resp = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching news for {ticker}: {e}")
        return [], "empty"

    # Free-tier daily cap / throttle messages.
    if "Information" in data or "Note" in data:
        logger.warning(f"Alpha Vantage limit: {data.get('Information') or data.get('Note')}")
        return [], "rate_limit"

    feed = data.get("feed")
    if not feed:
        return [], "empty"
    return feed, "ok"


def parse_articles(stock_id: str, ticker: str, feed: list, skip_urls: set) -> tuple[list, list]:
    """
    Split AV feed items into (article_rows, sentiment_payloads).
    sentiment_payloads carry the article URL so they can be linked to the
    inserted article id afterward.
    """
    articles, sentiments = [], []
    for item in feed:
        url = item.get("url")
        if not url or url in skip_urls:
            continue
        skip_urls.add(url)

        try:
            published = datetime.strptime(
                item["time_published"], "%Y%m%dT%H%M%S"
            ).replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue

        authors = item.get("authors") or []
        articles.append({
            "stock_id": stock_id,
            "published_at": published.isoformat(),
            "headline": item.get("title", "")[:1000],
            "summary": item.get("summary"),
            "source": item.get("source"),
            "url": url,
            "author": ", ".join(authors)[:500] if authors else None,
            "raw_text": item.get("summary"),
        })

        # Pull the sentiment specific to THIS ticker (not the overall article).
        ticker_score, ticker_label = None, None
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                try:
                    ticker_score = float(ts.get("ticker_sentiment_score"))
                except (TypeError, ValueError):
                    ticker_score = None
                ticker_label = ts.get("ticker_sentiment_label")
                break
        if ticker_score is None:
            try:
                ticker_score = float(item.get("overall_sentiment_score"))
            except (TypeError, ValueError):
                ticker_score = None
            ticker_label = item.get("overall_sentiment_label")

        sentiments.append({
            "url": url,
            "model_name": SENTIMENT_MODEL,
            "sentiment_score": ticker_score,
            "sentiment_label": _norm_label(ticker_label),
        })
    return articles, sentiments


def _norm_label(label: str | None) -> str | None:
    """Map AV sentiment labels to the table's allowed positive/negative/neutral."""
    if not label:
        return None
    l = label.lower()
    if "bullish" in l or "positive" in l:
        return "positive"
    if "bearish" in l or "negative" in l:
        return "negative"
    return "neutral"


def store_page(supabase: Client, articles: list, sentiments: list) -> int:
    """Insert a page of articles, then their sentiment rows linked by id."""
    if not articles:
        return 0
    resp = supabase.table("news_articles").insert(articles).execute()
    inserted = resp.data or []
    url_to_id = {r["url"]: r["id"] for r in inserted}

    sentiment_rows = []
    for s in sentiments:
        news_id = url_to_id.get(s["url"])
        if not news_id:
            continue
        sentiment_rows.append({
            "news_id": news_id,
            "model_name": s["model_name"],
            "sentiment_score": s["sentiment_score"],
            "sentiment_label": s["sentiment_label"],
        })
    if sentiment_rows:
        supabase.table("news_sentiment").insert(sentiment_rows).execute()
    return len(inserted)


def ingest_news():
    logger.info("=" * 60)
    logger.info("News ingestion (Alpha Vantage NEWS_SENTIMENT)")
    logger.info("=" * 60)

    api_key = get_env_var("ALPHA_VANTAGE_API_KEY")
    supabase = init_supabase()
    tickers = fetch_active_tickers(supabase)

    request_count = 0
    total_articles = 0

    for i, stock in enumerate(tickers, 1):
        ticker, stock_id = stock["ticker"], stock["id"]
        logger.info(f"\n[{i}/{len(tickers)}] {ticker}")

        time_from = latest_stored_time_from(supabase, stock_id)
        skip_urls = existing_urls(supabase, stock_id)
        logger.info(f"  Resuming from {time_from} ({len(skip_urls)} already stored)")

        while True:
            feed, status = fetch_av_news(ticker, api_key, time_from)
            request_count += 1

            if status == "rate_limit":
                logger.warning(
                    f"Daily request cap reached after {request_count} requests. "
                    f"Stored {total_articles} articles this run. Re-run tomorrow "
                    f"to continue."
                )
                logger.info("=" * 60)
                return
            if status == "empty" or not feed:
                logger.info(f"  No more news for {ticker}")
                break

            articles, sentiments = parse_articles(stock_id, ticker, feed, skip_urls)
            stored = store_page(supabase, articles, sentiments)
            total_articles += stored
            logger.info(f"  Stored {stored} new articles (page of {len(feed)})")

            # Advance the window past the last item on this page.
            last_time = feed[-1].get("time_published", "")
            if len(feed) < PAGE_LIMIT or not last_time:
                break
            try:
                nxt = datetime.strptime(last_time, "%Y%m%dT%H%M%S")
                time_from = nxt.strftime("%Y%m%dT%H%M")
            except ValueError:
                break

            time.sleep(RATE_LIMIT_DELAY)

        time.sleep(RATE_LIMIT_DELAY)

    logger.info("\n" + "=" * 60)
    logger.info(f"Done. Stored {total_articles} new articles in {request_count} requests.")
    logger.info("Next: python scripts/build_sentiment_features.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        ingest_news()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
