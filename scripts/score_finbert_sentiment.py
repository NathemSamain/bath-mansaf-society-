#!/usr/bin/env python3
"""
Stage 3: score Finnhub news with FinBERT (ProsusAI/finbert) on the GPU.

For every news_articles row with provider='finnhub' that has no FinBERT row in
news_sentiment yet, scores "headline. summary" (the summary helps disambiguate
Finnhub's loose ticker tagging) and writes:

    news_id, model_name='finbert', positive_score, negative_score,
    neutral_score, sentiment_label (argmax), sentiment_score (pos - neg),
    embedding = NULL

Resumable: already-scored news_ids are loaded up front and skipped, and writes
upsert on the (news_id, model_name) unique key with ignore_duplicates, so an
interrupted run just continues. Alpha Vantage rows (model_name='AlphaVantage')
are never read or written.

Refuses to run without CUDA: scoring ~290k articles on CPU is not intended.

Setup (once; the CUDA build of torch is not on PyPI):
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    pip install transformers

Usage:
    python scripts/score_finbert_sentiment.py --sample 10   # dry run, no writes
    python scripts/score_finbert_sentiment.py               # score everything

Environment variables (read from .env):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import sys
import time
import logging
import argparse

from dotenv import load_dotenv
from supabase import create_client, Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# Supabase / Hugging Face HTTP clients log every request at INFO; keep progress
# readable.
for _name in ("httpx", "httpx2"):
    logging.getLogger(_name).setLevel(logging.WARNING)

MODEL_ID = "ProsusAI/finbert"
MODEL_NAME = "finbert"           # news_sentiment.model_name for these rows
PROVIDER = "finnhub"
BATCH_SIZE = 64                  # articles per GPU forward pass
MAX_TOKENS = 512
PAGE_SIZE = 1000                 # articles fetched (and written) per page
UPSERT_BATCH_SIZE = 500
LOG_EVERY_BATCHES = 50


def get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


def init_supabase() -> Client:
    load_dotenv(".env")
    client = create_client(
        get_env_var("SUPABASE_URL"), get_env_var("SUPABASE_SERVICE_ROLE_KEY")
    )
    logger.info("Connected to Supabase")
    return client


def require_cuda():
    """Return the CUDA device, or exit: never fall back to CPU."""
    import torch

    if not torch.cuda.is_available():
        logger.error(
            f"CUDA is not available (torch {torch.__version__}, "
            f"cuda build {torch.version.cuda}). Refusing to score on CPU."
        )
        sys.exit(1)
    logger.info(
        f"GPU: {torch.cuda.get_device_name(0)} "
        f"(torch {torch.__version__}, CUDA {torch.version.cuda})"
    )
    return torch.device("cuda")


def load_model(device):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device).eval()
    # Read the label order from the config instead of assuming it.
    labels = {i: l.lower() for i, l in model.config.id2label.items()}
    logger.info(f"Loaded {MODEL_ID} (labels: {labels})")
    return tokenizer, model, labels


def scored_news_ids(supabase: Client) -> set:
    """news_ids that already have a FinBERT row (keyset-paginated)."""
    ids, last = set(), None
    while True:
        q = (
            supabase.table("news_sentiment").select("news_id")
            .eq("model_name", MODEL_NAME).order("news_id").limit(PAGE_SIZE)
        )
        if last:
            q = q.gt("news_id", last)
        rows = q.execute().data or []
        if not rows:
            break
        ids.update(r["news_id"] for r in rows)
        last = rows[-1]["news_id"]
        if len(rows) < PAGE_SIZE:
            break
    return ids


def count_finnhub_articles(supabase: Client) -> int:
    resp = (
        supabase.table("news_articles").select("id", count="exact")
        .eq("provider", PROVIDER).limit(1).execute()
    )
    return resp.count or 0


def iter_unscored_pages(supabase: Client, skip: set):
    """Yield pages of unscored Finnhub articles (keyset-paginated by id)."""
    last = None
    while True:
        q = (
            supabase.table("news_articles").select("id, headline, summary")
            .eq("provider", PROVIDER).order("id").limit(PAGE_SIZE)
        )
        if last:
            q = q.gt("id", last)
        rows = q.execute().data or []
        if not rows:
            return
        last = rows[-1]["id"]
        page = [r for r in rows if r["id"] not in skip]
        if page:
            yield page
        if len(rows) < PAGE_SIZE:
            return


def article_text(row: dict) -> str:
    """Headline plus summary, skipping a summary that just repeats the headline."""
    headline = (row.get("headline") or "").strip()
    summary = (row.get("summary") or "").strip()
    if summary and summary != headline:
        return f"{headline}. {summary}"
    return headline


def score_texts(texts: list[str], tokenizer, model, labels: dict, device) -> list[dict]:
    """FinBERT class probabilities for each text, in input order."""
    import torch

    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_TOKENS,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        logits = model(**enc).logits
    probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()

    results = []
    for p in probs:
        by_label = {labels[i]: v for i, v in enumerate(p)}
        pos, neg, neu = by_label["positive"], by_label["negative"], by_label["neutral"]
        results.append({
            "positive_score": round(pos, 6),
            "negative_score": round(neg, 6),
            "neutral_score": round(neu, 6),
            "sentiment_label": max(by_label, key=by_label.get),
            "sentiment_score": round(pos - neg, 6),
        })
    return results


def score_page(page: list[dict], tokenizer, model, labels: dict, device) -> list[dict]:
    """Score one page in GPU batches; returns news_sentiment rows."""
    # Batching similar lengths together cuts padding work considerably.
    page = sorted(page, key=lambda r: len(article_text(r)))
    rows = []
    for i in range(0, len(page), BATCH_SIZE):
        chunk = page[i:i + BATCH_SIZE]
        scores = score_texts(
            [article_text(r) for r in chunk], tokenizer, model, labels, device
        )
        for art, s in zip(chunk, scores):
            rows.append({"news_id": art["id"], "model_name": MODEL_NAME,
                         "embedding": None, **s})
    return rows


def write_rows(supabase: Client, rows: list[dict]) -> None:
    for i in range(0, len(rows), UPSERT_BATCH_SIZE):
        supabase.table("news_sentiment").upsert(
            rows[i:i + UPSERT_BATCH_SIZE],
            on_conflict="news_id,model_name",
            ignore_duplicates=True,
        ).execute()


def run_sample(supabase: Client, n: int, tokenizer, model, labels, device) -> None:
    """Dry run: score the first n unscored articles and print them; no writes."""
    skip = scored_news_ids(supabase)
    page = next(iter_unscored_pages(supabase, skip), [])[:n]
    rows = {r["news_id"]: r for r in score_page(page, tokenizer, model, labels, device)}
    for art in page:
        s = rows[art["id"]]
        logger.info(
            f"\n  {s['sentiment_label']:<8} score={s['sentiment_score']:+.3f} "
            f"(pos {s['positive_score']:.3f} / neg {s['negative_score']:.3f} / "
            f"neu {s['neutral_score']:.3f})\n  {article_text(art)[:220]}"
        )
    logger.info(f"\nSample of {len(page)} scored; nothing written (dry run).")


def run_full(supabase: Client, tokenizer, model, labels, device) -> None:
    total = count_finnhub_articles(supabase)
    skip = scored_news_ids(supabase)
    todo = total - len(skip)
    logger.info(
        f"Finnhub articles: {total}; already scored: {len(skip)}; to score: {todo}"
    )

    written, batches, start = 0, 0, time.time()
    next_log = LOG_EVERY_BATCHES
    for page in iter_unscored_pages(supabase, skip):
        rows = score_page(page, tokenizer, model, labels, device)
        write_rows(supabase, rows)
        written += len(rows)
        batches += -(-len(page) // BATCH_SIZE)
        if batches >= next_log:
            rate = written / (time.time() - start)
            eta_min = (todo - written) / rate / 60 if rate else float("nan")
            logger.info(
                f"  {written}/{todo} scored ({batches} batches, "
                f"{rate:.0f} articles/s, ETA {eta_min:.0f} min)"
            )
            next_log += LOG_EVERY_BATCHES

    final = len(scored_news_ids(supabase))
    logger.info("\n" + "=" * 60)
    logger.info(f"Scored and written this run: {written}")
    logger.info(f"FinBERT rows now in news_sentiment: {final} / {total} Finnhub articles")
    logger.info(f"Elapsed: {(time.time() - start) / 60:.1f} min")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sample", type=int, metavar="N",
        help="dry run: score N unscored articles, print them, write nothing",
    )
    args = parser.parse_args()

    device = require_cuda()
    supabase = init_supabase()
    tokenizer, model, labels = load_model(device)

    if args.sample:
        run_sample(supabase, args.sample, tokenizer, model, labels, device)
    else:
        run_full(supabase, tokenizer, model, labels, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user; re-run to resume")
        sys.exit(0)
