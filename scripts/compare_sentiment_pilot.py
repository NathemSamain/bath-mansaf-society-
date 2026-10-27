#!/usr/bin/env python3
"""
Pilot comparison: price-only vs price+sentiment, restricted to the news window.

News only exists for a small slice of the feature table (AAPL + MSFT, from
2025-06-25; MSFT news ends 2026-03-25), i.e. ~20% of AAPL days and ~15% of MSFT
days. Comparing on the full table would mostly measure "no coverage" rows
masquerading as zero-news days. So this script:

  1. Restricts to PILOT_TICKERS.
  2. Finds each ticker's news coverage (first/last trading day with
     news_count > 0) and keeps only the COMMON window across tickers, so both
     tickers are present on every date and the date-based split stays balanced.
  3. Inside that window, a day with no articles is a REAL zero: news_count = 0
     and avg_sentiment_score = 0 (neutral). These rows are kept, never dropped.
  4. Builds one labeled dataset with the union of features and evaluates both
     feature sets on the exact same rows and splits:
       - 70/15/15 date split (same protocol as train_price_model.py)
       - expanding-window walk-forward, pooling out-of-sample predictions, which
         gives a larger evaluation sample than the ~15% test slice.

Expect only ~250-350 usable rows after thresholding: treat results as a pilot,
not as evidence either way. Nothing here overwrites the production model; the
report is written to models/sentiment_pilot_report.json.

Prerequisite: run build_sentiment_features.py so features_daily has news_count
populated (the column defaults to 0, so un-built rows look like zero-news days).

Usage:
    python scripts/compare_sentiment_pilot.py
"""

import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import train_price_model as tpm
from train_price_model import (
    SCALE_FREE_FEATURES,
    DERIVED_FEATURES,
    MARKET_FEATURES,
    SENTIMENT_FEATURES,
    LABEL_COLUMN,
)

logger = logging.getLogger(__name__)

PILOT_TICKERS = ["AAPL", "MSFT"]

# Walk-forward: train on the first WF_INITIAL_TRAIN share of dates, then roll
# through the rest in WF_FOLDS contiguous blocks (expanding training window).
WF_INITIAL_TRAIN = 0.5
WF_FOLDS = 5

REPORT_PATH = Path("models") / "sentiment_pilot_report.json"


def restrict_to_news_window(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Keep PILOT_TICKERS rows inside the common news-coverage window, and turn
    missing sentiment inside that window into real zeros.
    """
    df = df[df["ticker"].isin(PILOT_TICKERS)].copy()
    missing = sorted(set(PILOT_TICKERS) - set(df["ticker"].unique()))
    if missing:
        logger.error(f"Pilot tickers not found in features_daily: {missing}")
        sys.exit(1)

    with_news = df[df["news_count"].fillna(0) > 0]
    coverage = {}
    for ticker in PILOT_TICKERS:
        dates = with_news.loc[with_news["ticker"] == ticker, "date"]
        if dates.empty:
            logger.error(
                f"{ticker} has no rows with news_count > 0. "
                f"Run build_sentiment_features.py first."
            )
            sys.exit(1)
        coverage[ticker] = {
            "first": dates.min(),
            "last": dates.max(),
            "news_days": int(len(dates)),
            "share_of_all_days": round(
                len(dates) / int((df["ticker"] == ticker).sum()), 4
            ),
        }
        logger.info(
            f"{ticker} news coverage: {coverage[ticker]['first']} -> "
            f"{coverage[ticker]['last']} ({len(dates)} days, "
            f"{coverage[ticker]['share_of_all_days'] * 100:.1f}% of its rows)"
        )

    window_start = max(c["first"] for c in coverage.values())
    window_end = min(c["last"] for c in coverage.values())
    df = df[(df["date"] >= window_start) & (df["date"] <= window_end)].copy()
    logger.info(
        f"Common news window: {window_start} -> {window_end} "
        f"({df['date'].nunique()} days, {len(df)} rows)"
    )

    # Inside the window, absence of news is a genuine observation: 0 articles
    # and neutral sentiment. Fill explicitly so these rows survive cleaning.
    df["news_count"] = df["news_count"].fillna(0)
    df["avg_sentiment_score"] = df["avg_sentiment_score"].where(
        df["news_count"] > 0, 0.0
    ).fillna(0.0)
    zero_news = int((df["news_count"] == 0).sum())
    logger.info(f"Zero-news days kept as real zeros: {zero_news}")

    window = {
        "start": window_start,
        "end": window_end,
        "days": int(df["date"].nunique()),
        "rows": int(len(df)),
        "zero_news_rows": zero_news,
        "per_ticker_coverage": coverage,
    }
    return df, window


def fit_models(X_train, y_train) -> dict:
    """Fit the same two model families as the production pipeline."""
    return {
        "Logistic Regression": tpm.train_logistic_regression(X_train, y_train),
        "Random Forest": tpm.train_random_forest(X_train, y_train),
    }


def evaluate_split(train_df, val_df, test_df, features: list[str]) -> dict:
    """
    Production protocol on one feature set: fit on train, calibrate on val,
    select by val MCC (F1 tie-break), report the selected model on test.
    """
    X_train, y_train = tpm.prepare_matrices(train_df, features)
    X_val, y_val = tpm.prepare_matrices(val_df, features)
    X_test, y_test = tpm.prepare_matrices(test_df, features)

    results = {}
    for name, raw in fit_models(X_train, y_train).items():
        model, calibrated = tpm.calibrate_model(raw, X_val, y_val)
        val_m = tpm.compute_metrics(y_val, model.predict(X_val))
        val_m["auc"] = tpm.compute_auc(y_val, tpm.positive_scores(model, X_val))
        test_pred = model.predict(X_test)
        test_m = tpm.compute_metrics(y_test, test_pred)
        test_m["auc"] = tpm.compute_auc(y_test, tpm.positive_scores(model, X_test))
        results[name] = {
            "calibrated": calibrated,
            "validation": val_m,
            "test": test_m,
            "test_collapse": tpm.check_class_collapse(name, test_pred),
        }

    best = max(
        results,
        key=lambda n: (results[n]["validation"]["mcc"], results[n]["validation"]["f1"]),
    )
    return {"models": results, "best_model": best}


def evaluate_walk_forward(df: pd.DataFrame, features: list[str]) -> dict:
    """
    Expanding-window walk-forward over unique dates. Each fold trains on all
    earlier dates and predicts the next block; out-of-sample predictions are
    pooled so metrics use every post-warmup row.
    """
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = np.sort(df["date"].unique())
    start = int(len(dates) * WF_INITIAL_TRAIN)
    blocks = np.array_split(dates[start:], WF_FOLDS)

    pooled = {}  # model -> {"y": [], "pred": [], "score": []}
    folds = []
    for i, block in enumerate(blocks, 1):
        train = df[df["date"] < block[0]]
        test = df[df["date"].isin(set(block))]
        X_train, y_train = tpm.prepare_matrices(train, features)
        X_test, y_test = tpm.prepare_matrices(test, features)

        fold = {"fold": i, "train_rows": len(train), "test_rows": len(test),
                "test_start": block[0], "test_end": block[-1], "mcc": {}}
        for name, model in fit_models(X_train, y_train).items():
            pred = model.predict(X_test)
            p = pooled.setdefault(name, {"y": [], "pred": [], "score": []})
            p["y"].extend(y_test)
            p["pred"].extend(pred)
            p["score"].extend(tpm.positive_scores(model, X_test))
            fold["mcc"][name] = tpm.compute_metrics(y_test, pred)["mcc"]
        folds.append(fold)

    summary = {}
    for name, p in pooled.items():
        y, pred = np.array(p["y"]), np.array(p["pred"])
        m = tpm.compute_metrics(y, pred)
        m["auc"] = tpm.compute_auc(y, np.array(p["score"]))
        m["n"] = int(len(y))
        summary[name] = m
    return {"pooled": summary, "folds": folds}


def log_comparison(split_results: dict, wf_results: dict, baselines: dict) -> None:
    """Side-by-side table: both feature sets on identical rows."""
    rows = []
    for name, m in baselines.items():
        rows.append({"model": name, "split": "Test", "metrics": m})
    for arm, res in split_results.items():
        best = res["best_model"]
        rows.append({
            "model": f"{arm}: {best}"[:22], "split": "Test",
            "metrics": res["models"][best]["test"],
        })
    for arm, res in wf_results.items():
        for name, m in res["pooled"].items():
            short = "LR" if name.startswith("Logistic") else "RF"
            rows.append({"model": f"{arm}: {short}", "split": "WalkFwd OOS",
                         "metrics": m})
    tpm.print_comparison_table(rows)


def run_pilot():
    logger.info("=" * 60)
    logger.info("Sentiment Pilot: price-only vs price+sentiment (news window)")
    logger.info("=" * 60)

    supabase = tpm.init_supabase()
    df = tpm.fetch_features(supabase)
    tpm.validate_columns(df)
    df = tpm.derive_normalized_features(df)

    df, window = restrict_to_news_window(df)

    price_features = list(SCALE_FREE_FEATURES + DERIVED_FEATURES)
    market_df = tpm.fetch_market_features(df["date"].min(), df["date"].max())
    if market_df is not None:
        df = df.merge(market_df, on="date", how="left")
        price_features += MARKET_FEATURES
    arms = {
        "price": price_features,
        "price+sent": price_features + SENTIMENT_FEATURES,
    }

    # Clean on the union of features so both arms see the exact same rows.
    df = tpm.clean_data(df, arms["price+sent"])
    df = tpm.apply_label_thresholding(df)
    df = tpm.add_previous_direction(df)
    n_rows = len(df)
    if n_rows < 200:
        logger.warning(f"Only {n_rows} labeled rows: results are very noisy.")

    train_df, val_df, test_df = tpm.split_data(df)

    # Baselines on the shared test split (feature-set independent).
    y_train = train_df[LABEL_COLUMN].values.astype(int)
    y_test = test_df[LABEL_COLUMN].values.astype(int)
    p_up = float(y_train.mean())
    rng = np.random.default_rng(tpm.RANDOM_SEED)
    baselines = {
        name: tpm.compute_metrics(y_test, preds)
        for name, preds in tpm.baseline_predictions(
            test_df, p_up, int(round(p_up)), rng
        ).items()
    }

    split_results, wf_results = {}, {}
    for arm, features in arms.items():
        logger.info("\n" + "-" * 60)
        logger.info(f"Feature set '{arm}' ({len(features)} columns)")
        logger.info("-" * 60)
        split_results[arm] = evaluate_split(train_df, val_df, test_df, features)
        wf_results[arm] = evaluate_walk_forward(df, features)

    log_comparison(split_results, wf_results, baselines)

    for name in wf_results["price"]["pooled"]:
        delta = (wf_results["price+sent"]["pooled"][name]["mcc"]
                 - wf_results["price"]["pooled"][name]["mcc"])
        logger.info(f"Walk-forward MCC delta (sent - price), {name}: {delta:+.4f}")
    logger.info(
        f"PILOT: {n_rows} labeled rows, {len(test_df)} test rows, "
        f"{wf_results['price']['pooled']['Logistic Regression']['n']} walk-forward "
        f"OOS rows. Differences of a few MCC points are within noise."
    )

    report = {
        "pilot_tickers": PILOT_TICKERS,
        "news_window": window,
        "market_features_used": market_df is not None,
        "feature_sets": arms,
        "label_threshold_up": tpm.LABEL_THRESHOLD_UP,
        "label_threshold_down": tpm.LABEL_THRESHOLD_DOWN,
        "rows_after_thresholding": n_rows,
        "zero_news_rows_after_thresholding": int((df["news_count"] == 0).sum()),
        "split_rows": {"train": len(train_df), "validation": len(val_df),
                       "test": len(test_df)},
        "test_baselines": baselines,
        "split_results": split_results,
        "walk_forward": {"initial_train_share": WF_INITIAL_TRAIN,
                         "folds": WF_FOLDS, **{a: r for a, r in wf_results.items()}},
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    try:
        run_pilot()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
