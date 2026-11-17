#!/usr/bin/env python3
"""
Weekly-horizon comparison: 5-trading-day targets with price, price+HL and
price+sentiment feature sets.

The next-day target showed no signal for any feature set (MCC ~ 0). This
script asks the same question at a 5-day horizon. Everything is computed here
from prices_daily / features_daily; no table, daily script or production
model is modified.

Target (per ticker, sorted by date): r5 = close(t + 5 trading days) / close(t) - 1
  relative (primary, pre-registered): 1 if r5 beats the median r5 of all
      tickers on date t, else 0. Rows whose |r5 - median| falls inside a band
      are dropped; the band is the REL_BAND_SHARE quantile of |r5 - median|.
  absolute (secondary): 1 if r5 > 0, with a symmetric band at the
      ABS_BAND_SHARE quantile of |r5|.
  Each band is chosen on the first WF_INITIAL_TRAIN share of dates (inside
  the training data of the date split and of every walk-forward fold) and then
  frozen for all rows.

Sampling
  weekly (primary): the last trading day of each week as t (non-overlapping).
  daily (secondary): every trading day, so targets overlap.
  Either way a row trains a model only if its target is realized before the
  next split or walk-forward block starts (target end < that block's first
  date). For daily rows that is a 5-trading-day embargo; for weekly rows it
  drops the last week before each boundary.

Feature sets (identical rows and splits within a run)
  price      : production scale-free price + market (SPY/QQQ/VIX) features
  price+HL   : + 14-day Williams %R and the 5-day mean of (high - low) / close
  price+sent : + 5-day mean FinBERT score, 5-day news_count sum and sentiment
               surprise (5-day mean minus the mean of the 60 trading days
               before that 5-day window; >= 20 days of history required)
  Sentiment uses the common Finnhub news window and zero-news handling of
  compare_sentiment_pilot.py (in-window days without news are real zeros).

Runs
  (a) full history:    price vs price+HL
  (b) Finnhub window:  price vs price+HL vs price+sent (2025-11-01 on)

Evaluation per run x target x sampling: 70/15/15 date split (all tickers of a
date together) and expanding-window walk-forward with pooled out-of-sample
predictions; baselines Always-UP, Persistence (direction of the 5 trading days
ending at t, i.e. the previous week's label) and Random; MCC headline, plus
AUC. For each feature set vs price, a paired bootstrap over whole weeks gives
the MCC difference with a 95% interval. Permutation importance of the
price+sent Random Forest on the test split shows where sentiment ranks.

Usage:
    python scripts/compare_weekly.py
"""

import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, matthews_corrcoef

import train_price_model as tpm
import compare_sentiment_pilot as pilot
from train_price_model import (
    SCALE_FREE_FEATURES,
    DERIVED_FEATURES,
    MARKET_FEATURES,
    LABEL_COLUMN,
)

logger = logging.getLogger(__name__)
for _name in ("httpx", "httpx2"):
    logging.getLogger(_name).setLevel(logging.WARNING)

HORIZON = 5                 # trading days between t and the target close
REL_BAND_SHARE = 0.20       # training rows dropped nearest the date's median
ABS_BAND_SHARE = 0.225      # training rows dropped nearest a zero return

WILLIAMS_WINDOW = 14
HL_WINDOW = 5
SENT_WINDOW = 5
SURPRISE_BASELINE_DAYS = 60
SURPRISE_MIN_HISTORY = 20

HL_FEATURES = ["williams_r_14", "hl_range_5d"]
SENT_FEATURES = ["sent_mean_5d", "news_count_5d", "sent_surprise"]

# Walk-forward: train on the first WF_INITIAL_TRAIN share of dates, then roll
# through the rest in WF_FOLDS contiguous blocks (expanding training window).
WF_INITIAL_TRAIN = 0.5
WF_FOLDS = 5

BOOTSTRAP_ITERATIONS = 2000
PERMUTATION_REPEATS = 30

# Shallow trees with large leaves: the signal, if any, is weak, so the forest
# must not be able to memorize individual weeks.
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "class_weight": "balanced",
}

TARGETS = {
    "relative": {"return": "excess", "past": "excess_past", "band_share": REL_BAND_SHARE},
    "absolute": {"return": "r5", "past": "r5_past", "band_share": ABS_BAND_SHARE},
}
SAMPLINGS = ["weekly", "daily"]
PRIMARY = ("relative", "weekly")
MODEL_SHORT = {"Logistic Regression": "LR", "Random Forest": "RF"}

FEATURE_SELECT = ", ".join(
    ["date", *tpm.RAW_FETCH_COLUMNS, "news_count", "avg_sentiment_score"]
)
REPORT_PATH = Path("models") / "weekly_comparison_report.json"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_stock_rows(supabase, table: str, columns: str, stock_id: str) -> list[dict]:
    """
    All rows of one stock. (stock_id, date) is unique in prices_daily and
    features_daily, so date alone is a stable pagination order here.
    """
    return tpm.fetch_paginated(
        lambda: supabase.table(table).select(columns).eq("stock_id", stock_id)
        .order("date")
    )


def load_panel(supabase) -> pd.DataFrame:
    """Daily prices (with high/low) joined to stored features, every active ticker."""
    stocks = (
        supabase.table("stocks").select("id, ticker").eq("is_active", True)
        .execute().data
    )
    frames, fetched = [], {"prices_daily": [], "features_daily": []}
    for s in stocks:
        prices = pd.DataFrame(fetch_stock_rows(
            supabase, "prices_daily", "date, high, low, close", s["id"]
        ))
        feats = pd.DataFrame(fetch_stock_rows(
            supabase, "features_daily", FEATURE_SELECT, s["id"]
        ))
        fetched["prices_daily"].append(prices.assign(stock_id=s["id"]))
        fetched["features_daily"].append(feats.assign(stock_id=s["id"]))
        if prices.empty or feats.empty:
            logger.warning(f"{s['ticker']}: no prices or features, skipped")
            continue
        frame = prices.merge(feats, on="date", how="left")
        frame["ticker"] = s["ticker"]
        frames.append(frame)

    for table, parts in fetched.items():
        tpm.verify_stock_date_rows(
            supabase, pd.concat(parts, ignore_index=True), table, stocks
        )
    df = pd.concat(frames, ignore_index=True)
    numeric = ["high", "low", "close", *tpm.RAW_FETCH_COLUMNS,
               "news_count", "avg_sentiment_score"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    logger.info(
        f"Loaded {len(df)} daily rows for {df['ticker'].nunique()} tickers "
        f"({df['date'].min()} -> {df['date'].max()})"
    )
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward 5-day return r5, its market-relative version, and the same pair
    for the 5 trading days ending at t (known at t; used by Persistence).
    Expects rows sorted by ticker, then date.
    """
    by_ticker = df.groupby("ticker")
    df["r5"] = by_ticker["close"].shift(-HORIZON) / df["close"] - 1
    df["target_end"] = by_ticker["date"].shift(-HORIZON)
    df["r5_past"] = df["close"] / by_ticker["close"].shift(HORIZON) - 1
    # Median over every ticker with a return that date, before any filtering.
    for col, out in (("r5", "excess"), ("r5_past", "excess_past")):
        df[out] = df[col] - df.groupby("date")[col].transform("median")
    return df


def add_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Williams %R (14) and mean (high - low) / close (5), using data up to t."""
    by_ticker = df.groupby("ticker")
    hh = by_ticker["high"].transform(lambda s: s.rolling(WILLIAMS_WINDOW).max())
    ll = by_ticker["low"].transform(lambda s: s.rolling(WILLIAMS_WINDOW).min())
    df["williams_r_14"] = (-100 * (hh - df["close"]) / (hh - ll)).replace(
        [np.inf, -np.inf], np.nan
    )
    df["hl_range_5d"] = (
        ((df["high"] - df["low"]) / df["close"])
        .groupby(df["ticker"]).transform(lambda s: s.rolling(HL_WINDOW).mean())
    )
    return df


def add_sentiment_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    5-day FinBERT features inside the common news window. Rolling windows only
    span in-window days, so rows before enough news history stay NaN.
    """
    news, window = pilot.restrict_to_news_window(
        df[["ticker", "date", "news_count", "avg_sentiment_score"]]
    )
    news = news.sort_values(["ticker", "date"])
    by_ticker = news.groupby("ticker")
    news["sent_mean_5d"] = by_ticker["avg_sentiment_score"].transform(
        lambda s: s.rolling(SENT_WINDOW).mean()
    )
    news["news_count_5d"] = by_ticker["news_count"].transform(
        lambda s: s.rolling(SENT_WINDOW).sum()
    )
    # Baseline: the 60 trading days before the 5-day window, so the surprise
    # compares this week against the ticker's own past, not against itself.
    baseline = by_ticker["avg_sentiment_score"].transform(
        lambda s: s.shift(SENT_WINDOW)
        .rolling(SURPRISE_BASELINE_DAYS, min_periods=SURPRISE_MIN_HISTORY).mean()
    )
    news["sent_surprise"] = news["sent_mean_5d"] - baseline
    df = df.merge(news[["ticker", "date", *SENT_FEATURES]],
                  on=["ticker", "date"], how="left")
    return df, window


def build_rows(panel: pd.DataFrame, sampling: str, start: str, end: str,
               features: list[str]) -> pd.DataFrame:
    """Rows of one sampling inside [start, end] with every feature and target."""
    if sampling == "weekly":
        # Week ends come from the full calendar, so a run window that stops
        # mid-week never turns a Thursday into a "last trading day".
        week_ends = set(panel.groupby("week")["date"].max())
        rows = panel[panel["date"].isin(week_ends)]
    else:
        rows = panel
    rows = rows[(rows["date"] >= start) & (rows["date"] <= end)]
    rows = rows.dropna(subset=features + ["r5", "excess", "r5_past", "excess_past"])
    return rows.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Labels and splits
# ---------------------------------------------------------------------------

def apply_band_labels(rows: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """
    Label by the sign of the target return and drop the band nearest zero.
    The band is fixed from the first WF_INITIAL_TRAIN share of dates only.
    """
    spec = TARGETS[target]
    ret = rows[spec["return"]]
    dates = np.sort(rows["date"].unique())
    band_dates = dates[: int(len(dates) * WF_INITIAL_TRAIN)]
    in_band_period = rows["date"].isin(set(band_dates))
    band = float(ret[in_band_period].abs().quantile(spec["band_share"]))

    keep = ret.abs() > band
    labeled = rows[keep].copy()
    labeled[LABEL_COLUMN] = (labeled[spec["return"]] > 0).astype(int)
    labeled["prev_direction"] = (labeled[spec["past"]] > 0).astype(int)

    info = {
        "band_abs_return_pct": round(band * 100, 4),
        "band_chosen_on": [band_dates[0], band_dates[-1]],
        "rows_before_band": int(len(rows)),
        "rows_after_labeling": int(len(labeled)),
        "dropped_share_band_period": round(float(1 - keep[in_band_period].mean()), 4),
        "dropped_share_overall": round(float(1 - keep.mean()), 4),
        "up_share": round(float(labeled[LABEL_COLUMN].mean()), 4),
    }
    return labeled.reset_index(drop=True), info


def embargo(rows: pd.DataFrame, boundary: str) -> pd.DataFrame:
    """Rows whose target is realized before `boundary`, the next block's first date."""
    return rows[rows["target_end"] < boundary]


def date_split(df: pd.DataFrame) -> tuple[dict, dict]:
    """70/15/15 split on unique dates, embargoed at both boundaries."""
    dates = np.sort(df["date"].unique())
    val_start = dates[int(len(dates) * tpm.TRAIN_RATIO)]
    test_start = dates[int(len(dates) * (tpm.TRAIN_RATIO + tpm.VAL_RATIO))]
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    splits = {
        "train": embargo(train, val_start),
        "validation": embargo(val, test_start),
        "test": df[df["date"] >= test_start],
    }
    info = {}
    for name, part in splits.items():
        info[name] = {
            "rows": int(len(part)),
            "weeks": int(part["week"].nunique()),
            "first": part["date"].min(),
            "last": part["date"].max(),
            "up_share": round(float(part[LABEL_COLUMN].mean()), 4),
        }
    info["train"]["embargoed_rows"] = int(len(train) - len(splits["train"]))
    info["validation"]["embargoed_rows"] = int(len(val) - len(splits["validation"]))
    return splits, info


# ---------------------------------------------------------------------------
# Models and metrics
# ---------------------------------------------------------------------------

def train_regularized_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        **RF_PARAMS, random_state=tpm.RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def fit_models(X_train, y_train) -> dict:
    return {
        "Logistic Regression": tpm.train_logistic_regression(X_train, y_train),
        "Random Forest": train_regularized_forest(X_train, y_train),
    }


def score(y: np.ndarray, pred: np.ndarray, scores) -> dict:
    m = tpm.compute_metrics(y, pred)
    m["auc"] = tpm.compute_auc(y, scores)
    m["pred_up_share"] = round(float(np.mean(pred)), 4)
    m["n"] = int(len(y))
    return m


def baselines(train: pd.DataFrame, rows: pd.DataFrame) -> dict:
    p_up = float(train[LABEL_COLUMN].mean())
    rng = np.random.default_rng(tpm.RANDOM_SEED)
    y = rows[LABEL_COLUMN].values.astype(int)
    preds = tpm.baseline_predictions(rows, p_up, int(round(p_up)), rng)
    return {name: score(y, pred, None) for name, pred in preds.items()}


def evaluate_split(splits: dict, arms: dict) -> tuple[dict, dict, dict]:
    """Fit on train; score validation and test. No calibration or selection."""
    results, test_preds, fitted = {}, {}, {}
    for arm, features in arms.items():
        X_train, y_train = tpm.prepare_matrices(splits["train"], features)
        results[arm] = {}
        for name, model in fit_models(X_train, y_train).items():
            res = {}
            for split in ("validation", "test"):
                X, y = tpm.prepare_matrices(splits[split], features)
                pred = model.predict(X)
                res[split] = score(y, pred, tpm.positive_scores(model, X))
                if split == "test":
                    test_preds[(arm, name)] = pred
            results[arm][MODEL_SHORT[name]] = res
            fitted[(arm, name)] = model
    return results, test_preds, fitted


def walk_forward(df: pd.DataFrame, arms: dict):
    """
    Expanding-window walk-forward over unique dates. Each fold trains on the
    embargoed earlier rows and predicts the next block; out-of-sample
    predictions are pooled in date order.
    """
    dates = np.sort(df["date"].unique())
    blocks = np.array_split(dates[int(len(dates) * WF_INITIAL_TRAIN):], WF_FOLDS)

    parts, folds, preds, scores = [], [], {}, {}
    for i, block in enumerate(blocks, 1):
        train = embargo(df[df["date"] < block[0]], block[0])
        test = df[df["date"].isin(set(block))]
        parts.append(test)
        fold = {"fold": i, "test_start": block[0], "test_end": block[-1],
                "train_rows": int(len(train)), "test_rows": int(len(test)),
                "test_weeks": int(test["week"].nunique()), "mcc": {}}
        for arm, features in arms.items():
            X_train, y_train = tpm.prepare_matrices(train, features)
            X_test, y_test = tpm.prepare_matrices(test, features)
            for name, model in fit_models(X_train, y_train).items():
                pred = model.predict(X_test)
                preds.setdefault((arm, name), []).append(pred)
                scores.setdefault((arm, name), []).append(
                    tpm.positive_scores(model, X_test)
                )
                fold["mcc"][f"{arm}: {MODEL_SHORT[name]}"] = (
                    tpm.compute_metrics(y_test, pred)["mcc"]
                )
        folds.append(fold)

    oos = pd.concat(parts, ignore_index=True)
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    y = oos[LABEL_COLUMN].values.astype(int)
    pooled = {arm: {} for arm in arms}
    for (arm, name), pred in preds.items():
        pooled[arm][MODEL_SHORT[name]] = score(
            y, pred, np.concatenate(scores[(arm, name)])
        )
    first_train = df[df["date"] < blocks[0][0]]
    return oos, preds, pooled, folds, first_train


# ---------------------------------------------------------------------------
# Paired bootstrap over weeks
# ---------------------------------------------------------------------------

def confusion_by_week(y, pred, week_idx, n_weeks) -> np.ndarray:
    """Per-week (tp, fp, tn, fn) counts, shape (n_weeks, 4)."""
    cells = [(pred == 1) & (y == 1), (pred == 1) & (y == 0),
             (pred == 0) & (y == 0), (pred == 0) & (y == 1)]
    return np.column_stack([
        np.bincount(week_idx, weights=c, minlength=n_weeks) for c in cells
    ])


def mcc_from_counts(c: np.ndarray) -> np.ndarray:
    """MCC from (..., 4) confusion counts; 0 when undefined, like sklearn."""
    tp, fp, tn, fn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)


def paired_bootstrap(y, base_pred, arm_pred, weeks) -> dict:
    """
    Resample whole weeks with replacement and recompute both MCCs on the same
    resampled rows. Drawing per-week counts from a multinomial is the same as
    drawing weeks with replacement, but vectorized.
    """
    uniq, week_idx = np.unique(weeks, return_inverse=True)
    n = len(uniq)
    base = confusion_by_week(y, base_pred, week_idx, n)
    arm = confusion_by_week(y, arm_pred, week_idx, n)
    rng = np.random.default_rng(tpm.RANDOM_SEED)
    draws = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP_ITERATIONS)
    diff = mcc_from_counts(draws @ arm) - mcc_from_counts(draws @ base)
    low, high = np.percentile(diff, [2.5, 97.5])
    return {
        "observed": round(float(matthews_corrcoef(y, arm_pred)
                                - matthews_corrcoef(y, base_pred)), 4),
        "boot_mean": round(float(diff.mean()), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "share_above_zero": round(float((diff > 0).mean()), 4),
        "weeks": int(n),
    }


def compare_arms(rows: pd.DataFrame, preds: dict, arms: dict) -> dict:
    y = rows[LABEL_COLUMN].values.astype(int)
    weeks = rows["week"].values
    out = {}
    for arm in arms:
        if arm == "price":
            continue
        out[arm] = {
            MODEL_SHORT[name]: paired_bootstrap(
                y, preds[("price", name)], preds[(arm, name)], weeks
            )
            for name in MODEL_SHORT
        }
    return out


def sentiment_importance(model, test: pd.DataFrame, features: list[str]) -> dict:
    """Permutation importance on the test split, by AUC and by MCC."""
    X, y = tpm.prepare_matrices(test, features)
    out = {}
    for metric, scorer in (("auc", "roc_auc"), ("mcc", make_scorer(matthews_corrcoef))):
        r = permutation_importance(
            model, X, y, scoring=scorer, n_repeats=PERMUTATION_REPEATS,
            random_state=tpm.RANDOM_SEED,
        )
        order = np.argsort(-r.importances_mean)
        ranking = [
            {"rank": k, "feature": features[i],
             "mean": round(float(r.importances_mean[i]), 5),
             "std": round(float(r.importances_std[i]), 5)}
            for k, i in enumerate(order, 1)
        ]
        out[metric] = {
            "ranking": ranking,
            "sentiment_ranks": {f["feature"]: f["rank"] for f in ranking
                                if f["feature"] in SENT_FEATURES},
        }
    return out


# ---------------------------------------------------------------------------
# One evaluation: a run x target x sampling
# ---------------------------------------------------------------------------

def evaluate(rows: pd.DataFrame, target: str, arms: dict) -> dict:
    labeled, label_info = apply_band_labels(rows, target)
    splits, split_info = date_split(labeled)

    split_results, test_preds, fitted = evaluate_split(splits, arms)
    oos, wf_preds, wf_pooled, folds, wf_first_train = walk_forward(labeled, arms)

    result = {
        "labels": label_info,
        "splits": split_info,
        "walk_forward_oos": {"rows": int(len(oos)), "weeks": int(oos["week"].nunique()),
                             "first": oos["date"].min(), "last": oos["date"].max()},
        "baselines": {"test": baselines(splits["train"], splits["test"]),
                      "walk_forward": baselines(wf_first_train, oos)},
        "split_results": split_results,
        "walk_forward": {"pooled": wf_pooled, "folds": folds},
        "mcc_delta_vs_price": {
            "test": compare_arms(splits["test"], test_preds, arms),
            "walk_forward": compare_arms(oos, wf_preds, arms),
        },
    }
    if "price+sent" in arms:
        result["permutation_importance_price_sent_rf_test"] = sentiment_importance(
            fitted[("price+sent", "Random Forest")], splits["test"], arms["price+sent"]
        )
    return result


def log_evaluation(title: str, ev: dict) -> None:
    logger.info("\n" + "#" * 78)
    logger.info(title)
    logger.info("#" * 78)

    lab = ev["labels"]
    logger.info(
        f"Band: |return| > {lab['band_abs_return_pct']:.3f}% "
        f"(chosen on {lab['band_chosen_on'][0]} -> {lab['band_chosen_on'][1]}, "
        f"drops {lab['dropped_share_band_period'] * 100:.1f}% there, "
        f"{lab['dropped_share_overall'] * 100:.1f}% overall)"
    )
    logger.info(
        f"Rows after labeling: {lab['rows_after_labeling']} of "
        f"{lab['rows_before_band']} (UP share {lab['up_share'] * 100:.1f}%)"
    )
    for name, s in ev["splits"].items():
        extra = (f", embargoed {s['embargoed_rows']}"
                 if "embargoed_rows" in s else "")
        logger.info(
            f"  {name:<10} {s['rows']:>6} rows  {s['weeks']:>4} weeks  "
            f"{s['first']} -> {s['last']}  UP {s['up_share'] * 100:.1f}%{extra}"
        )
    wf = ev["walk_forward_oos"]
    logger.info(
        f"  {'WF OOS':<10} {wf['rows']:>6} rows  {wf['weeks']:>4} weeks  "
        f"{wf['first']} -> {wf['last']}"
    )

    table = []
    for scope, split_label in (("test", "Test"), ("walk_forward", "WalkFwd OOS")):
        for name, m in ev["baselines"][scope].items():
            table.append({"model": name, "split": split_label, "metrics": m})
        for arm in ev["split_results"]:
            for short in MODEL_SHORT.values():
                m = (ev["split_results"][arm][short]["test"] if scope == "test"
                     else ev["walk_forward"]["pooled"][arm][short])
                table.append({"model": f"{arm}: {short}", "split": split_label,
                              "metrics": m})
    tpm.print_comparison_table(table)

    folds = ev["walk_forward"]["folds"]
    keys = list(folds[0]["mcc"])
    logger.info("Walk-forward MCC per fold:")
    logger.info(f"  {'Fold':<4} {'Test dates':<23} {'Weeks':>5} {'Rows':>6}  "
                + "  ".join(f"{k:>15}" for k in keys))
    for f in folds:
        logger.info(
            f"  {f['fold']:<4} {f['test_start']} -> {f['test_end']} "
            f"{f['test_weeks']:>5} {f['test_rows']:>6}  "
            + "  ".join(f"{f['mcc'][k]:>+15.4f}" for k in keys)
        )

    logger.info("MCC delta vs price (paired week bootstrap, "
                f"{BOOTSTRAP_ITERATIONS} iterations):")
    logger.info(f"  {'Feature set':<11} {'Model':<5} {'Scope':<12} {'Observed':>9} "
                f"{'Boot mean':>10} {'95% CI':>19} {'P(>0)':>6} {'Weeks':>5}")
    for scope, label in (("test", "Test"), ("walk_forward", "WalkFwd OOS")):
        for arm, by_model in ev["mcc_delta_vs_price"][scope].items():
            for short, d in by_model.items():
                ci = f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                logger.info(
                    f"  {arm:<11} {short:<5} {label:<12} {d['observed']:>+9.4f} "
                    f"{d['boot_mean']:>+10.4f} {ci:>19} "
                    f"{d['share_above_zero']:>6.2f} {d['weeks']:>5}"
                )

    imp = ev.get("permutation_importance_price_sent_rf_test")
    if imp:
        n_features = len(imp["auc"]["ranking"])
        for metric in ("auc", "mcc"):
            ranks = ", ".join(f"{f} #{r}" for f, r in imp[metric]["sentiment_ranks"].items())
            top = ", ".join(f["feature"] for f in imp[metric]["ranking"][:3])
            logger.info(
                f"Permutation importance (price+sent RF, test, by {metric.upper()}): "
                f"sentiment ranks {ranks} of {n_features}; top 3: {top}"
            )


def log_headline(evaluations: dict) -> None:
    logger.info("\n" + "=" * 100)
    logger.info("Headline: walk-forward pooled MCC, and MCC delta vs price "
                "(bootstrap mean [95% CI])")
    logger.info("=" * 100)
    logger.info(f"{'Evaluation':<34} {'Feature set':<11} {'Model':<5} {'WF MCC':>7} "
                f"{'WF delta [95% CI]':>27} {'Test delta [95% CI]':>27}")
    logger.info("-" * 100)
    for key, ev in evaluations.items():
        label = key + (" *" if ev["primary"] else "")
        for arm, by_model in ev["walk_forward"]["pooled"].items():
            for short, m in by_model.items():
                deltas = []
                for scope in ("walk_forward", "test"):
                    d = ev["mcc_delta_vs_price"][scope].get(arm, {}).get(short)
                    deltas.append(
                        f"{d['boot_mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                        if d else "-"
                    )
                logger.info(f"{label:<34} {arm:<11} {short:<5} {m['mcc']:>+7.4f} "
                            f"{deltas[0]:>27} {deltas[1]:>27}")
                label = ""
    logger.info("-" * 100)
    logger.info("* primary (pre-registered): market-relative target, weekly rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison():
    logger.info("=" * 60)
    logger.info("Weekly-horizon comparison: price vs price+HL vs price+sent")
    logger.info("=" * 60)

    supabase = tpm.init_supabase()
    panel = tpm.derive_normalized_features(load_panel(supabase))
    market = tpm.fetch_market_features(panel["date"].min(), panel["date"].max())
    if market is None:
        logger.error("Market features unavailable; the price arm would not match "
                     "production. Aborting.")
        sys.exit(1)
    panel = panel.merge(market, on="date", how="left")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel = add_range_features(add_targets(panel))
    panel, news_window = add_sentiment_features(panel)
    panel["week"] = pd.to_datetime(panel["date"]).dt.to_period("W-FRI").astype(str)

    price = list(SCALE_FREE_FEATURES + DERIVED_FEATURES + MARKET_FEATURES)
    runs = {
        "a_full_history": {
            "start": panel["date"].min(), "end": panel["date"].max(),
            "arms": {"price": price, "price+HL": price + HL_FEATURES},
        },
        "b_finnhub_window": {
            "start": news_window["start"], "end": news_window["end"],
            "arms": {"price": price, "price+HL": price + HL_FEATURES,
                     "price+sent": price + SENT_FEATURES},
        },
    }

    evaluations = {}
    for run_name, run in runs.items():
        # Clean on the union of every arm's features so all arms share rows.
        union = list(dict.fromkeys(f for feats in run["arms"].values() for f in feats))
        for sampling in SAMPLINGS:
            rows = build_rows(panel, sampling, run["start"], run["end"], union)
            for target in TARGETS:
                key = f"{run_name}/{target}/{sampling}"
                logger.info(f"\nEvaluating {key} ({len(rows)} rows before labeling)...")
                ev = evaluate(rows, target, run["arms"])
                ev["primary"] = (target, sampling) == PRIMARY
                ev["rows_with_all_features"] = int(len(rows))
                log_evaluation(key, ev)
                evaluations[key] = ev

    log_headline(evaluations)

    report = {
        "description": "5-trading-day targets; price vs price+HL (full history) "
                       "and price vs price+HL vs price+sent (Finnhub window)",
        "primary": {"target": PRIMARY[0], "sampling": PRIMARY[1]},
        "config": {
            "horizon_trading_days": HORIZON,
            "relative_band_share": REL_BAND_SHARE,
            "absolute_band_share": ABS_BAND_SHARE,
            "band_chosen_on": f"first {WF_INITIAL_TRAIN:.0%} of each evaluation's dates",
            "embargo": "training rows need target_end < first date of the next "
                       "split / walk-forward block",
            "persistence": "sign of the 5 trading days ending at t (previous "
                           "week's label)",
            "williams_window": WILLIAMS_WINDOW,
            "hl_window": HL_WINDOW,
            "sentiment_window": SENT_WINDOW,
            "surprise_baseline_days": SURPRISE_BASELINE_DAYS,
            "surprise_baseline": "trading days t-64 .. t-5 (before the 5-day window)",
            "surprise_min_history": SURPRISE_MIN_HISTORY,
            "split_ratios": [tpm.TRAIN_RATIO, tpm.VAL_RATIO, tpm.TEST_RATIO],
            "walk_forward": {"initial_train_share": WF_INITIAL_TRAIN, "folds": WF_FOLDS},
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "permutation_repeats": PERMUTATION_REPEATS,
            "random_forest": RF_PARAMS,
            "logistic_regression": "StandardScaler + LogisticRegression(class_weight='balanced')",
            "calibration": "none (no model selection; both models reported)",
        },
        "news_window": news_window,
        "runs": {name: {"start": r["start"], "end": r["end"], "feature_sets": r["arms"]}
                 for name, r in runs.items()},
        "evaluations": evaluations,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    try:
        run_comparison()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
