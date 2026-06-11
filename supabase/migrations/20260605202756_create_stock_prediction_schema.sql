
/*
# AI Stock Prediction System — Core Schema

## Overview
Full relational schema for an AI-driven stock prediction and alert system. Designed to feed a
Python FastAPI backend, FinBERT sentiment pipeline, Streamlit dashboard, and an Alpaca
paper-trading module.

## New Tables

### 1. stocks
Master list of tracked tickers. Stores exchange, sector, industry metadata and an `is_active`
flag to soft-disable delisted or unwatched symbols.

### 2. prices_daily
Daily OHLCV (Open/High/Low/Close/Volume) + adjusted close for every tracked stock. One row per
stock per calendar date. Source field records the data provider (e.g. Alpha Vantage, Alpaca).
Composite unique constraint on (stock_id, date) prevents duplicate ingestion.

### 3. news_articles
Financial news headlines and full text tied to a specific stock. The `date` column is a
PostgreSQL GENERATED ALWAYS STORED column derived from `published_at`, ensuring date-level
aggregation is always consistent. Partial unique index on (stock_id, url) where url IS NOT NULL
prevents duplicate article ingestion while allowing null-URL items.

### 4. news_sentiment
FinBERT (or any NLP model) sentiment scores for each article. Stores raw positive/negative/neutral
probabilities, a dominant label, a compound score, and an optional embedding stored as JSONB
(compatible with plain arrays or future pgvector migration). One row per (news_id, model_name)
so multiple models can score the same article.

### 5. features_daily
One engineered feature row per stock per trading day. This is the table consumed by the ML
training pipeline. Includes price-based features (returns, volatility, MA, RSI, MACD) and
news-based sentiment aggregates. Binary classification target (0/1) for next-day direction and
a continuous return target for regression models.

### 6. models
Registry of trained model versions with full date split metadata (train/val/test) and key
classification metrics (accuracy, precision, recall, F1). `model_file_path` points to the
artifact stored externally (S3, local fs, MLflow, etc.).

### 7. predictions
One row per (stock, model, prediction_date, target_date). Stores raw up/down probabilities,
the predicted label, confidence, and an optional JSONB explanation payload (e.g. SHAP values).

### 8. alerts
Buy-watch / sell-risk / hold / no-action signals derived from predictions. Tracks whether the
alert has been dispatched and via which channel (email, Telegram, dashboard).

### 9. portfolio_positions
Single-tenant portfolio tracker. Each row is one (stock, status) pair — a stock can be both
'owned' and on the 'watchlist' as separate rows. Quantity and average buy price support basic
P&L calculations.

### 10. backtest_results
Stores aggregate statistics from strategy backtests or paper-trading runs keyed to a model and
stock. `strategy_rules` JSONB holds the exact rule set so results are reproducible.

## Infrastructure
- `update_updated_at_column()` trigger function updates `updated_at` automatically on stocks
  and portfolio_positions.
- Foreign key indexes on all FK columns to avoid sequential scans on joins.
- Composite and single-column indexes on the highest-cardinality query patterns.

## Important Notes
1. `news_articles.date` is a GENERATED STORED column — never set it manually.
2. The unique constraint on predictions (stock_id, model_id, prediction_date, target_date)
   has a known PostgreSQL behavior: two rows with NULL model_id are NOT treated as duplicates
   (NULL != NULL in unique indexes). Handle deduplication at the application layer if model_id
   can be null.
3. No user_id / auth.users references — this is a private single-tenant system.
*/

-- ============================================================
-- Shared trigger function for updated_at maintenance
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 1. stocks
-- ============================================================
CREATE TABLE IF NOT EXISTS stocks (
  id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  ticker         TEXT        UNIQUE NOT NULL,
  company_name   TEXT,
  exchange       TEXT,
  sector         TEXT,
  industry       TEXT,
  currency       TEXT        NOT NULL DEFAULT 'USD',
  is_active      BOOLEAN     NOT NULL DEFAULT TRUE,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DROP TRIGGER IF EXISTS stocks_updated_at ON stocks;
CREATE TRIGGER stocks_updated_at
  BEFORE UPDATE ON stocks
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 2. prices_daily
-- ============================================================
CREATE TABLE IF NOT EXISTS prices_daily (
  id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  stock_id       UUID        NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
  date           DATE        NOT NULL,
  open           NUMERIC,
  high           NUMERIC,
  low            NUMERIC,
  close          NUMERIC,
  adjusted_close NUMERIC,
  volume         BIGINT,
  source         TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (stock_id, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_daily_stock_date ON prices_daily (stock_id, date);
CREATE INDEX IF NOT EXISTS idx_prices_daily_date       ON prices_daily (date);

-- ============================================================
-- 3. news_articles
-- ============================================================
CREATE TABLE IF NOT EXISTS news_articles (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  stock_id     UUID        NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
  published_at TIMESTAMPTZ NOT NULL,
  date         DATE        GENERATED ALWAYS AS ((published_at AT TIME ZONE 'UTC')::DATE) STORED,
  headline     TEXT        NOT NULL,
  summary      TEXT,
  source       TEXT,
  url          TEXT,
  author       TEXT,
  raw_text     TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Partial unique index: no duplicate (stock, url) pairs when url is present
CREATE UNIQUE INDEX IF NOT EXISTS idx_news_articles_stock_url
  ON news_articles (stock_id, url)
  WHERE url IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_news_articles_stock_published ON news_articles (stock_id, published_at);
CREATE INDEX IF NOT EXISTS idx_news_articles_date            ON news_articles (date);
CREATE INDEX IF NOT EXISTS idx_news_articles_stock_id        ON news_articles (stock_id);

-- ============================================================
-- 4. news_sentiment
-- ============================================================
CREATE TABLE IF NOT EXISTS news_sentiment (
  id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  news_id          UUID        NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
  model_name       TEXT        NOT NULL DEFAULT 'FinBERT',
  positive_score   NUMERIC,
  negative_score   NUMERIC,
  neutral_score    NUMERIC,
  sentiment_label  TEXT        CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),
  sentiment_score  NUMERIC,
  embedding        JSONB,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (news_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_news_sentiment_label      ON news_sentiment (sentiment_label);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_created_at ON news_sentiment (created_at);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_news_id    ON news_sentiment (news_id);

-- ============================================================
-- 5. features_daily
-- ============================================================
CREATE TABLE IF NOT EXISTS features_daily (
  id                         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  stock_id                   UUID        NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
  date                       DATE        NOT NULL,
  return_1d                  NUMERIC,
  return_5d                  NUMERIC,
  return_20d                 NUMERIC,
  volatility_5d              NUMERIC,
  volatility_20d             NUMERIC,
  volume_change_1d           NUMERIC,
  moving_average_5d          NUMERIC,
  moving_average_20d         NUMERIC,
  rsi_14                     NUMERIC,
  macd                       NUMERIC,
  macd_signal                NUMERIC,
  news_count                 INTEGER     NOT NULL DEFAULT 0,
  avg_positive_sentiment     NUMERIC,
  avg_negative_sentiment     NUMERIC,
  avg_neutral_sentiment      NUMERIC,
  avg_sentiment_score        NUMERIC,
  target_next_day_direction  INTEGER     CHECK (target_next_day_direction IN (0, 1)),
  target_next_day_return     NUMERIC,
  created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (stock_id, date)
);

CREATE INDEX IF NOT EXISTS idx_features_daily_stock_date      ON features_daily (stock_id, date);
CREATE INDEX IF NOT EXISTS idx_features_daily_stock_id        ON features_daily (stock_id);
CREATE INDEX IF NOT EXISTS idx_features_daily_target_direction ON features_daily (target_next_day_direction);

-- ============================================================
-- 6. models
-- ============================================================
CREATE TABLE IF NOT EXISTS models (
  id                    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name            TEXT        NOT NULL,
  model_type            TEXT,
  description           TEXT,
  training_start_date   DATE,
  training_end_date     DATE,
  validation_start_date DATE,
  validation_end_date   DATE,
  test_start_date       DATE,
  test_end_date         DATE,
  accuracy              NUMERIC,
  precision_score       NUMERIC,
  recall_score          NUMERIC,
  f1_score              NUMERIC,
  model_file_path       TEXT,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 7. predictions
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
  id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  stock_id            UUID        NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
  model_id            UUID        REFERENCES models(id) ON DELETE SET NULL,
  prediction_date     DATE        NOT NULL,
  target_date         DATE        NOT NULL,
  probability_up      NUMERIC,
  probability_down    NUMERIC,
  predicted_direction INTEGER     CHECK (predicted_direction IN (0, 1)),
  confidence_score    NUMERIC,
  explanation         JSONB,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (stock_id, model_id, prediction_date, target_date)
);

CREATE INDEX IF NOT EXISTS idx_predictions_stock_prediction_date ON predictions (stock_id, prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_target_date           ON predictions (target_date);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_direction   ON predictions (predicted_direction);
CREATE INDEX IF NOT EXISTS idx_predictions_stock_id              ON predictions (stock_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_id              ON predictions (model_id);

-- ============================================================
-- 8. alerts
-- ============================================================
CREATE TABLE IF NOT EXISTS alerts (
  id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  stock_id         UUID        NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
  prediction_id    UUID        REFERENCES predictions(id) ON DELETE SET NULL,
  alert_date       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  alert_type       TEXT        CHECK (alert_type IN ('buy_watch', 'sell_risk', 'hold', 'no_action')),
  title            TEXT,
  message          TEXT,
  confidence_score NUMERIC,
  risk_level       TEXT        CHECK (risk_level IN ('low', 'medium', 'high')),
  is_sent          BOOLEAN     NOT NULL DEFAULT FALSE,
  sent_channel     TEXT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_stock_date    ON alerts (stock_id, alert_date);
CREATE INDEX IF NOT EXISTS idx_alerts_alert_type    ON alerts (alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_is_sent       ON alerts (is_sent);
CREATE INDEX IF NOT EXISTS idx_alerts_stock_id      ON alerts (stock_id);
CREATE INDEX IF NOT EXISTS idx_alerts_prediction_id ON alerts (prediction_id);

-- ============================================================
-- 9. portfolio_positions
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_positions (
  id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  stock_id          UUID        NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
  status            TEXT        NOT NULL CHECK (status IN ('owned', 'watchlist')),
  quantity          NUMERIC     NOT NULL DEFAULT 0,
  average_buy_price NUMERIC,
  notes             TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (stock_id, status)
);

DROP TRIGGER IF EXISTS portfolio_positions_updated_at ON portfolio_positions;
CREATE TRIGGER portfolio_positions_updated_at
  BEFORE UPDATE ON portfolio_positions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE INDEX IF NOT EXISTS idx_portfolio_positions_stock_id ON portfolio_positions (stock_id);

-- ============================================================
-- 10. backtest_results
-- ============================================================
CREATE TABLE IF NOT EXISTS backtest_results (
  id                    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id              UUID        REFERENCES models(id) ON DELETE SET NULL,
  stock_id              UUID        NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
  start_date            DATE,
  end_date              DATE,
  initial_capital       NUMERIC,
  final_capital         NUMERIC,
  total_return_percent  NUMERIC,
  max_drawdown_percent  NUMERIC,
  win_rate_percent      NUMERIC,
  number_of_trades      INTEGER,
  strategy_rules        JSONB,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_results_model_id ON backtest_results (model_id);
CREATE INDEX IF NOT EXISTS idx_backtest_results_stock_id ON backtest_results (stock_id);
