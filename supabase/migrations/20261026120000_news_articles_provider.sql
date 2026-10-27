-- ============================================================
-- news_articles: provider marker + full (stock_id, url) unique constraint
-- ============================================================
-- `source` holds the publisher (e.g. "Yahoo"), so the feed an article came
-- from is tracked separately in `provider` ('alphavantage' | 'finnhub'). This
-- keeps the two providers distinguishable for per-provider resume logic and
-- for checking they agree before mixing them into sentiment features.
--
-- The previous partial unique index (WHERE url IS NOT NULL) cannot be used as
-- a PostgREST upsert on_conflict target, so it is replaced by a plain UNIQUE
-- constraint. NULL urls remain distinct, so the semantics are unchanged.

ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS provider TEXT;

-- Every article stored before this migration came from Alpha Vantage.
UPDATE news_articles SET provider = 'alphavantage' WHERE provider IS NULL;

ALTER TABLE news_articles ALTER COLUMN provider SET NOT NULL;

DROP INDEX IF EXISTS idx_news_articles_stock_url;
ALTER TABLE news_articles
  ADD CONSTRAINT news_articles_stock_url_key UNIQUE (stock_id, url);

-- Supports the per-provider resume query (latest published_at per ticker).
CREATE INDEX IF NOT EXISTS idx_news_articles_provider_stock_published
  ON news_articles (provider, stock_id, published_at);
