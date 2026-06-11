
/*
# Row Level Security — AI Stock Prediction System

## Overview
Enables RLS on all 10 tables and creates permissive CRUD policies for the anon + authenticated
roles. This is a single-tenant private system with no per-user data isolation — the policies
keep RLS active (blocking direct psql/REST access without credentials) while allowing the
application layer (FastAPI service-role key, Streamlit anon-key dashboard) full read/write
access.

## Security Model
- Service-role key (FastAPI backend, FinBERT pipeline, Alpaca module) bypasses RLS entirely.
- Anon-key clients (Streamlit dashboard, local dev) are granted full CRUD through these policies.
- Direct database access without any key is blocked by RLS.
- No user_id scoping — all rows are globally accessible within the project.

## Tables Modified
All 10 tables: stocks, prices_daily, news_articles, news_sentiment, features_daily, models,
predictions, alerts, portfolio_positions, backtest_results.

## Policy Convention
One policy per CRUD verb (SELECT / INSERT / UPDATE / DELETE) per table, named
`<verb>_<table>`. USING (true) / WITH CHECK (true) is intentional for a single-tenant system
where all data belongs to the same project/operator.
*/

-- ============================================================
-- stocks
-- ============================================================
ALTER TABLE stocks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_stocks" ON stocks;
CREATE POLICY "select_stocks" ON stocks FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_stocks" ON stocks;
CREATE POLICY "insert_stocks" ON stocks FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_stocks" ON stocks;
CREATE POLICY "update_stocks" ON stocks FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_stocks" ON stocks;
CREATE POLICY "delete_stocks" ON stocks FOR DELETE TO anon, authenticated USING (true);

-- ============================================================
-- prices_daily
-- ============================================================
ALTER TABLE prices_daily ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_prices_daily" ON prices_daily;
CREATE POLICY "select_prices_daily" ON prices_daily FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_prices_daily" ON prices_daily;
CREATE POLICY "insert_prices_daily" ON prices_daily FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_prices_daily" ON prices_daily;
CREATE POLICY "update_prices_daily" ON prices_daily FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_prices_daily" ON prices_daily;
CREATE POLICY "delete_prices_daily" ON prices_daily FOR DELETE TO anon, authenticated USING (true);

-- ============================================================
-- news_articles
-- ============================================================
ALTER TABLE news_articles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_news_articles" ON news_articles;
CREATE POLICY "select_news_articles" ON news_articles FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_news_articles" ON news_articles;
CREATE POLICY "insert_news_articles" ON news_articles FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_news_articles" ON news_articles;
CREATE POLICY "update_news_articles" ON news_articles FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_news_articles" ON news_articles;
CREATE POLICY "delete_news_articles" ON news_articles FOR DELETE TO anon, authenticated USING (true);

-- ============================================================
-- news_sentiment
-- ============================================================
ALTER TABLE news_sentiment ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_news_sentiment" ON news_sentiment;
CREATE POLICY "select_news_sentiment" ON news_sentiment FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_news_sentiment" ON news_sentiment;
CREATE POLICY "insert_news_sentiment" ON news_sentiment FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_news_sentiment" ON news_sentiment;
CREATE POLICY "update_news_sentiment" ON news_sentiment FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_news_sentiment" ON news_sentiment;
CREATE POLICY "delete_news_sentiment" ON news_sentiment FOR DELETE TO anon, authenticated USING (true);

-- ============================================================
-- features_daily
-- ============================================================
ALTER TABLE features_daily ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_features_daily" ON features_daily;
CREATE POLICY "select_features_daily" ON features_daily FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_features_daily" ON features_daily;
CREATE POLICY "insert_features_daily" ON features_daily FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_features_daily" ON features_daily;
CREATE POLICY "update_features_daily" ON features_daily FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_features_daily" ON features_daily;
CREATE POLICY "delete_features_daily" ON features_daily FOR DELETE TO anon, authenticated USING (true);

-- ============================================================
-- models
-- ============================================================
ALTER TABLE models ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_models" ON models;
CREATE POLICY "select_models" ON models FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_models" ON models;
CREATE POLICY "insert_models" ON models FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_models" ON models;
CREATE POLICY "update_models" ON models FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_models" ON models;
CREATE POLICY "delete_models" ON models FOR DELETE TO anon, authenticated USING (true);

-- ============================================================
-- predictions
-- ============================================================
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_predictions" ON predictions;
CREATE POLICY "select_predictions" ON predictions FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_predictions" ON predictions;
CREATE POLICY "insert_predictions" ON predictions FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_predictions" ON predictions;
CREATE POLICY "update_predictions" ON predictions FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_predictions" ON predictions;
CREATE POLICY "delete_predictions" ON predictions FOR DELETE TO anon, authenticated USING (true);

-- ============================================================
-- alerts
-- ============================================================
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_alerts" ON alerts;
CREATE POLICY "select_alerts" ON alerts FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_alerts" ON alerts;
CREATE POLICY "insert_alerts" ON alerts FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_alerts" ON alerts;
CREATE POLICY "update_alerts" ON alerts FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_alerts" ON alerts;
CREATE POLICY "delete_alerts" ON alerts FOR DELETE TO anon, authenticated USING (true);

-- ============================================================
-- portfolio_positions
-- ============================================================
ALTER TABLE portfolio_positions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_portfolio_positions" ON portfolio_positions;
CREATE POLICY "select_portfolio_positions" ON portfolio_positions FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_portfolio_positions" ON portfolio_positions;
CREATE POLICY "insert_portfolio_positions" ON portfolio_positions FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_portfolio_positions" ON portfolio_positions;
CREATE POLICY "update_portfolio_positions" ON portfolio_positions FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_portfolio_positions" ON portfolio_positions;
CREATE POLICY "delete_portfolio_positions" ON portfolio_positions FOR DELETE TO anon, authenticated USING (true);

-- ============================================================
-- backtest_results
-- ============================================================
ALTER TABLE backtest_results ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_backtest_results" ON backtest_results;
CREATE POLICY "select_backtest_results" ON backtest_results FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "insert_backtest_results" ON backtest_results;
CREATE POLICY "insert_backtest_results" ON backtest_results FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "update_backtest_results" ON backtest_results;
CREATE POLICY "update_backtest_results" ON backtest_results FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "delete_backtest_results" ON backtest_results;
CREATE POLICY "delete_backtest_results" ON backtest_results FOR DELETE TO anon, authenticated USING (true);
