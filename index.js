import { createClient } from "@supabase/supabase-js";
import "dotenv/config";

const supabase = createClient(
  process.env.VITE_SUPABASE_URL,
  process.env.VITE_SUPABASE_ANON_KEY
);

// ---------------------------------------------------------------------------
// 1. Latest price for every active stock
//    Supabase JS does not support DISTINCT ON natively; use the raw SQL below
//    via supabase.rpc() once you create a matching DB function, or run it
//    directly in psql / the Supabase SQL editor.
//
// SELECT DISTINCT ON (p.stock_id)
//   s.ticker, s.company_name,
//   p.date, p.open, p.high, p.low, p.close, p.adjusted_close, p.volume, p.source
// FROM prices_daily p
// JOIN stocks s ON s.id = p.stock_id
// WHERE s.is_active = TRUE
// ORDER BY p.stock_id, p.date DESC;
// ---------------------------------------------------------------------------
export async function getLatestPrices() {
  const { data, error } = await supabase.rpc("get_latest_prices_per_stock");
  if (error) throw error;
  return data;
}

// ---------------------------------------------------------------------------
// 2. Latest prediction for a given ticker
// ---------------------------------------------------------------------------
export async function getLatestPrediction(ticker) {
  const { data: stock, error: stockErr } = await supabase
    .from("stocks")
    .select("id")
    .eq("ticker", ticker.toUpperCase())
    .maybeSingle();

  if (stockErr) throw stockErr;
  if (!stock) throw new Error(`Ticker not found: ${ticker}`);

  const { data, error } = await supabase
    .from("predictions")
    .select(`
      id,
      prediction_date,
      target_date,
      probability_up,
      probability_down,
      predicted_direction,
      confidence_score,
      explanation,
      models (model_name, model_type, accuracy, f1_score)
    `)
    .eq("stock_id", stock.id)
    .order("prediction_date", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error) throw error;
  return data;
}

// Raw SQL equivalent:
//
// SELECT p.prediction_date, p.target_date, p.probability_up, p.probability_down,
//        p.predicted_direction, p.confidence_score, p.explanation,
//        m.model_name, m.model_type, m.accuracy, m.f1_score
// FROM predictions p
// LEFT JOIN models m ON m.id = p.model_id
// WHERE p.stock_id = (SELECT id FROM stocks WHERE ticker = 'AAPL')
// ORDER BY p.prediction_date DESC
// LIMIT 1;

// ---------------------------------------------------------------------------
// 3. All active (unsent) alerts, newest first
// ---------------------------------------------------------------------------
export async function getActiveAlerts() {
  const { data, error } = await supabase
    .from("alerts")
    .select(`
      id,
      alert_date,
      alert_type,
      title,
      message,
      confidence_score,
      risk_level,
      sent_channel,
      stocks (ticker, company_name)
    `)
    .eq("is_sent", false)
    .order("alert_date", { ascending: false });

  if (error) throw error;
  return data;
}

// Raw SQL equivalent:
//
// SELECT a.id, a.alert_date, a.alert_type, a.title, a.message,
//        a.confidence_score, a.risk_level, a.sent_channel,
//        s.ticker, s.company_name
// FROM alerts a
// JOIN stocks s ON s.id = a.stock_id
// WHERE a.is_sent = FALSE
// ORDER BY a.alert_date DESC;

// ---------------------------------------------------------------------------
// 4. Feature rows for ML model training
//    Returns only labelled rows (target_next_day_direction IS NOT NULL).
//    Optionally filter by ticker and/or date range.
// ---------------------------------------------------------------------------
export async function getTrainingFeatures({ ticker, startDate, endDate } = {}) {
  let query = supabase
    .from("features_daily")
    .select(`
      date,
      return_1d, return_5d, return_20d,
      volatility_5d, volatility_20d,
      volume_change_1d,
      moving_average_5d, moving_average_20d,
      rsi_14, macd, macd_signal,
      news_count,
      avg_positive_sentiment, avg_negative_sentiment,
      avg_neutral_sentiment, avg_sentiment_score,
      target_next_day_direction, target_next_day_return,
      stocks (ticker)
    `)
    .not("target_next_day_direction", "is", null)
    .order("date", { ascending: true });

  if (ticker) {
    const { data: stock } = await supabase
      .from("stocks")
      .select("id")
      .eq("ticker", ticker.toUpperCase())
      .maybeSingle();
    if (stock) query = query.eq("stock_id", stock.id);
  }
  if (startDate) query = query.gte("date", startDate);
  if (endDate) query = query.lte("date", endDate);

  const { data, error } = await query;
  if (error) throw error;
  return data;
}

// Raw SQL equivalent:
//
// SELECT s.ticker, f.date,
//        f.return_1d, f.return_5d, f.return_20d,
//        f.volatility_5d, f.volatility_20d, f.volume_change_1d,
//        f.moving_average_5d, f.moving_average_20d,
//        f.rsi_14, f.macd, f.macd_signal,
//        f.news_count,
//        f.avg_positive_sentiment, f.avg_negative_sentiment,
//        f.avg_neutral_sentiment, f.avg_sentiment_score,
//        f.target_next_day_direction, f.target_next_day_return
// FROM features_daily f
// JOIN stocks s ON s.id = f.stock_id
// WHERE f.target_next_day_direction IS NOT NULL
//   -- AND s.ticker = 'AAPL'
//   -- AND f.date BETWEEN '2020-01-01' AND '2024-12-31'
// ORDER BY f.date ASC;

// ---------------------------------------------------------------------------
// 5. Average FinBERT sentiment per stock per day
//    Aggregated SQL is more useful for dashboards — run it directly in psql
//    or via supabase.rpc() after wrapping in a DB function.
//
// SELECT s.ticker, na.date,
//        COUNT(ns.id)                              AS article_count,
//        ROUND(AVG(ns.positive_score)::numeric, 4) AS avg_positive,
//        ROUND(AVG(ns.negative_score)::numeric, 4) AS avg_negative,
//        ROUND(AVG(ns.neutral_score)::numeric, 4)  AS avg_neutral,
//        ROUND(AVG(ns.sentiment_score)::numeric, 4) AS avg_compound,
//        MODE() WITHIN GROUP (ORDER BY ns.sentiment_label) AS dominant_label
// FROM news_articles na
// JOIN stocks s          ON s.id = na.stock_id
// JOIN news_sentiment ns ON ns.news_id = na.id
// -- WHERE s.ticker = 'NVDA'
// -- AND na.date >= '2024-01-01'
// GROUP BY s.ticker, na.date
// ORDER BY na.date DESC;
// ---------------------------------------------------------------------------
export async function getDailySentiment({ ticker, startDate, endDate } = {}) {
  let query = supabase
    .from("news_articles")
    .select(`
      date,
      stock_id,
      stocks (ticker, company_name),
      news_sentiment (
        positive_score,
        negative_score,
        neutral_score,
        sentiment_score,
        sentiment_label
      )
    `)
    .order("date", { ascending: false });

  if (ticker) {
    const { data: stock } = await supabase
      .from("stocks")
      .select("id")
      .eq("ticker", ticker.toUpperCase())
      .maybeSingle();
    if (stock) query = query.eq("stock_id", stock.id);
  }
  if (startDate) query = query.gte("date", startDate);
  if (endDate) query = query.lte("date", endDate);

  const { data, error } = await query;
  if (error) throw error;
  return data;
}

// ---------------------------------------------------------------------------
// Demo runner
// ---------------------------------------------------------------------------
async function main() {
  console.log("=== 2. Latest prediction for AAPL ===");
  const pred = await getLatestPrediction("AAPL");
  console.log(pred ?? "No predictions yet — ingest price data and run the model first.");

  console.log("\n=== 3. Active (unsent) alerts ===");
  const activeAlerts = await getActiveAlerts();
  console.log(`${activeAlerts.length} active alert(s) pending dispatch`);

  console.log("\n=== 4. Training features for NVDA ===");
  const features = await getTrainingFeatures({ ticker: "NVDA" });
  console.log(`${features.length} labelled feature row(s) available for training`);

  console.log("\n=== 5. Daily sentiment articles for TSLA ===");
  const sentiment = await getDailySentiment({ ticker: "TSLA" });
  console.log(`${sentiment.length} news article(s) with sentiment scores`);
}

main().catch(console.error);
