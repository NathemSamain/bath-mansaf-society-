# AI Stock Prediction System

PostgreSQL/Supabase schema + ingestion pipeline for AI-powered stock predictions.

## Database Schema

10 tables designed for:
- Historical daily OHLCV prices
- News articles and sentiment scoring
- Feature engineering for ML models
- Model training registry
- Predictions and alerts
- Portfolio tracking

See `supabase/migrations/` for full schema definition.

---

## Price Ingestion Pipeline

### Setup

1. **Get Alpha Vantage API key** (free)
   - Visit: https://www.alphavantage.co/support/#api-key
   - Sign up for a free API key

2. **Set environment variables**

   Create a `.env` file in the project root:

   ```bash
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
   ```

   Or export them in your shell:

   ```bash
   export SUPABASE_URL=https://your-project.supabase.co
   export SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   export ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
   ```

   **Important:**
- Use the `service_role` key (not `anon`) for the Python ingestion script
- Get your keys from: Supabase Dashboard → Settings → API

3. **Install Python dependencies**

   ```bash
   pip install requests supabase
   ```

### Run the Ingestion Script

```bash
python scripts/ingest_prices_daily.py
```

The script will:
- Read all active tickers from the `stocks` table
- Fetch full daily adjusted history from Alpha Vantage
- Upsert into `prices_daily` (avoiding duplicates)
- Wait 12s between API calls (free tier rate limit)

### Verify Data Was Inserted

**Option 1: Check record count**

```sql
SELECT
  s.ticker,
  COUNT(*) as price_records,
  MIN(p.date) as earliest_date,
  MAX(p.date) as latest_date
FROM prices_daily p
JOIN stocks s ON s.id = p.stock_id
GROUP BY s.ticker
ORDER BY s.ticker;
```

**Option 2: View latest price for each stock**

```sql
SELECT
  s.ticker,
  p.date,
  p.open,
  p.high,
  p.low,
  p.close,
  p.adjusted_close,
  p.volume
FROM prices_daily p
JOIN stocks s ON s.id = p.stock_id
WHERE p.date = (
  SELECT MAX(date) FROM prices_daily WHERE stock_id = s.id
)
ORDER BY s.ticker;
```

**Option 3: Check ingestion metadata**

```sql
SELECT
  source,
  COUNT(*) as records,
  MIN(date) as earliest,
  MAX(date) as latest
FROM prices_daily
GROUP BY source;
```

---

## Stage 2 — News & Sentiment

Adds news-based sentiment features on top of the price-only baseline. Two
scripts, run after `build_features_daily.py`:

```bash
python scripts/ingest_news.py              # Alpha Vantage -> news_articles + news_sentiment
python scripts/build_sentiment_features.py # aggregate -> features_daily sentiment columns
```

**`ingest_news.py`** pulls news per active ticker from the Alpha Vantage
`NEWS_SENTIMENT` endpoint into `news_articles`, and AV's per-ticker sentiment
score into `news_sentiment` (`model_name='AlphaVantage'`). It is resumable
(continues from the newest stored article, de-dupes by URL) and free-tier aware
(stops cleanly at the ~25 requests/day cap — just re-run on subsequent days to
backfill more). By default it fetches the **last 12 months** (the window that
overlaps validation/test); override with `NEWS_LOOKBACK_DAYS` (e.g. a large
value to backfill all available history, ~2022+). Requires
`ALPHA_VANTAGE_API_KEY` in `.env`.

**`build_sentiment_features.py`** aggregates stored sentiment into the
`features_daily` columns (`news_count`, `avg_sentiment_score`) using a
**leakage-safe, StockNet-style alignment**: an article is attached to the first
trading day whose close (~16:00 ET) falls on/after the article's timestamp, so a
feature row for day D only ever sees news a trader could have known before the
D → D+1 prediction.

> Roadmap: the schema supports multiple sentiment rows per article, so a later
> pass can add a **Finnhub** news provider and recompute sentiment with
> **FinBERT** (`model_name='FinBERT'`) without disturbing the AV rows. After
> ingesting news, the sentiment columns are added to the model's feature set and
> `train_price_model.py` is rerun to test whether MCC finally clears the
> baselines out of sample.

---

## Price Ingestion (yfinance — real data, no API key)

`scripts/ingest_prices_yfinance.py` is a free alternative to the Alpha Vantage
script. It pulls **real** daily OHLCV from Yahoo Finance for every active ticker
in `stocks` and **replaces** any existing rows for those tickers (e.g. the
synthetic `generated` seed data) with real prices tagged `source = 'yfinance'`.
The `prices_daily` schema is unchanged.

```bash
pip install -r requirements.txt          # installs yfinance
python scripts/ingest_prices_yfinance.py
```

It needs only `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` (no API key, no
daily request cap). History defaults to 5 years; override with `YF_PERIOD`
(e.g. `max`) or `YF_START_DATE` (e.g. `2015-01-01`). After ingesting, rerun
`build_features_daily.py` then `train_price_model.py`.

---

## Feature Engineering Pipeline

### Overview

The feature engineering script reads historical prices from `prices_daily` and calculates
technical indicators for ML model training:

**Price-based features:**
- `return_1d`, `return_5d`, `return_20d` - Percentage returns
- `volatility_5d`, `volatility_20d` - Rolling standard deviation of returns
- `volume_change_1d` - Day-over-day volume change
- `moving_average_5d`, `moving_average_20d` - Simple moving averages
- `rsi_14` - Relative Strength Index (14-period)
- `macd`, `macd_signal` - MACD indicator

**Supervised learning targets:**
- `target_next_day_direction` - 1 if next day close > current, else 0
- `target_next_day_return` - Percentage return to next day

**(Note: Sentiment features are placeholders for future news integration)**

### Run the Script

```bash
python scripts/build_features_daily.py
```

The script will:
- Read all active stocks from the `stocks` table
- Fetch historical prices for each stock
- Calculate technical indicators requiring 20+ days of history
- Calculate next-day prediction targets
- Upsert into `features_daily` (avoiding duplicates)

### Verify Features in Supabase

**Check feature count by stock:**

```sql
SELECT
  s.ticker,
  COUNT(*) as feature_rows,
  MIN(f.date) as earliest_date,
  MAX(f.date) as latest_date
FROM features_daily f
JOIN stocks s ON s.id = f.stock_id
GROUP BY s.ticker
ORDER BY s.ticker;
```

**View latest feature row for each stock:**

```sql
SELECT
  s.ticker,
  f.date,
  f.close,
  f.return_1d,
  f.rsi_14,
  f.macd,
  f.target_next_day_direction
FROM (
  SELECT
    stock_id,
    date,
    return_1d,
    rsi_14,
    macd,
    target_next_day_direction,
    ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY date DESC) as rn
  FROM features_daily
) f
JOIN stocks s ON s.id = f.stock_id
WHERE f.rn = 1
ORDER BY s.ticker;
```

---

## Model Training Pipeline

### Overview

The training script reads features from `features_daily`, trains two baseline models,
and saves the best one for making predictions.

**Models trained:**
- Logistic Regression
- Random Forest Classifier

**Features used:**
- Scale-free price/momentum: `return_1d`, `return_5d`, `return_20d`,
  `volatility_5d`, `volatility_20d`, `volume_change_1d`, `rsi_14`
- Normalized trend/MACD ratios (derived, so a $20 and a $500 stock are
  comparable): `ma_5_vs_20`, `macd_norm`, `macd_signal_norm`, `macd_hist_norm`
- Market-wide context (fetched from Yahoo Finance, joined by date, same-day /
  no lookahead; skipped automatically if unavailable): `mkt_spy_ret_1d`,
  `mkt_spy_ret_5d`, `mkt_qqq_ret_1d`, `mkt_vix`, `mkt_vix_chg_1d`

> Raw dollar-level moving averages and MACD are intentionally **not** fed to the
> model — they're replaced by the ratios above so it learns shape, not price tier.

**Target (label thresholding):**
- The label is built from `target_next_day_return` (stored in percent), **not** from
  the raw `target_next_day_direction` column:
  - return **> +0.5%** → `1` (UP)
  - return **< −0.5%** → `0` (DOWN)
  - return **between −0.5% and +0.5%** → dropped as *noise*
- This follows the StockNet convention: near-flat days carry little signal and
  mostly add label noise, so they are excluded from training and evaluation.

**Time-based split (by unique date):**
- 70% oldest trading days → training
- 15% → validation
- 15% newest trading days → testing
- Splitting on dates (not raw rows) keeps every ticker from the same day in the
  same split — the correct grouping for pooled multi-ticker data.

**Trustworthy-evaluation upgrades:** see
[Trustworthy Evaluation](#trustworthy-evaluation) below for the full rationale.

### Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib supabase python-dotenv
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### Run the Training Script

```bash
python scripts/train_price_model.py
```

The script will:
- Load features from `features_daily` table (including `target_next_day_return`)
- Clean rows with missing values
- Apply label thresholding (drop the ±0.5% noise band)
- Split data chronologically (no data leakage)
- Evaluate three naive baselines (Always-UP, Persistence, Random)
- Train Logistic Regression and Random Forest, then calibrate their probabilities
- Run the lag-mimic diagnostic on each model
- Select best model by **validation MCC** (F1 as tie-breaker)
- Evaluate best model on held-out test set
- Print a single comparison table across baselines and models
- Save model to `models/price_direction_model.joblib`
- Save metadata to `models/price_direction_model_metadata.json`

### Output Metrics

The script reports for **every** model and baseline:
- Accuracy
- Precision
- Recall
- F1 Score
- **MCC** (Matthews Correlation Coefficient)

**Example output:**
```
Label thresholding (UP > 0.5%, DOWN < -0.5%):
  Rows removed as noise (flat band): 286
  Final rows used for training/eval: 952
  Class balance: UP=490 (51.5%), DOWN=462 (48.5%)

Best model: Random Forest (val MCC = 0.0412, val F1 = 0.5180, calibrated = True)

==========================================================================
Comparison Table
==========================================================================
Model                  | Split       | Accuracy | Precision | Recall |     F1 |     MCC
--------------------------------------------------------------------------
Always-UP              | Validation  |   0.5315 |    0.5315 | 1.0000 | 0.6941 |  0.0000
Persistence            | Validation  |   0.5105 |    0.5395 | 0.5395 | 0.5395 |  0.0171
Random                 | Validation  |   0.4895 |    0.5200 | 0.5132 | 0.5166 | -0.0241
Logistic Regression    | Validation  |   0.5245 |    0.5310 | 0.5400 | 0.5354 |  0.0250
Random Forest          | Validation  |   0.5420 |    0.5410 | 0.5180 | 0.5292 |  0.0412
Best (Random Forest)   | Test        |   0.5380 |    0.5360 | 0.5100 | 0.5226 |  0.0380
==========================================================================
```

> Numbers above are illustrative. On real data, treat **MCC near 0** as
> "no real edge over chance" and any **>0.85 accuracy as a leakage bug**.

### Saved Model Files

After training, check:

```bash
ls models/
# price_direction_model.joblib
# price_direction_model_metadata.json
```

**Metadata JSON includes:**
```json
{
  "model_name": "Random Forest",
  "feature_columns": ["return_1d", "return_5d", "..."],
  "label_threshold_up": 0.5,
  "label_threshold_down": -0.5,
  "rows_loaded": 1240,
  "rows_after_cleaning": 1235,
  "rows_after_thresholding": 952,
  "class_balance": { "up": 490, "down": 462, "up_ratio": 0.5147 },
  "training_rows": 666,
  "validation_rows": 143,
  "test_rows": 143,
  "validation_metrics": { "accuracy": 0.542, "precision": 0.541, "recall": 0.518, "f1": 0.5292, "mcc": 0.0412 },
  "test_metrics": { "accuracy": 0.538, "precision": 0.536, "recall": 0.51, "f1": 0.5226, "mcc": 0.038 },
  "baseline_metrics": { "validation": { "Always-UP": {}, "Persistence": {}, "Random": {} }, "test": {} },
  "lag_mimic_diagnostic": { "best_model_test_agreement": 0.61, "random_forest_validation_agreement": 0.58 },
  "calibrated": true,
  "created_at": "2026-06-16T..."
}
```

---

## Trustworthy Evaluation

Predicting next-day direction is *hard* — published price-only ceilings sit
around ~58% accuracy, and a lot of headline results in the literature are
false positives. This pipeline bakes in several checks so we can tell a real
edge apart from an artifact.

### Label thresholding

Instead of labeling every up-tick as UP and every down-tick as DOWN, we build
the label from `target_next_day_return` and **drop moves inside ±0.5%**:

- `return > +0.5%` → UP (1)
- `return < −0.5%` → DOWN (0)
- otherwise → dropped as noise

Tiny moves are mostly microstructure noise; including them forces the model to
"explain" randomness and inflates apparent difficulty. Dropping them (the
StockNet convention) yields cleaner, more learnable labels and a more honest
class balance. The script logs rows loaded, rows dropped for missing values,
rows dropped as noise, the final count, and the resulting class balance.

### MCC (Matthews Correlation Coefficient)

Accuracy and F1 are misleading on imbalanced or trivially-predicted targets —
an "always UP" model can score high F1 while learning nothing. **MCC** uses all
four confusion-matrix quadrants and ranges from −1 to +1, where **0 means no
correlation with the truth (chance)**. We report it for every model and
baseline, and select the best model by **validation MCC** (F1 as tie-breaker).
A model that doesn't clear the baselines on MCC has no demonstrated edge.

### Naive baselines

Every model is compared against three baselines on both validation and test:

- **Always-UP** — always predicts 1. Exposes class imbalance (high F1, MCC ≈ 0).
- **Persistence** — predicts the *previous* trading day's actual direction.
  This is the bar a "real" model must beat; markets have weak day-to-day
  autocorrelation, so persistence is a surprisingly tough baseline.
- **Random** — predicts 0/1 using the training set's class balance.

If a model can't beat these, its "accuracy" is an illusion.

### Lag-mimic diagnostic

A classic failure mode (Radfar): models that look accurate but are really just
**copying yesterday's direction one day late**. For each model we compute the
**lag-mimic agreement** — the share of predictions equal to the previous day's
actual direction. If a model's accuracy is within ~2 points of the persistence
baseline **and** its lag-mimic agreement exceeds 90%, the script logs:

```
WARNING: <model> may be copying yesterday's direction rather than learning
useful predictive structure.
```

### Class-collapse diagnostic

The most common Stage-1 failure is a model that quietly predicts **one class for
everything** (e.g. "always UP"), riding the slight class imbalance to a
deceptively high F1 while its MCC sits at ~0. For each model the script logs its
predicted class balance and, if it predicts a single class more than 90% of the
time, warns:

```
WARNING: <model> collapsed to the majority class (predicts UP 100% of the
time) - it is likely riding class imbalance, not learning. Trust MCC over F1 here.
```

The dominant-class share is recorded in metadata under `class_collapse_diagnostic`.

### Probability calibration

Downstream alerting needs trustworthy *probabilities*, not just labels. Raw
classifier scores (especially Random Forest) are often poorly calibrated. We
wrap the fitted model in `CalibratedClassifierCV` (Platt/sigmoid scaling) and
fit the calibrator on the **validation set**, so `probability_up` /
`confidence_score` better reflect real-world frequencies. The metadata records
whether the saved model was calibrated. (The calibrated model still exposes
`predict` / `predict_proba`, so prediction generation is unchanged.)

> Note: calibration is fit on the validation set and selection also uses
> validation MCC, so validation figures are mildly optimistic — the **test**
> numbers are the honest read.

### Why this matters

Together these turn "the model got 54%" into a defensible claim: it clears the
naive baselines on MCC, it isn't just echoing yesterday, and its probabilities
mean something. Equally important, they make it obvious when there's **no edge**
(MCC ≈ 0) — which on a 5-ticker, one-year, mega-cap-tech window is the result
to expect and report honestly, rather than chasing a leakage-driven 90%.

---

## Prediction Generation Pipeline

### Overview

The prediction script loads the trained model and generates next-day direction
predictions for all active stocks based on their latest feature values.

**Process:**
- Load model from `models/price_direction_model.joblib`
- Load feature columns from metadata
- For each active stock, get latest features from `features_daily`
- Generate probability prediction using `predict_proba`
- Store prediction in `predictions` table

**Prediction outputs:**
- `predicted_direction` - 1 (UP) or 0 (DOWN) based on probability >= 0.5
- `probability_up` - Model's confidence the price will go up
- `probability_down` - Model's confidence the price will go down
- `confidence_score` - Maximum of the two probabilities

### Run the Prediction Script

```bash
python scripts/generate_daily_predictions.py
```

The script will:
- Load the trained model and metadata
- Fetch all active stocks from `stocks` table
- Get latest features for each stock from `features_daily`
- Generate predictions using the model
- Insert or update predictions in `predictions` table

### Output Format

**Example output:**
```
Model: Random Forest
Features: 11 columns

Prediction date: 2026-06-09
Target date: 2026-06-10
------------------------------------------------------------

[1/5] AAPL
  Latest feature date: 2026-06-08
  Prediction: UP
  Probability UP:   0.5234
  Probability DOWN: 0.4766
  Confidence:       0.5234
  Inserted new prediction

[2/5] MSFT
  Latest feature date: 2026-06-08
  Prediction: DOWN
  Probability UP:   0.4512
  Probability DOWN: 0.5488
  Confidence:       0.5488
  Updated existing prediction
```

### View Predictions in Supabase

**Latest predictions for each stock:**

```sql
SELECT
  s.ticker,
  p.prediction_date,
  p.target_date,
  CASE p.predicted_direction
    WHEN 1 THEN 'UP'
    ELSE 'DOWN'
  END as direction,
  ROUND(p.probability_up::numeric, 4) as prob_up,
  ROUND(p.probability_down::numeric, 4) as prob_down,
  ROUND(p.confidence_score::numeric, 4) as confidence
FROM predictions p
JOIN stocks s ON s.id = p.stock_id
WHERE p.prediction_date = CURRENT_DATE
ORDER BY s.ticker;
```

**Prediction history for a specific stock:**

```sql
SELECT
  prediction_date,
  target_date,
  predicted_direction,
  probability_up,
  confidence_score
FROM predictions
WHERE stock_id = (SELECT id FROM stocks WHERE ticker = 'AAPL')
ORDER BY prediction_date DESC
LIMIT 10;
```

---

## Project Structure

```
.
├── index.html              # Simple database viewer dashboard
├── models/
│   ├── price_direction_model.joblib      # Trained model
│   └── price_direction_model_metadata.json
├── scripts/
│   ├── ingest_prices_daily.py        # Alpha Vantage price ingestion
│   ├── build_features_daily.py       # Feature engineering pipeline
│   ├── train_price_model.py          # Model training pipeline
│   └── generate_daily_predictions.py # Prediction generation pipeline
├── seed-prices.js          # Node.js script to seed example data
├── supabase/
│   └── migrations/        # Database schema migrations
│       ├── 20260605202756_create_stock_prediction_schema.sql
│       ├── 20260605202838_add_rls_policies_stock_prediction.sql
│       └── 20260605202901_seed_example_stocks.sql
└── .env                    # Environment variables (not committed)
```

---

## Next Steps

- [ ] News article ingestion (NewsAPI, Yahoo Finance)
- [ ] Sentiment scoring with FinBERT
- [ ] Advanced models (XGBoost, LSTM)
- [ ] Prediction API for live forecasting
- [ ] Alert notifications (email, Telegram)
