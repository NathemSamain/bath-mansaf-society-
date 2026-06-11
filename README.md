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

**Features used (11 total):**
- `return_1d`, `return_5d`, `return_20d` - Price momentum
- `volatility_5d`, `volatility_20d` - Volatility measures
- `volume_change_1d` - Volume dynamics
- `moving_average_5d`, `moving_average_20d` - Trend indicators
- `rsi_14` - Overbought/oversold oscillator
- `macd`, `macd_signal` - MACD indicator

**Target:**
- `target_next_day_direction` - 1 if price goes up next day, 0 otherwise

**Time-based split:**
- 70% training (oldest data)
- 15% validation
- 15% testing (newest data)

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
- Load features from `features_daily` table
- Clean rows with missing values
- Split data chronologically (no data leakage)
- Train Logistic Regression and Random Forest
- Evaluate both on validation set
- Select best model by F1 score
- Evaluate best model on held-out test set
- Save model to `models/price_direction_model.joblib`
- Save metadata to `models/price_direction_model_metadata.json`

### Output Metrics

The script prints for each model:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

**Example output:**
```
Random Forest (Validation) Results:
  Accuracy:  0.5421
  Precision: 0.5389
  Recall:    0.4912
  F1 Score:  0.5140
  Confusion Matrix:
  [[45 32]
   [38 35]]

Best model: Random Forest (F1 = 0.5140)

Random Forest (Test) Results:
  Accuracy:  0.5512
  F1 Score:  0.5203
```

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
  "feature_columns": ["return_1d", "return_5d", ...],
  "training_rows": 700,
  "validation_rows": 150,
  "test_rows": 150,
  "validation_metrics": {...},
  "test_metrics": {...},
  "created_at": "2026-06-09T..."
}
```

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
