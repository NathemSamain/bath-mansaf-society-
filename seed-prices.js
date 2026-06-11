import { createClient } from "@supabase/supabase-js";
import "dotenv/config";

const supabase = createClient(
  process.env.VITE_SUPABASE_URL,
  process.env.VITE_SUPABASE_ANON_KEY
);

// Deterministic stock IDs from seed migration
const STOCK_IDS = {
  AAPL: "a1000000-0000-0000-0000-000000000001",
  MSFT: "a1000000-0000-0000-0000-000000000002",
  NVDA: "a1000000-0000-0000-0000-000000000003",
  TSLA: "a1000000-0000-0000-0000-000000000004",
  AMZN: "a1000000-0000-0000-0000-000000000005",
};

// Realistic starting prices and volatility profiles
const STOCK_CONFIG = {
  AAPL: { start: 152.0, volatility: 0.018 },
  MSFT: { start: 380.0, volatility: 0.016 },
  NVDA: { start: 875.0, volatility: 0.032 },
  TSLA: { start: 240.0, volatility: 0.045 },
  AMZN: { start: 187.0, volatility: 0.020 },
};

function generatePriceData(ticker, startPrice, volatility) {
  const prices = [];
  const endDate = new Date(2026, 5, 5); // 2026-06-05
  const startDate = new Date(2025, 5, 5); // 2025-06-05
  let price = startPrice;

  for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
    // Skip weekends
    if (d.getDay() === 0 || d.getDay() === 6) continue;

    // Random walk with drift
    const dailyReturn = (Math.random() - 0.5) * volatility * 2 + 0.0002;
    const newPrice = price * (1 + dailyReturn);

    // Generate OHLC within the day
    const open = price;
    const close = newPrice;
    const high = Math.max(open, close) * (1 + Math.random() * 0.01);
    const low = Math.min(open, close) * (1 - Math.random() * 0.01);

    prices.push({
      stock_id: STOCK_IDS[ticker],
      date: d.toISOString().split("T")[0],
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(close.toFixed(2)),
      adjusted_close: parseFloat(close.toFixed(2)),
      volume: Math.floor(20000000 + Math.random() * 30000000),
      source: "generated",
    });

    price = newPrice;
  }

  return prices;
}

async function insertPrices() {
  console.log("Generating 1 year of daily prices for 5 stocks...");

  let totalInserted = 0;
  let errors = [];

  for (const [ticker, config] of Object.entries(STOCK_CONFIG)) {
    console.log(`\nGenerating ${ticker}...`);
    const prices = generatePriceData(ticker, config.start, config.volatility);
    console.log(`  Generated ${prices.length} trading days`);

    // Insert in batches of 50 to avoid payload limits
    const batchSize = 50;
    for (let i = 0; i < prices.length; i += batchSize) {
      const batch = prices.slice(i, i + batchSize);
      const { error } = await supabase.from("prices_daily").insert(batch);

      if (error) {
        errors.push({ ticker, batch: i / batchSize + 1, error: error.message });
        console.error(`  ✗ Batch ${i / batchSize + 1} failed:`, error.message);
      } else {
        console.log(`  ✓ Batch ${i / batchSize + 1} inserted (${batch.length} rows)`);
        totalInserted += batch.length;
      }
    }
  }

  console.log("\n" + "=".repeat(60));
  console.log(`Total rows inserted: ${totalInserted}`);

  if (errors.length > 0) {
    console.log(`\nErrors encountered: ${errors.length}`);
    errors.forEach((e) => console.log(`  - ${e.ticker} batch ${e.batch}: ${e.error}`));
    process.exit(1);
  } else {
    console.log("✓ All data inserted successfully!");
    process.exit(0);
  }
}

insertPrices().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
