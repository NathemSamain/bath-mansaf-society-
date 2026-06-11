
/*
# Seed Data — 5 Example Stocks

## Overview
Inserts 5 well-known US equity tickers as the initial stock universe for the prediction system.
Uses ON CONFLICT DO NOTHING so the migration is safe to re-run without errors or duplicates.

## Inserted Stocks
- AAPL  — Apple Inc. (NASDAQ, Technology / Consumer Electronics)
- MSFT  — Microsoft Corporation (NASDAQ, Technology / Software)
- NVDA  — NVIDIA Corporation (NASDAQ, Technology / Semiconductors)
- TSLA  — Tesla Inc. (NASDAQ, Consumer Discretionary / Auto Manufacturers)
- AMZN  — Amazon.com Inc. (NASDAQ, Consumer Discretionary / Internet Retail)

## Notes
- All tickers default to USD, is_active = true.
- The `id` values are deterministic UUIDs so downstream fixtures or test seeds can reference
  them without a lookup.
*/

INSERT INTO stocks (id, ticker, company_name, exchange, sector, industry, currency, is_active)
VALUES
  (
    'a1000000-0000-0000-0000-000000000001',
    'AAPL',
    'Apple Inc.',
    'NASDAQ',
    'Technology',
    'Consumer Electronics',
    'USD',
    TRUE
  ),
  (
    'a1000000-0000-0000-0000-000000000002',
    'MSFT',
    'Microsoft Corporation',
    'NASDAQ',
    'Technology',
    'Software—Infrastructure',
    'USD',
    TRUE
  ),
  (
    'a1000000-0000-0000-0000-000000000003',
    'NVDA',
    'NVIDIA Corporation',
    'NASDAQ',
    'Technology',
    'Semiconductors',
    'USD',
    TRUE
  ),
  (
    'a1000000-0000-0000-0000-000000000004',
    'TSLA',
    'Tesla Inc.',
    'NASDAQ',
    'Consumer Discretionary',
    'Auto Manufacturers',
    'USD',
    TRUE
  ),
  (
    'a1000000-0000-0000-0000-000000000005',
    'AMZN',
    'Amazon.com Inc.',
    'NASDAQ',
    'Consumer Discretionary',
    'Internet Retail',
    'USD',
    TRUE
  )
ON CONFLICT (ticker) DO NOTHING;
