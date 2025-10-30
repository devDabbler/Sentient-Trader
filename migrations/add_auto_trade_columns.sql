-- Add auto-trade columns to saved_tickers table
-- Run this in your Supabase SQL Editor

ALTER TABLE saved_tickers 
ADD COLUMN IF NOT EXISTS auto_trade_enabled BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS auto_trade_strategy TEXT;

-- Create index for faster queries on auto-trade enabled tickers
CREATE INDEX IF NOT EXISTS idx_saved_tickers_auto_trade 
ON saved_tickers(auto_trade_enabled) 
WHERE auto_trade_enabled = TRUE;

-- Add comment for documentation
COMMENT ON COLUMN saved_tickers.auto_trade_enabled IS 'Whether automated trading is enabled for this ticker';
COMMENT ON COLUMN saved_tickers.auto_trade_strategy IS 'Preferred auto-trade strategy: STOCKS, OPTIONS, SCALPING, or ALL';
