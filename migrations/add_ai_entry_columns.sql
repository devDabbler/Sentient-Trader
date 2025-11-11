-- Migration: Add AI Entry Analysis columns to saved_tickers table
-- Date: 2025-11-07
-- Description: Adds columns to track AI-powered entry timing analysis results

-- Add AI entry analysis columns
ALTER TABLE saved_tickers
ADD COLUMN IF NOT EXISTS ai_entry_action TEXT,
ADD COLUMN IF NOT EXISTS ai_entry_confidence NUMERIC,
ADD COLUMN IF NOT EXISTS ai_entry_reasoning TEXT,
ADD COLUMN IF NOT EXISTS ai_entry_timestamp TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS ai_entry_price NUMERIC,
ADD COLUMN IF NOT EXISTS ai_technical_score NUMERIC,
ADD COLUMN IF NOT EXISTS ai_timing_score NUMERIC,
ADD COLUMN IF NOT EXISTS ai_trend_score NUMERIC,
ADD COLUMN IF NOT EXISTS ai_risk_score NUMERIC;

-- Add index for filtering by AI action
CREATE INDEX IF NOT EXISTS idx_saved_tickers_ai_action 
ON saved_tickers(ai_entry_action);

-- Add index for filtering by confidence
CREATE INDEX IF NOT EXISTS idx_saved_tickers_ai_confidence 
ON saved_tickers(ai_entry_confidence DESC);

-- Add index for sorting by timestamp
CREATE INDEX IF NOT EXISTS idx_saved_tickers_ai_timestamp 
ON saved_tickers(ai_entry_timestamp DESC);

-- Add comments for documentation
COMMENT ON COLUMN saved_tickers.ai_entry_action IS 'AI recommendation: ENTER_NOW, WAIT_FOR_PULLBACK, WAIT_FOR_BREAKOUT, PLACE_LIMIT_ORDER, DO_NOT_ENTER';
COMMENT ON COLUMN saved_tickers.ai_entry_confidence IS 'AI confidence score 0-100';
COMMENT ON COLUMN saved_tickers.ai_entry_reasoning IS 'AI explanation for the recommendation';
COMMENT ON COLUMN saved_tickers.ai_entry_timestamp IS 'When the AI analysis was performed';
COMMENT ON COLUMN saved_tickers.ai_entry_price IS 'Suggested entry price (if applicable)';
COMMENT ON COLUMN saved_tickers.ai_technical_score IS 'Technical analysis score 0-100';
COMMENT ON COLUMN saved_tickers.ai_timing_score IS 'Entry timing score 0-100';
COMMENT ON COLUMN saved_tickers.ai_trend_score IS 'Trend strength score 0-100';
COMMENT ON COLUMN saved_tickers.ai_risk_score IS 'Risk assessment score 0-100';
