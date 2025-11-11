# AI Entry Analysis Migration

## Purpose
Adds columns to the `saved_tickers` table to store AI entry timing analysis results.

## How to Run

### Option 1: Supabase Dashboard (Recommended)
1. Go to your Supabase project dashboard
2. Click **SQL Editor** in the left sidebar
3. Click **New Query**
4. Copy the contents of `add_ai_entry_columns.sql`
5. Paste into the SQL editor
6. Click **Run** (or press Ctrl+Enter)
7. Verify success message appears

### Option 2: Supabase CLI
```bash
# If you have Supabase CLI installed
supabase db push

# Or run directly
psql -h your-project.supabase.co -U postgres -d postgres -f migrations/add_ai_entry_columns.sql
```

### Option 3: SQL Snippet (Quick Copy-Paste)
```sql
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

CREATE INDEX IF NOT EXISTS idx_saved_tickers_ai_action ON saved_tickers(ai_entry_action);
CREATE INDEX IF NOT EXISTS idx_saved_tickers_ai_confidence ON saved_tickers(ai_entry_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_saved_tickers_ai_timestamp ON saved_tickers(ai_entry_timestamp DESC);
```

## Columns Added

| Column | Type | Description |
|--------|------|-------------|
| `ai_entry_action` | TEXT | ENTER_NOW, WAIT_FOR_PULLBACK, WAIT_FOR_BREAKOUT, PLACE_LIMIT_ORDER, DO_NOT_ENTER |
| `ai_entry_confidence` | NUMERIC | AI confidence score 0-100 |
| `ai_entry_reasoning` | TEXT | AI explanation for the recommendation |
| `ai_entry_timestamp` | TIMESTAMPTZ | When the AI analysis was performed |
| `ai_entry_price` | NUMERIC | Suggested entry price (if applicable) |
| `ai_technical_score` | NUMERIC | Technical analysis score 0-100 |
| `ai_timing_score` | NUMERIC | Entry timing score 0-100 |
| `ai_trend_score` | NUMERIC | Trend strength score 0-100 |
| `ai_risk_score` | NUMERIC | Risk assessment score 0-100 |

## Indexes Created
- `idx_saved_tickers_ai_action` - Fast filtering by AI action
- `idx_saved_tickers_ai_confidence` - Sort by confidence descending
- `idx_saved_tickers_ai_timestamp` - Sort by analysis date descending

## Verification

After running the migration, verify it worked:

```sql
-- Check if columns exist
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'saved_tickers' 
  AND column_name LIKE 'ai_%'
ORDER BY column_name;

-- Check if indexes exist
SELECT indexname 
FROM pg_indexes 
WHERE tablename = 'saved_tickers' 
  AND indexname LIKE 'idx_saved_tickers_ai_%';
```

Expected output: 9 columns and 3 indexes

## Rollback (If Needed)

```sql
-- Remove indexes
DROP INDEX IF EXISTS idx_saved_tickers_ai_action;
DROP INDEX IF EXISTS idx_saved_tickers_ai_confidence;
DROP INDEX IF EXISTS idx_saved_tickers_ai_timestamp;

-- Remove columns
ALTER TABLE saved_tickers
DROP COLUMN IF EXISTS ai_entry_action,
DROP COLUMN IF EXISTS ai_entry_confidence,
DROP COLUMN IF EXISTS ai_entry_reasoning,
DROP COLUMN IF EXISTS ai_entry_timestamp,
DROP COLUMN IF EXISTS ai_entry_price,
DROP COLUMN IF EXISTS ai_technical_score,
DROP COLUMN IF EXISTS ai_timing_score,
DROP COLUMN IF EXISTS ai_trend_score,
DROP COLUMN IF EXISTS ai_risk_score;
```

## Next Steps

After running this migration:
1. ✅ Bulk AI analysis will save results to database
2. ✅ Ticker cards will show AI recommendations
3. ✅ Filter/sort by AI confidence and action
4. ✅ Track analysis history over time
