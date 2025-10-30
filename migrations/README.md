# Database Migrations

This folder contains SQL migration scripts for updating your Supabase database schema.

## How to Run Migrations

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor** (left sidebar)
3. Click **New Query**
4. Copy and paste the SQL from the migration file
5. Click **Run** to execute

## Available Migrations

### `add_auto_trade_columns.sql`

**Purpose:** Adds auto-trading functionality to the `saved_tickers` table

**Adds:**
- `auto_trade_enabled` (BOOLEAN) - Whether auto-trading is enabled for this ticker
- `auto_trade_strategy` (TEXT) - Preferred strategy: STOCKS, OPTIONS, SCALPING, or ALL

**When to run:** Required for the Auto-Trader feature to work

**Safe to re-run:** Yes, uses `IF NOT EXISTS` clauses

## Verification

After running a migration, verify it worked:

```sql
-- Check if columns exist
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'saved_tickers' 
  AND column_name IN ('auto_trade_enabled', 'auto_trade_strategy');
```

You should see both columns listed.
