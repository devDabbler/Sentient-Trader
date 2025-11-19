# Multi-Configuration Bulk Analysis for Stocks

## Overview
Added comprehensive multi-configuration bulk analysis to the Watchlist tab, similar to the crypto multi-config feature. This allows you to test **multiple trading configurations** across your entire watchlist and find the optimal setup for each stock.

## What It Does

### Tests Multiple Configurations Per Ticker:
1. **Position Sizes** - Conservative ($1,000), Moderate ($2,000), Aggressive ($5,000) - customizable
2. **Risk Levels** - 1%, 2%, 3% - customizable
3. **Trading Styles:**
   - **Swing Trading** (3:1 Risk/Reward ratio)
   - **Day Trading** (2:1 Risk/Reward ratio)
   - **Scalping** (1.5:1 Risk/Reward ratio)

### AI Analysis for Each Configuration:
- Runs AI Entry Analysis on **every combination**
- Scores each configuration with AI confidence
- Identifies optimal entry timing (ENTER_NOW, WAIT_FOR_PULLBACK, etc.)
- Provides detailed reasoning and risk assessment

### Intelligent Results Display:
- **Best Configuration Per Ticker** - Shows the highest-scoring setup for each stock
- **Summary Metrics** - Total configs tested, ENTER NOW count, average confidence, best score
- **Full Comparison Table** - View all configurations side-by-side
- **Filtering & Export** - Filter by action, confidence, style; export to CSV

## How to Use

### Step 1: Navigate to Watchlist
Go to **Watchlist** tab → Scroll to bottom

### Step 2: Select Multi-Config Analysis
Click the **"🎯 Multi-Config Analysis"** tab

### Step 3: Configure Settings
1. **Select Trading Styles:**
   - ✅ Swing Trading (3:1 R:R) - Default ON
   - ✅ Day Trading (2:1 R:R) - Default ON
   - ⬜ Scalping (1.5:1 R:R) - Default OFF

2. **Set Position Sizes:**
   - Input comma-separated values (e.g., `1000,2000,5000`)
   - Default: $1,000, $2,000, $5,000

3. **Set Risk Levels:**
   - Input comma-separated percentages (e.g., `1.0,2.0,3.0`)
   - Default: 1%, 2%, 3%

4. **Choose Tickers:**
   - Slider to select how many tickers to analyze (1-20)
   - Default: First 5 tickers in your watchlist

### Step 4: Run Analysis
Click **"🚀 Analyze All Configurations"** button

The system will:
- Calculate total configurations (e.g., 5 tickers × 3 positions × 3 risks × 2 styles = 90 configs)
- Show real-time progress bar
- Complete in ~1-3 minutes depending on count

### Step 5: Review Results

#### 📊 Summary Metrics
- **Total Configs** - How many were tested
- **ENTER NOW** - Count and percentage of immediate entry signals
- **Avg Confidence** - Average AI confidence score
- **Best Score** - Highest confidence found

#### 🏆 Best Configuration Per Ticker
Shows the optimal setup for each ticker:
- **Position Size** - Best dollar amount
- **Risk %** - Optimal risk level
- **Trading Style** - SWING, DAY_TRADE, or SCALP
- **Take Profit %** - Target based on style
- **AI Confidence** - Score from 0-100%
- **Action** - ENTER_NOW, WAIT_FOR_PULLBACK, WAIT_FOR_BREAKOUT, DO_NOT_ENTER
- **Pricing** - Current, Entry, Stop, Target prices
- **Analysis Scores** - Technical, Trend, Timing, Risk (each 0-100)
- **AI Reasoning** - Detailed explanation

**Save Button** - Click "💾 Save Best Config" to update the database with `ai_entry_action` field

#### 📋 Full Comparison Table
View all configurations in sortable table with filters:
- **Filter by Action** - Show only ENTER_NOW, WAIT, etc.
- **Filter by Confidence** - Min confidence slider (0-100%)
- **Filter by Style** - Multi-select SWING/DAY_TRADE/SCALP
- **Export to CSV** - Download full results

## Example Workflow

### Scenario: You have 10 stocks in your watchlist

1. **Configure:**
   - Styles: Swing + Day Trading
   - Position Sizes: $1000, $2000, $5000
   - Risk Levels: 1%, 2%, 3%
   - Tickers: Select 5 to analyze

2. **Total Configurations:**
   - 5 tickers × 3 positions × 3 risks × 2 styles = **90 AI analyses**

3. **Results:**
   - AAPL: Best = SWING, $2000, 2% risk → **ENTER_NOW (85% confidence)**
   - TSLA: Best = DAY_TRADE, $5000, 3% risk → **WAIT_FOR_PULLBACK (72% confidence)**
   - NVDA: Best = SWING, $1000, 1% risk → **ENTER_NOW (78% confidence)**
   - AMD: Best = DAY_TRADE, $2000, 2% risk → **WAIT_FOR_BREAKOUT (68% confidence)**
   - MSFT: Best = SWING, $5000, 1% risk → **DO_NOT_ENTER (45% confidence)**

4. **Action:**
   - Click "💾 Save Best Config" for AAPL and NVDA
   - Database updated with `ai_entry_action = "ENTER_NOW"`
   - Filter watchlist by "ENTER NOW" to see only these stocks

## Benefits

### 1. Data-Driven Decisions
- No guessing which position size or risk level is optimal
- AI tests all combinations and ranks them
- See exactly which configuration has highest confidence

### 2. Time Savings
- Manual testing: 5 tickers × 9 configs each = 45 individual analyses (hours of work)
- Bulk analysis: 1 click, 2 minutes, all done

### 3. Risk Management
- See how different risk levels affect AI confidence
- Identify which stocks perform better with lower/higher risk
- Find optimal risk/reward ratios per ticker

### 4. Style Matching
- Discover which trading style suits each stock
- Some stocks better for swing, others for day trading
- AI evaluates entry timing specific to each style

### 5. Portfolio Optimization
- Find best opportunities across entire watchlist
- Allocate capital to highest-confidence setups
- Balance position sizes based on AI recommendations

## What Gets Saved to Database

When you click "💾 Save Best Config for [TICKER]":

```python
{
    'confidence': 85.5,  # AI confidence score
    'action': 'ENTER_NOW',  # Entry timing action
    'reasons': ['Strong bullish trend with RSI at 45...'],  # AI reasoning
    'targets': {
        'suggested_entry': 150.25,  # Suggested entry price
        'suggested_stop': 147.50,   # Stop loss
        'suggested_target': 155.75, # Take profit
        'position_size': 2000,      # Optimal position size
        'risk_pct': 2.0,            # Optimal risk %
        'style': 'SWING'            # Best trading style
    }
}
```

This populates the `ai_entry_action` field, enabling the watchlist filter to work!

## Filtering Your Watchlist

After running bulk analysis and saving results:

1. Go to **"📋 Your Saved Tickers"** section
2. Use **"Filter by AI Action"** dropdown
3. Select **"ENTER_NOW"** 
4. See only stocks with immediate entry signals!

Similarly for:
- **WAIT_FOR_PULLBACK** - Good stocks, wait for better entry
- **WAIT_FOR_BREAKOUT** - Wait for confirmation
- **DO_NOT_ENTER** - Avoid these trades

## Files Modified

1. **`ui/bulk_ai_entry_analysis_ui.py`**
   - Added `analyze_multi_config_bulk()` function
   - Added `display_multi_config_bulk_analysis()` UI component
   - ~400 lines of multi-config analysis logic

2. **`ui/tabs/watchlist_tab.py`**
   - Added tab selector for bulk analysis type
   - Import multi-config display function
   - Seamless integration with existing watchlist

## Performance

- **Speed:** ~0.5-1 second per AI analysis
- **Example:** 100 configurations = ~50-100 seconds (1-2 minutes)
- **Progress:** Real-time progress bar shows completion status
- **Caching:** Results cached in session state until next run

## Comparison with Crypto Multi-Config

### Similarities:
- Tests multiple configurations per asset
- AI scoring and ranking
- Best config per asset display
- Full comparison table with filters
- Export to CSV

### Differences:
| Feature | Crypto | Stocks |
|---------|--------|--------|
| Leverage | 1x, 2x, 3x, 5x | N/A (stocks don't use leverage) |
| Direction | BUY (LONG) / SELL (SHORT) | BUY only |
| Position Size | Based on leverage | Direct USD amounts |
| Trading Styles | Fixed (scalp/momentum/swing) | Configurable (SWING/DAY/SCALP) |
| Risk/Reward | Penalized by leverage | Based on trading style |

## Next Steps

1. **Run your first multi-config analysis**
   - Start with 3-5 tickers
   - Use default settings
   - Review results

2. **Save best configurations**
   - Click "💾 Save" on high-confidence setups
   - Database will populate `ai_entry_action` field

3. **Use watchlist filters**
   - Filter by "ENTER NOW"
   - See only your best opportunities
   - Execute trades with confidence

4. **Experiment with settings**
   - Try different position sizes
   - Test various risk levels
   - Compare SWING vs DAY_TRADE vs SCALP

## Troubleshooting

### "No configurations match your filters"
- Reduce min confidence threshold
- Select more trading styles
- Check action filter is not too restrictive

### Analysis takes too long
- Reduce number of tickers (slider)
- Use fewer position sizes
- Use fewer risk levels
- Disable scalping (tests fewer configs)

### AI Entry Assistant not available
- Ensure broker is connected (IBKR or Tradier)
- Ensure LLM is configured (OpenRouter API key)
- Check logs for initialization errors

## Tips for Best Results

1. **Start Small** - Test 3-5 tickers first to understand results
2. **Use Realistic Position Sizes** - Match your actual trading capital
3. **Vary Risk Levels** - Test 1-3% to find your comfort zone
4. **Save Winning Configs** - Click "Save" to populate database for filtering
5. **Review AI Reasoning** - Read the "AI Analysis" section to understand recommendations
6. **Compare Styles** - See which style (SWING/DAY/SCALP) fits each stock
7. **Export Results** - Download CSV for further analysis in Excel/Sheets

## Technical Details

### AI Entry Analysis Criteria:
- **Technical Score (0-100)** - RSI, MACD, indicators
- **Trend Score (0-100)** - Trend strength and direction
- **Timing Score (0-100)** - Entry timing quality
- **Risk Score (0-100)** - Risk assessment (lower = safer)

### Composite Score Calculation:
```python
composite_score = ai_confidence
# No leverage penalties for stocks (unlike crypto)
# Pure AI confidence is the score
```

### Risk/Reward Ratios by Style:
- **Scalping:** 1.5:1 (1% risk → 1.5% profit target)
- **Day Trading:** 2:1 (2% risk → 4% profit target)
- **Swing Trading:** 3:1 (3% risk → 9% profit target)

## Success Metrics

After using multi-config analysis, you should see:
- ✅ Populated `ai_entry_action` field in database
- ✅ Functional "ENTER NOW" filter in watchlist
- ✅ Clear view of best opportunities
- ✅ Optimal position sizing per ticker
- ✅ Data-driven entry timing decisions

Enjoy your new bulk analysis superpowers! 🚀
