# Penny Stock Watchlist Integration

## Overview

The Penny Stock Watchlist has been successfully integrated into the AI Options Trader application as a **standalone, self-contained module** that operates independently of Google Sheets.

## ✨ Key Features

### 1. **Comprehensive Scoring System**
- **Momentum Score (0-100)**: Based on price action, volume, social buzz, and technical indicators
- **Valuation Score (0-100)**: Analyzes fundamentals including P/E ratio, revenue growth, profit margins
- **Catalyst Score (0-100)**: Evaluates news sentiment, upcoming events, and insider activity
- **Composite Score (0-100)**: Weighted combination of all scores for overall assessment

### 2. **Confidence Levels**
Automatically determined based on composite score:
- **VERY HIGH**: ≥80
- **HIGH**: ≥65
- **MEDIUM**: ≥50
- **LOW**: ≥35
- **VERY LOW**: <35

### 3. **Persistent Storage**
- **SQLite Database**: All watchlist data stored locally in `watchlist.db`
- **Historical Tracking**: Score history tracked over time
- **No Cloud Dependency**: Completely standalone operation

### 4. **Rich Analysis**
- Real-time stock data via yFinance
- Technical indicators (RSI, moving averages)
- Fundamental metrics (P/E, revenue growth, margins)
- Risk narrative generation
- Confidence reasoning

## 📁 New Files

### Core Modules

1. **`penny_stock_analyzer.py`**
   - `PennyStockScorer`: Calculates all scoring metrics
   - `PennyStockAnalyzer`: Performs comprehensive stock analysis
   - `StockScores`: Data class for score results

2. **`watchlist_manager.py`**
   - `WatchlistManager`: Complete CRUD operations for watchlist
   - SQLite database management
   - Import/export functionality (CSV, JSON)
   - Filtering and statistics

### Updated Files

3. **`app.py`**
   - New tab: "💰 Penny Stock Watchlist"
   - Four main views: Dashboard, Add Stock, Filter, Statistics

## 🚀 Usage Guide

### Getting Started

1. **Launch the Application**
```powershell
streamlit run app.py
```

2. **Navigate to Penny Stock Watchlist Tab**
   - Click on the "💰 Penny Stock Watchlist" tab

3. **Add Your First Stock**
   - Click "➕ Add Stock" button
   - Enter a ticker symbol (e.g., SOFI, PLTR, NIO)
   - Click "🔍 Analyze & Add"
   - Review the scores and add to watchlist

### Dashboard Features

#### **📊 View Dashboard**
- See all stocks sorted by composite score
- View key metrics: Total stocks, average scores, high confidence count
- Customize display: Sort by any metric, adjust number of stocks shown
- Drill down into individual stock details

#### **➕ Add Stock**
- Enter ticker symbol
- Automatic analysis with comprehensive scoring
- Add optional information: sector, buzz level, catalysts, notes
- One-click addition to watchlist

#### **🔍 Filter Stocks**
- Filter by minimum scores (composite, momentum, valuation, catalyst)
- Filter by confidence levels
- Filter by sectors
- Filter by risk levels
- View filtered results instantly

#### **📈 Statistics**
- Overview metrics with visual charts
- Score averages across all stocks
- Confidence level distribution
- Sector distribution

### Stock Detail Actions

For each stock in your watchlist, you can:
- **🔄 Refresh Data**: Update with latest market data and recalculate scores
- **📊 View History**: See score trends over time
- **🗑️ Remove Stock**: Delete from watchlist

### Export Options

Export your watchlist in two formats:
- **📄 CSV**: For Excel, Google Sheets, or data analysis
- **📋 JSON**: For programmatic access or backup

## 🎯 Scoring Algorithm Details

### Momentum Score Calculation

```python
Base score: 50

Price momentum (30 points):
- >20% change: +30
- >10% change: +20
- >5% change: +10
- >0% change: +5
- <-10% change: -20
- <-5% change: -10

Volume vs average (25 points):
- 3x+ average: +25
- 2x+ average: +20
- 1.5x+ average: +15
- Above average: +10
- <50% average: -15

Social buzz (20 points):
- High: +20
- Med: +10
- Low: +5

Technical score (15 points):
- Scaled from technical analysis

RSI consideration (10 points):
- <30 (oversold): +10
- >70 (overbought): -5
```

### Valuation Score Calculation

```python
Base score: 50

P/E Ratio (25 points):
- <10: +25 (very cheap)
- <15: +20
- <20: +15
- <30: +5
- >50: -15 (expensive)

Revenue Growth (25 points):
- >50%: +25
- >30%: +20
- >15%: +15
- >5%: +10
- <-10%: -20

Profit Margin (20 points):
- >20%: +20
- >10%: +15
- >5%: +10
- >0%: +5
- <-20%: -20

Analyst Rating (15 points):
- Strong Buy: +15
- Buy: +10
- Hold: 0
- Sell: -10

Cash/Debt Status (15 points):
- Strong/Excellent: +15
- Good/Positive: +10
- Weak/Poor: -15
```

### Catalyst Score Calculation

```python
Base score: 50

News Sentiment (35 points):
- Very Positive: +35
- Positive: +25
- Neutral: 0
- Negative: -20
- Very Negative: -35

News Volume (10 points):
- >10 articles: +10
- >5 articles: +7
- >2 articles: +5
- 0 articles: -10

Catalyst Events (30 points):
- FDA/Approval: +30
- Earnings/Revenue: +25
- Partnership/Deal: +25
- Merger/Acquisition: +25
- Contract/Award: +20
- Launch/Release: +15

Insider Activity (15 points):
- Buy/Accumulation: +15
- Sell/Distribution: -15
```

### Composite Score

```python
Weighted Average:
- Momentum: 35%
- Valuation: 25%
- Catalyst: 20%
- Technical: 10%
- News Sentiment: 10%
```

## 🗄️ Database Schema

### Watchlist Table
```sql
- ticker (TEXT, UNIQUE)
- sector, price, pct_change, volume, avg_volume
- float_m, market_cap, pe_ratio, revenue_growth, profit_margin
- analyst_rating, analyst_target
- technical_score, rsi, ma_signal
- news_sentiment, news_count, news_summary
- buzz, catalyst, insider, cash_debt, risk, dilution, verified
- momentum_score, valuation_score, catalyst_score, composite_score
- confidence_level, reasoning, risk_narrative
- notes, date_added, last_updated
```

### Score History Table
```sql
- ticker, date, price
- momentum_score, valuation_score, catalyst_score, composite_score
- confidence_level
```

### Tags Table
```sql
- ticker, tag
```

## 🔌 API Integration

The module uses **yFinance** for market data:
- Real-time stock prices
- Historical data for technical analysis
- Company fundamentals
- News articles
- No API key required!

## 💡 Best Practices

### 1. **Regular Updates**
- Refresh stock data daily for best accuracy
- Review and update watchlist weekly

### 2. **Score Interpretation**
- **Composite 80+**: Exceptional opportunity - worth deep dive
- **Composite 65-79**: Strong opportunity - monitor closely
- **Composite 50-64**: Moderate opportunity - proceed with caution
- **Composite <50**: Weak signal - consider removing

### 3. **Risk Management**
- Always review risk narrative before trading
- High confidence ≠ no risk
- Diversify across sectors and confidence levels
- Use stop-losses and position sizing

### 4. **Filtering Strategy**
- Start broad, narrow down based on your criteria
- Consider combining filters (e.g., High Confidence + Tech Sector)
- Export filtered results for further analysis

## 🔄 Migration from Google Sheets

If you have existing data in Google Sheets:

1. **Export from Google Sheets**
   - File → Download → CSV

2. **Import to Watchlist**
   ```python
   from watchlist_manager import WatchlistManager
   wm = WatchlistManager()
   wm.import_from_csv('your_export.csv')
   ```

3. **Refresh Data**
   - Use the dashboard to refresh each stock
   - This will recalculate all scores with live data

## 🛠️ Customization

### Adjust Score Weights

Edit `penny_stock_analyzer.py`:
```python
SCORE_WEIGHTS = {
    'MOMENTUM': 0.35,    # Change these values
    'VALUATION': 0.25,
    'CATALYST': 0.20,
    'TECHNICAL': 0.10,
    'NEWS_SENTIMENT': 0.10
}
```

### Adjust Confidence Thresholds

Edit `penny_stock_analyzer.py`:
```python
CONFIDENCE_THRESHOLDS = {
    'VERY_HIGH': 80,
    'HIGH': 65,
    'MEDIUM': 50,
    'LOW': 35
}
```

## 📊 Example Workflow

### Daily Trading Routine

1. **Morning Review** (Pre-market)
   - Open Penny Stock Watchlist
   - Click "📊 View Dashboard"
   - Sort by Composite Score (descending)
   - Review top 5-10 stocks

2. **Deep Dive** (Market Hours)
   - Select high-confidence stocks
   - Click "🔄 Refresh Data" for latest info
   - Review reasoning and risk narrative
   - Check news sentiment

3. **Position Selection**
   - Use Filter view to narrow down
   - Export filtered list for tracking
   - Execute trades based on your strategy

4. **End of Day**
   - Update notes for stocks you traded
   - Remove low-performing stocks
   - Add new candidates for next day

## 🚨 Troubleshooting

### Database Issues
If you encounter database errors:
```powershell
# Delete and recreate database
rm watchlist.db
# Restart app - database will be recreated
```

### Missing Data
- Some penny stocks lack analyst coverage (normal)
- Technical indicators require sufficient price history
- News may be sparse for very small companies

### Performance
- First analysis of a stock takes longer (fetching data)
- Subsequent refreshes are faster (some caching)
- Limit watchlist to 50-100 stocks for best performance

## 🎓 Learning Resources

### Understanding Scores
- Review the scoring algorithm details above
- Compare scores across different sectors
- Track how scores correlate with actual price movements

### Backtesting
- Export historical data
- Compare past composite scores with subsequent performance
- Refine your personal filtering criteria

## 📝 Changelog

### Version 1.0 (Current)
- ✅ Complete integration with AI Options Trader
- ✅ SQLite persistence
- ✅ Comprehensive scoring system
- ✅ Four main views (Dashboard, Add, Filter, Stats)
- ✅ Export functionality
- ✅ Historical tracking
- ✅ No Google Sheets dependency

### Future Enhancements (Roadmap)
- 🔮 Alerts and notifications
- 🔮 Technical analysis charts
- 🔮 Comparison view (side-by-side stocks)
- 🔮 Backtesting module
- 🔮 Integration with options analysis
- 🔮 Portfolio tracking

## 🤝 Support

For issues or questions:
1. Check this documentation first
2. Review the scoring algorithm details
3. Check the troubleshooting section
4. Verify your Python environment and dependencies

## ⚖️ Disclaimer

This tool is for **informational and educational purposes only**. 
- Not financial advice
- Do your own research
- Past performance ≠ future results
- Trading involves risk of loss

Always consult with a qualified financial advisor before making investment decisions.
