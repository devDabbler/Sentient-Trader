# Phase 3 Advanced Features - Complete Guide

This document covers the advanced features added in Phase 3:
1. Real-time alerts for high-confidence setups
2. EMA Reclaim + Fibonacci backtesting framework
3. Preset scan filters for rapid opportunity identification
4. Fibonacci-based options chain integration

---

## 1. Real-Time Alert System

### Overview
The alert system monitors stock analysis in real-time and triggers notifications when high-probability setups occur.

### Key Components

**Alert Types:**
- `EMA_RECLAIM` - EMA reclaim confirmed
- `HIGH_CONFIDENCE` - Confidence score ≥ 85 or triple threat setup
- `TIMEFRAME_ALIGNED` - Multi-timeframe alignment detected
- `SECTOR_LEADER` - Strong relative strength (RS > 70)
- `FIBONACCI_SETUP` - Fibonacci pattern + DeMarker entry
- `DEMARKER_ENTRY` - DeMarker oversold/overbought signal

**Alert Priorities:**
- `CRITICAL` - Triple threat, EMA reclaim
- `HIGH` - Timeframe alignment, sector leader, Fibonacci setup
- `MEDIUM` - DeMarker entry signals

### Usage

```python
from services.alert_system import get_alert_system, SetupDetector, console_callback

# Initialize alert system
alert_system = get_alert_system()
alert_system.add_callback(console_callback)  # Print to console

# Create detector
detector = SetupDetector(alert_system)

# Analyze stock and generate alerts
analysis = ComprehensiveAnalyzer.analyze_stock("AAPL", "SWING_TRADE")
alerts = detector.analyze_for_alerts(analysis)

# Get recent critical alerts
critical_alerts = alert_system.get_recent_alerts(priority=AlertPriority.CRITICAL)
```

### Custom Callbacks

You can add custom callbacks for email, SMS, webhook notifications:

```python
def my_webhook_callback(alert: TradingAlert):
    """Send alert to webhook"""
    import requests
    requests.post("https://mywebhook.com/alerts", json=alert.to_dict())

alert_system.add_callback(my_webhook_callback)
```

### Alert Log

Alerts are automatically logged to `logs/trading_alerts.json` with:
- Timestamp
- Ticker
- Alert type and priority
- Confidence score
- Setup details (EMAs, DeMarker, Fibonacci levels, etc.)

---

## 2. EMA Reclaim + Fibonacci Backtesting

### Overview
Comprehensive backtesting framework to validate the EMA Reclaim + Fibonacci strategy with historical data.

### Features

- **Configurable Filters**: Test different combinations of reclaim, alignment, and sector RS requirements
- **Fibonacci Targets**: Automatic T1/T2/T3 profit-taking
- **Dynamic Stops**: Stop loss below 21 EMA
- **Performance Metrics**: Win rate, profit factor, Sharpe ratio, max drawdown
- **Setup-Specific Stats**: Track performance by setup type (reclaim, aligned, triple threat)

### Basic Usage

```python
from services.backtest_ema_fib import EMAFibonacciBacktester

# Create backtester
bt = EMAFibonacciBacktester(
    initial_capital=10000,
    position_size_pct=0.10,  # 10% per trade
    max_hold_days=15,
    use_fibonacci_targets=True,
    require_reclaim=True,
    require_alignment=False,
    require_strong_rs=False,
    min_confidence=60.0
)

# Run backtest
results = bt.backtest("AAPL", "2023-01-01", "2024-10-01")
results.print_summary()
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `initial_capital` | Starting capital | $10,000 |
| `position_size_pct` | Position size (% of capital) | 10% |
| `max_hold_days` | Maximum holding period | 15 days |
| `use_fibonacci_targets` | Use Fib T1/T2/T3 exits | True |
| `require_reclaim` | Only trade EMA reclaims | False |
| `require_alignment` | Only trade aligned setups | False |
| `require_strong_rs` | Only trade RS > 60 | False |
| `min_confidence` | Minimum confidence score | 60.0 |

### Performance Metrics

The backtest calculates:
- **Overall**: Total trades, win rate, total P&L, avg win/loss
- **Risk Metrics**: Max drawdown, Sharpe ratio, profit factor
- **Setup-Specific**: Win rates for reclaim, aligned, and triple threat setups
- **Trade Details**: Entry/exit prices, reasons, hold time for each trade

### Parameter Optimization

```python
# Optimize parameters using grid search
optimization = bt.optimize_parameters("AAPL", "2023-01-01", "2024-10-01")

print("Best Parameters:", optimization['best_params'])
print("Best Sharpe Ratio:", optimization['best_results'].sharpe_ratio)
```

### Backtesting Multiple Tickers

```python
tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
results_dict = bt.backtest_multiple(tickers, "2023-01-01", "2024-10-01")

for ticker, results in results_dict.items():
    print(f"\n{ticker}:")
    print(f"  Win Rate: {results.win_rate:.1f}%")
    print(f"  Profit Factor: {results.profit_factor:.2f}")
```

---

## 3. Preset Scan Filters

### Overview
Pre-configured scanners that filter stocks based on specific setup criteria for rapid opportunity identification.

### Available Presets

| Preset | Description | Criteria |
|--------|-------------|----------|
| `TRIPLE_THREAT` | Highest conviction | Reclaim + Aligned + Strong RS |
| `EMA_RECLAIM` | EMA reclaim only | Reclaim confirmed |
| `TIMEFRAME_ALIGNED` | Multi-timeframe confirmation | 2+ timeframes agree |
| `SECTOR_LEADERS` | Relative strength leaders | RS > 70 |
| `DEMARKER_PULLBACK` | Precise entries | DeMarker ≤ 0.30 in uptrend |
| `FIBONACCI_SETUP` | Fib patterns | A-B-C detected |
| `HIGH_CONFIDENCE` | High conviction | Confidence ≥ 85 |
| `POWER_ZONE` | Momentum filter | EMA 8 > 21, price above both |
| `OPTIONS_PREMIUM_SELL` | Sell premium setups | High IV + Power Zone |
| `OPTIONS_DIRECTIONAL` | Buy options setups | Low IV + Reclaim/Aligned |

### Basic Usage

```python
from services.preset_scanners import PresetScanner, ScanPreset, get_high_volume_tech

# Create scanner
scanner = PresetScanner()

# Get watchlist
tickers = get_high_volume_tech()

# Scan with specific preset
results = scanner.scan(tickers, ScanPreset.TRIPLE_THREAT)
scanner.print_results(results, show_details=True)
```

### Scan All Presets

```python
# Scan with all presets
all_results = scanner.scan_all_presets(tickers)

for preset, results in all_results.items():
    print(f"\n{preset.value}: {len(results)} matches")
```

### Get Top Opportunities

```python
# Get top 10 opportunities across all presets
top_opps = scanner.get_top_opportunities(tickers, top_n=10)

for opp in top_opps:
    print(f"{opp.ticker}: {opp.preset.value} - Score {opp.priority_score:.0f}")
```

### Built-in Watchlists

```python
from services.preset_scanners import (
    get_sp500_tickers,      # S&P 500 stocks
    get_nasdaq100_tickers,  # NASDAQ 100 stocks
    get_high_volume_tech    # High volume tech stocks
)

# Scan S&P 500 for reclaim setups
sp500 = get_sp500_tickers()
reclaim_setups = scanner.scan(sp500, ScanPreset.EMA_RECLAIM)
```

### Integration with Alerts

```python
# Automatically generate alerts for matches
results = scanner.scan(tickers, ScanPreset.TRIPLE_THREAT, generate_alerts=True)
# Alerts will be triggered for all matches
```

---

## 4. Fibonacci-Based Options Chain Integration

### Overview
Automatically finds option strikes near Fibonacci levels and suggests spreads based on market context.

### Features

- **Strike Selection**: Finds strikes closest to Fib A, B, C, T1, T2, T3
- **Spread Suggestions**: Auto-generates spreads based on trend and IV context
- **Real Data**: Uses live options chain from yfinance
- **Risk Metrics**: Calculates max profit/loss, breakeven, probability of profit

### Basic Usage

```python
from services.options_chain_fib import FibonacciOptionsChain

# Analyze stock for Fibonacci pattern
analysis = ComprehensiveAnalyzer.analyze_stock("AAPL", "OPTIONS")

if analysis.fib_targets:
    # Initialize options chain analyzer
    fib_chain = FibonacciOptionsChain("AAPL")
    
    # Find strikes near Fibonacci levels (45 DTE)
    fib_strikes = fib_chain.find_strikes_near_fibonacci(
        analysis.fib_targets, 
        target_dte=45
    )
    
    # Print strikes
    fib_chain.print_fibonacci_strikes(fib_strikes)
```

### Generate Spread Suggestions

```python
# Get spread suggestions based on context
spreads = fib_chain.suggest_fibonacci_spreads(
    analysis.fib_targets,
    analysis,
    target_dte=45
)

# Print suggestions
fib_chain.print_spread_suggestions(spreads)

# Get best spread by risk/reward
best_spread = max(spreads, key=lambda s: s.risk_reward_ratio)
```

### Spread Types Generated

**Bull Call Spread (Fib T1)**
- Uptrend + targets detected
- Buy ATM call, Sell call at T1 (127.2%)
- DTE: 30-60 days

**Bull Put Spread (Fib C Support)**
- EMA Reclaim + High IV
- Sell put at C level, Buy lower put
- DTE: 30-45 days
- Credit spread for income

**Wide Call Spread (Fib T2)**
- Strong uptrend + extended targets
- Buy ATM call, Sell call at T2 (161.8%)
- DTE: 45-60 days
- Larger profit potential

### Strike Information

For each Fibonacci level, you get:
- **Strike Price**: Closest available strike
- **Distance**: % from current spot price
- **DTE**: Days to expiration
- **Bid/Ask**: Call and put pricing
- **Open Interest**: Liquidity indicators
- **Greeks**: Delta, theta, vega (if available)

### Integration with Analysis

```python
# Complete workflow
analysis = ComprehensiveAnalyzer.analyze_stock("AAPL", "OPTIONS")

if analysis.ema_reclaim and analysis.iv_rank > 60:
    # High IV reclaim → Sell puts at support
    fib_chain = FibonacciOptionsChain("AAPL")
    strikes = fib_chain.find_strikes_near_fibonacci(analysis.fib_targets, 45)
    
    if 'C' in strikes:
        c_strike = strikes['C']
        print(f"Sell {c_strike.strike} put @ ${c_strike.get_mid_price('put'):.2f}")
        print(f"Support at C level: ${c_strike.strike:.2f}")

elif analysis.ema_reclaim and analysis.iv_rank < 40:
    # Low IV reclaim → Buy calls to T1
    spreads = fib_chain.suggest_fibonacci_spreads(analysis.fib_targets, analysis, 45)
    call_spreads = [s for s in spreads if 'Call' in s.strategy_name]
    
    if call_spreads:
        best = call_spreads[0]
        print(f"Buy {best.long_strike} / Sell {best.short_strike} call spread")
        print(f"Max profit: ${best.max_profit:.2f} | Risk/Reward: {best.risk_reward_ratio:.2f}")
```

---

## Complete Trading Workflow

### Daily Routine Example

```python
# 1. Scan for opportunities
scanner = PresetScanner()
watchlist = get_high_volume_tech()

top_setups = scanner.get_top_opportunities(watchlist, top_n=5)

# 2. For each opportunity, backtest the strategy
bt = EMAFibonacciBacktester(require_reclaim=True, require_alignment=True)

for setup in top_setups:
    results = bt.backtest(setup.ticker, "2023-01-01", "2024-10-01")
    
    # 3. If backtest is positive, analyze options
    if results.win_rate > 60 and results.profit_factor > 1.5:
        analysis = setup.analysis
        
        if analysis.fib_targets:
            fib_chain = FibonacciOptionsChain(setup.ticker)
            spreads = fib_chain.suggest_fibonacci_spreads(
                analysis.fib_targets, analysis, 45
            )
            
            # 4. Execute best spread
            if spreads:
                best = max(spreads, key=lambda s: s.risk_reward_ratio)
                print(f"\n✓ TRADE: {setup.ticker}")
                print(f"  {best.strategy_name}")
                print(f"  R/R: {best.risk_reward_ratio:.2f}")
```

---

## Performance Tips

1. **Scanning Large Watchlists**: Use multiprocessing for 50+ tickers
2. **Alert Frequency**: Set minimum time between alerts per ticker to avoid spam
3. **Backtesting Speed**: Reduce lookback period or use weekly data for faster results
4. **Options Chain**: Cache chain data to reduce API calls

---

## Testing

Run the comprehensive example script:

```powershell
python examples/phase3_advanced_features.py
```

This demonstrates all four features with live examples.

---

## Next Steps

1. **Custom Alerts**: Implement email/SMS callbacks
2. **Web Dashboard**: Build Streamlit dashboard for scanning and alerts
3. **Automated Execution**: Connect to broker API for automatic trade placement
4. **Machine Learning**: Train models on backtest results to predict setup quality
5. **Portfolio Management**: Track multiple positions with Fibonacci targets
