# 📈 Sentient Trader Platform

> **A comprehensive, AI-driven options and stock trading platform with real-time analysis, news integration, technical indicators, and intelligent strategy recommendations for automated and manual trading.**

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

## 🆕 Latest Updates (October 2025)

### **Google Gemini 2.5 Flash Integration**
- ✅ **AI Trading Signals** now powered by Gemini for superior buy/sell recommendations
- ✅ **Strategy Analyzer** uses Gemini for advanced bot configuration critique
- ✅ **AI Confidence Scanner** upgraded to Gemini for better stock analysis
- ✅ Configurable via `.env` - easily switch between free and premium models
- ✅ Fixed environment variable handling (`AI_TRADING_MODEL`, `AI_CONFIDENCE_MODEL`, `AI_ANALYZER_MODEL`)

### **Bug Fixes & Improvements**
- 🐛 Fixed `AttributeError` in stock analysis (corrected `sentiment_score` and removed non-existent `social_sentiment`)
- 🐛 Resolved module import issues with `ComprehensiveAnalyzer`
- 🔧 Improved Streamlit cache handling for reliable code reloading
- 📝 Updated documentation with comprehensive AI model configuration guide

**Performance:** Gemini provides 2-3x better analysis quality with ~2 second response times at a fraction of GPT-4 cost.

---

## 🌟 Overview

This platform transforms options trading by combining real-time market data, technical analysis, news sentiment, and AI-powered strategy recommendations into a single, intuitive interface. Built for traders of all experience levels, it integrates seamlessly with Option Alpha's webhook system to automate your trading strategies with comprehensive guardrails and risk management.

### **What Makes This Special?**

- 🔍 **360° Stock Intelligence** - Real-time technical indicators, IV metrics, news, and catalysts
- 🤖 **AI Strategy Advisor** - Personalized recommendations based on 15+ market factors
- 📊 **Multi-Factor Analysis** - Combines RSI, MACD, IV Rank, sentiment, and catalysts
- 📈 **EMA Power Zone & Fibonacci System** - 8-21 EMA reclaim detection, DeMarker timing, A-B-C extension targets
- 🎯 **Multi-Timeframe Alignment** - Weekly/Daily/4H trend confirmation for highest-conviction setups
- 💪 **Sector Relative Strength** - Compare vs sector ETFs and SPY for leading stock selection
- 🧠 **Microsoft Qlib Integration** - Advanced ML models with 158 alpha factors and backtesting (optional)
- 🛡️ **Smart Guardrails** - Built-in risk management and position limits
- 📰 **Live News Integration** - Real-time sentiment analysis from market news
- 📅 **Catalyst Detection** - Automatic earnings and event tracking
- 🔔 **Smart Alerts** - Discord notifications for My Tickers setups + real-time position monitoring
- 🎯 **Option Alpha Integration** - Direct webhook support for automated execution
- 🚀 **Advanced Scanner** - 200+ ticker universe, **social sentiment analysis** (Reddit/Twitter/StockTwits), buzzing stock detection with Crawl4ai web scraping, reverse merger candidates, penny stock risk analysis

---

## 🚀 Quick Start

### Installation

# Sentient Trader Platform

A Streamlit-based research and signal generation tool that combines market data, technical indicators, news sentiment, implied volatility analysis, and an LLM-powered strategy critic to produce actionable options strategy recommendations and signals.

This repository contains a local GUI that helps you research tickers, evaluate option strategies, run pricing and greeks, and build/send validated signals (paper or live) to Option Alpha (via webhook) or Tradier (via API client).

## What this project contains

- `app.py` — Full Streamlit application. Tabs include Stock Intelligence, Strategy Advisor, Signal Builder, Signal History, Strategy Guide, Advanced Analytics (pricing + greeks + binomial convergence tooling), Tradier account helpers, and an LLM Strategy Analyzer panel.
- `llm_strategy_analyzer.py` — Utilities to analyze Option Alpha bot configurations using various LLM providers (OpenAI, Anthropic, Google, OpenRouter). Exposes `LLMStrategyAnalyzer` and example extraction helpers.
- `options_pricing.py` — Black-Scholes pricing + analytical greeks and a Cox-Ross-Rubinstein binomial pricer for American options. Also includes a finite-difference greeks wrapper used by the UI.
- `tradier_client.py` — Lightweight Tradier API client used for paper/live order placement, quotes, and account summary helpers.
- `services/qlib_integration.py` — Microsoft Qlib integration for advanced ML-based stock prediction, 158 alpha factors (Alpha158 dataset), backtesting framework, and rolling model retraining for market adaptation (optional feature).
- `services/ml_enhanced_scanner.py` — **Your ultimate confidence scanner** that combines Qlib ML predictions + LLM reasoning + quantitative analysis for maximum confidence trading decisions. This is the recommended scanner for serious trading.
- `examples/ml_enhanced_trading_workflow.py` — Complete daily trading workflow showing how to use ML-enhanced scanning for the most confident decisions.
- `requirements.txt` — Python dependencies used by the project.
- `tests/` — Pytest tests for pricing and analyzer modules.

## Main features

- Comprehensive on-screen stock analysis (RSI, MACD, support/resistance, IV Rank/Percentile, news & catalysts).
- AI-driven strategy recommendations (several predefined strategies: sell put, sell call, iron condor, credit/debit spreads, straddles, wheel, etc.).
- Signal builder with guardrails (DTE, quantity limits, max daily orders, max daily risk, allowed strategies).
- Paper trading mode (default) and webhook integration for Option Alpha.
- Tradier client helpers (create client from env, validate connection, convert signals to Tradier orders).
- Advanced analytics: option pricing, greeks, binomial convergence benchmarking, and sensitivity explorers.
- **Microsoft Qlib ML enhancements (optional)**: 158 alpha factors, LightGBM/LSTM predictions, backtesting framework, rolling retraining for market adaptation.
- **Alert system**: Discord notifications for technical setups on your saved tickers (My Tickers) + real-time position monitoring with P&L alerts.

## Quickstart (Windows)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Install Crawl4ai for social sentiment analysis:

```powershell
pip install crawl4ai
crawl4ai-setup  # Downloads headless browser for JS rendering
python test_crawl4ai.py  # Test installation
```

4. (Optional) Install Microsoft Qlib for advanced ML features:

```powershell
pip install pyqlib
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

5. Run the app:

```powershell
streamlit run app.py
```

The Streamlit UI opens (default) at http://localhost:8501.

## 🤖 AI Model Configuration (Google Gemini Integration)

The platform now supports **Google Gemini 2.5 Flash** via OpenRouter for all AI-powered features, providing superior analysis quality compared to free models.

### **Three AI Components:**

| Component | Environment Variable | Purpose |
|-----------|---------------------|---------|
| **AI Confidence Scanner** | `AI_CONFIDENCE_MODEL` | High-level stock analysis & confidence scoring |
| **AI Trading Signals** | `AI_TRADING_MODEL` | Generates specific buy/sell signals with entry/exit points |
| **Strategy Analyzer** | `AI_ANALYZER_MODEL` | Critiques trading bot configurations & strategies |

### **Configuration in `.env`:**

```bash
# OpenRouter API Key (required)
OPENROUTER_API_KEY=sk-or-v1-...

# AI Model Configuration (via OpenRouter)
AI_CONFIDENCE_MODEL=google/gemini-2.5-flash
AI_TRADING_MODEL=google/gemini-2.5-flash
AI_ANALYZER_MODEL=google/gemini-2.5-flash
```

### **Available Models:**

**Free Options:**
- `meta-llama/llama-3.1-8b-instruct:free` - Fast, good for basic analysis
- `mistralai/mistral-7b-instruct:free` - Balanced performance
- `huggingfaceh4/zephyr-7b-beta:free` - Lightweight option

**Premium Options (Recommended):**
- `google/gemini-2.5-flash` - **Best value** - Fast, accurate, cost-effective
- `google/gemini-flash-1.5` - Previous generation Gemini
- `openai/gpt-4o` - High quality, higher cost
- `anthropic/claude-3-haiku` - Fast Claude model

### **Why Gemini 2.5 Flash?**

✅ **Superior reasoning** - Better analysis quality than free models  
✅ **Fast responses** - 2-3 second generation time  
✅ **Cost-effective** - Excellent performance-to-cost ratio  
✅ **Consistent output** - Reliable JSON formatting for signals  
✅ **Context understanding** - Better at interpreting complex market conditions  

### **Recent Updates (October 2025):**

- ✅ Fixed `AttributeError` in stock analysis (sentiment/social_sentiment attributes)
- ✅ Integrated Gemini across all three AI components
- ✅ Updated environment variable handling for consistent model configuration
- ✅ Improved error handling and module import reliability

**Find more models at:** https://openrouter.ai/models

## Environment variables

Some integrations require API keys or credentials. You can put them in a `.env` file (project uses python-dotenv) or export them into your environment.

- Option Alpha webhook — Provided in the Streamlit sidebar when sending signals (no env var required).
- LLM provider API keys (only required if you use the Strategy Analyzer):
  - OPENAI_API_KEY — OpenAI API key
  - ANTHROPIC_API_KEY — Anthropic API key
  - GOOGLE_API_KEY — Google Generative API key
  - OPENROUTER_API_KEY — OpenRouter API key
  - Set `LLM_PROVIDER` to one of `openai`, `anthropic`, `google`, `openrouter` if you want to change the default provider.
- Tradier (optional, for paper/live order execution):
  - **Paper Trading:**
    - TRADIER_PAPER_ACCOUNT_ID (or TRADIER_ACCOUNT_ID for backward compatibility)
    - TRADIER_PAPER_ACCESS_TOKEN (or TRADIER_ACCESS_TOKEN for backward compatibility)
    - TRADIER_PAPER_API_URL (optional, defaults to https://sandbox.tradier.com)
  - **Production Trading:**
    - TRADIER_PROD_ACCOUNT_ID
    - TRADIER_PROD_ACCESS_TOKEN
    - TRADIER_PROD_API_URL (optional, defaults to https://api.tradier.com)
- Crawl4ai (optional, for social sentiment analysis):
  - Install with `pip install crawl4ai` and run `crawl4ai-setup`
  - No API keys required, uses public endpoints
  - Requires headless browser (Chromium) for JavaScript rendering
- Microsoft Qlib (optional, for ML-enhanced predictions):
  - Install with `pip install pyqlib` and download data using `python -m qlib.run.get_data`
  - No API keys required, uses local data and models

Example `.env` (do not commit your secrets):

```bash
# OpenRouter API Key (for AI features)
OPENROUTER_API_KEY=sk-or-v1-...

# AI Model Configuration (via OpenRouter)
AI_CONFIDENCE_MODEL=google/gemini-2.5-flash
AI_TRADING_MODEL=google/gemini-2.5-flash
AI_ANALYZER_MODEL=google/gemini-2.5-flash

# Paper Trading (Sandbox)
TRADIER_PAPER_ACCOUNT_ID=ABC123
TRADIER_PAPER_ACCESS_TOKEN=xxxx_paper_token
TRADIER_PAPER_API_URL=https://sandbox.tradier.com

# Production Trading (Live)
TRADIER_PROD_ACCOUNT_ID=XYZ789
TRADIER_PROD_ACCESS_TOKEN=xxxx_prod_token
TRADIER_PROD_API_URL=https://api.tradier.com

# Discord Alerts (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Legacy LLM Keys (optional, for direct API access)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## How to use the app (short walkthrough)

1. Start in Paper Trading Mode (default) — this logs signals locally and prevents real executions.
2. Go to the "Stock Intelligence" tab and enter a ticker. Click Analyze to fetch history, compute indicators, extract news, and produce an AI-style recommendation.
3. Switch to the "Strategy Advisor" tab to input your profile (experience, capital, outlook) and generate personalized strategy suggestions.
4. Use "Generate Trading Signal" to populate a signal (you can load example trades). Click Validate to run guardrails, then Send Signal to log or dispatch the signal.
   - Paper mode logs to session state and keeps a local signal history.
   - Live mode will attempt to POST to the provided Option Alpha webhook URL or use the Tradier client if you integrate it.
5. Inspect and export your signals under "Signal History". You can edit fields in the interactive data editor and export CSV/JSON.
6. Advanced users: Use the "Strategy Analyzer" to analyze Option Alpha bot configs with an LLM. The analysis requires valid API credentials for your chosen provider.

## Testing

Run unit tests with pytest:

```powershell
pytest -q
```

Tests cover pricing functions and some analyzer helpers. They are useful to validate changes to pricing/greeks logic.

## Options concepts explained (practical glossary)

This short reference explains the most important options concepts used throughout the app. Each entry explains the meaning, why it matters, and a short practical tip.

### DTE — Days to Expiration

- What it is: Number of calendar days remaining until the option contract expires.
- Why it matters: DTE controls time value (theta). Shorter DTE means faster time decay and typically lower premium for the same strike. Many strategies choose a DTE band to target (e.g., 7–45 days).
- Practical tip: Use shorter DTE if you want quick time decay (selling premium), and longer DTE to reduce gamma risk for directional buys.

### Implied Volatility (IV)

- What it is: The market's expectation of future volatility embedded in option prices. IV is inferred from market option prices using a pricing model.
- Why it matters: Higher IV → higher option premiums → better for premium sellers. Lower IV → cheaper options → better for buyers.
- Practical tip: Always compare IV to historical volatility and to the rest of the market (IV Rank) before deciding to buy or sell premium.

### IV Rank and IV Percentile

- What IV Rank means: IV Rank is a normalized measure that shows where current IV sits relative to its historical range (commonly the past 52 weeks). If IV Rank is 80, current IV is high relative to the last year.
- IV Percentile: Similar idea; percentile of daily IV values over a lookback window.
- Why it matters: IV Rank tells you whether option prices are unusually expensive or cheap for this underlying. High IV Rank favors selling premium; low IV Rank favors buying.
- Practical tip: Use IV Rank thresholds in the app (e.g., >60 = prefer selling strategies, <40 = buying strategies).

### Theta — Time Decay

- What it is: Theta is the rate at which an option's extrinsic value decays as time passes (usually quoted per day).
- Why it matters: Theta works against buyers (they lose time value each day) and helps sellers (they collect decay). Theta accelerates as DTE approaches zero, especially in the last 30 days.
- Practical tip: If you sell premium, you're collecting theta — but beware that gamma increases near expiry, making large moves more impactful.

### Vega — Sensitivity to Volatility

- What it is: Vega measures how much an option's price will change for a 1 percentage-point change in IV (often expressed per 1 vol point).
- Why it matters: Options with higher vega are more sensitive to IV moves. Long options gain when IV rises; short options lose when IV rises.
- Practical tip: When trading around events (earnings), expect vega to move; selling premium into high vega can be profitable if realized volatility stays lower than implied.

### Delta & Gamma — Directional Risk and Convexity

- Delta: The option's approximate sensitivity to a small move in the underlying (0–1 for calls, -1–0 for puts). Delta can be interpreted as the approximate probability the option finishes in the money for short-dated options.
- Gamma: The rate of change of delta as the underlying moves. High gamma means delta can move quickly — this increases risk for short options and potential reward for long options.
- Practical tip: Long options gain gamma (benefit from large moves) while short options carry gamma risk near expiry.

### Intrinsic vs Extrinsic (Time) Value

- Intrinsic value: The immediate exercise value (for a call: underlying - strike if positive). Intrinsic cannot be negative.
- Extrinsic value: The remainder of the option premium (price - intrinsic). This includes time value and implied vol component.
- Practical tip: Deep in-the-money options are largely intrinsic (less sensitivity to IV), while out-of-the-money options are mostly extrinsic (more sensitive to IV and theta).

### Skew and Term Structure

- Skew: The difference in implied volatility across strikes. Equity indices usually exhibit higher IV for lower strikes (put skew) reflecting demand for downside protection.
- Term structure: How IV changes across expirations (near-term vs back-term). A steep term structure means near-term IV differs a lot from back-term IV.
- Practical tip: Use skew and term-structure awareness to design spread trades (e.g., diagonal spreads) that exploit relative mispricings.

### Probability, Breakeven & Moneyness

- Moneyness: Relationship between spot and strike (ITM, ATM, OTM). ATM options have the most extrinsic value and highest vega.
- Breakeven: For a long call, breakeven = strike + premium paid; for a short option, breakeven is adjusted accordingly.
- Practical tip: Translate model outputs (delta, P&L scenarios) into simple breakeven levels before placing a trade.

### Example: How terms drive strategy selection

- Selling premium (e.g., iron condor, short strangle): Prefer high IV Rank, low likelihood of a large move, and enough DTE to collect theta but not so short that gamma risk explodes.
- Buying options (e.g., long calls/puts, straddles): Prefer low IV (cheap premium) or a specific event where you expect realized vol > implied vol. Choose DTE to balance time decay vs gamma exposure.

### Quick rules of thumb

- IV Rank > 60 → consider premium-selling or defined-risk credit strategies.
- IV Rank < 40 → consider buying directional options or debit spreads.
- DTE 7–45 → common sweet spot for many income-oriented strategies (sells collect theta but avoid extreme gamma).
- Always check upcoming catalysts (earnings, dividends) — they can invalidate otherwise-sensible trades.

### Further reading

- Options as a Strategic Investment (Lawrence McMillan) — classic reference
- Option Volatility & Pricing (Sheldon Natenberg) — deep dive on IV and greeks
- Practical online resources: CME education pages, OCC options education, and broker option education centers

## Microsoft Qlib ML Enhancement (Optional)

The platform now supports optional integration with Microsoft's Qlib quantitative investment platform for advanced ML capabilities:

### **What Qlib Adds:**

- **Alpha158 Features**: 158 technical/fundamental factors vs standard 10-15 indicators
- **ML Models**: LightGBM, LSTM, Transformer models trained on historical data
- **Backtesting**: Comprehensive strategy backtesting with realistic execution simulation
- **Market Adaptation**: Rolling model retraining to adapt to changing market dynamics
- **Ensemble Predictions**: Combine multiple ML models for robust forecasting

### **Setup:**

```powershell
# Install Qlib
pip install pyqlib

# Download US stock data
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us

# Use in code
from services.qlib_integration import QLIbEnhancedAnalyzer
analyzer = QLIbEnhancedAnalyzer()
features = analyzer.get_alpha158_features('AAPL')  # 158 alpha factors
prediction = analyzer.get_ml_prediction('AAPL')    # ML-based score
```

### **Key Capabilities:**

- Extract 158 alpha factors (price/volume momentum, volatility, cross-sectional features)
- Train ML models on historical data for price/return prediction
- Backtest strategies with transaction costs and slippage
- Rolling retraining to handle non-stationary market conditions
- Ensemble multiple models (LightGBM + LSTM + Transformer)

**Note**: Qlib is optional. The app works fully without it using rule-based analysis.

## 📊 Advanced EMA/DeMarker/Fibonacci Trading System

The platform now includes a sophisticated swing and options trading framework based on the **8-21 EMA Power Zone**, **DeMarker oscillator**, and **Fibonacci A-B-C extensions**. This system provides precise entry timing, dynamic stops, and target-based scale-outs.

### **Core Components:**

#### **1. EMA(8/21) Power Zone & Reclaim**

- **Power Zone**: Price > EMA8 > EMA21 signals strong uptrend momentum
- **Reclaim**: Prior close below EMAs → current close above both with rising EMAs, volume surge, and follow-through
- **Usage**: High-probability swing entries and options structure selection

#### **2. DeMarker(14) Oscillator**

- **Range**: 0-1 oscillator measuring buying/selling pressure
- **Oversold**: ≤ 0.30 in uptrend = pullback entry opportunity
- **Overbought**: ≥ 0.70 in downtrend = bounce short opportunity
- **Usage**: Precision timing for swing trades and options entries

#### **3. Fibonacci A-B-C Extensions**

- **Pattern**: A (low) → B (high) → C (pullback) → Targets
- **Targets**:
  - T1 (127.2%): Take 25% profit
  - T2 (161.8%): Take 50% profit
  - T3 (200-261.8%): Trail remaining with stop below 21 EMA
- **Usage**: Strike selection for options spreads, scale-out planning

#### **4. Multi-Timeframe Alignment**

- **Analysis**: Weekly, Daily, 4-Hour trend confirmation using EMA8/21
- **Alignment Score**: 0-100% showing agreement across timeframes
- **Aligned**: ≥66.7% (2 of 3 timeframes agree)
- **Usage**: Filter for highest-confidence setups, boost confidence score

#### **5. Sector Relative Strength**

- **Comparison**: Stock vs Sector ETF vs SPY (3-month performance)
- **RS Score**: 0-100 scale (50 = market neutral, >60 = outperforming)
- **Usage**: Trade leading stocks in leading sectors, avoid laggards

### **How It Works in Each Trading Style:**

#### **Swing Trading**

```
✅ Entry Checklist:
- EMA Power Zone or Reclaim confirmed
- DeMarker ≤ 0.30 in uptrend (pullback entry)
- Multi-timeframe alignment (2+ agree)
- Sector RS > 60 (relative strength)

🎯 Targets & Stops:
- Stop: Just below 21 EMA (dynamic)
- T1 (127.2%): Take 25% → Move stop to breakeven
- T2 (161.8%): Take 50% → Trail below 21 EMA
- T3 (200-261.8%): Trail remaining 25%

⏱️ Hold Time: 3-10 days
```

#### **Day Trading**

```
✅ Power Zone Filter:
- 8>21 and price above both → Favor long setups
- Intraday pullbacks to EMA8/21 = buying opportunities
- Exit by market close

🛡️ Risk Management:
- Stop Loss: 0.5-1%
- Take Profit: 1-3%
- Trade with momentum, not against
```

#### **Options Trading (Enhanced)**

```
🔥 EMA Reclaim Signals:
- Reclaim + High IV (>60): SELL cash-secured puts at 21 EMA (30-45 DTE)
- Reclaim + Low IV (<40): BUY call spreads ATM to T1 strike (45-60 DTE)

📐 Fibonacci-Based Strike Selection:
- Long call spread: Buy ATM, Sell at T1 (127.2% target)
- Time to target: 30-60 DTE
- Bull put spread: Sell 10-15% OTM below C (support)

🎯 DeMarker Timing:
- DeMarker ≤ 0.30 in uptrend: Time call entries for bounce
- Ideal for 0-14 DTE scalps or 30-45 DTE swings

📊 Multi-Timeframe Filter:
- Only trade options when 2+ timeframes aligned
- Higher alignment = higher confidence = larger position size
```

### **Confidence Score Enhancement:**

The confidence calculation now incorporates:

- **Base Score**: RSI, MACD, IV Rank, sentiment, catalysts, earnings risk (up to 90 points)
- **Timeframe Bonus**: +10 points for aligned trends across timeframes
- **Sector RS Bonus**: +10 points for strong relative strength (RS > 60)
- **Maximum Score**: 100 (requires all factors aligned)

**Confidence Levels:**

- **85-100**: Highest conviction trades (all systems agree)
- **75-84**: Strong setups (good alignment)
- **60-74**: Decent setups (monitor closely)
- **<60**: Wait for better setup

### **Quick Start Example:**

```python
from analyzers.comprehensive import ComprehensiveAnalyzer

# Analyze with all new indicators
analysis = ComprehensiveAnalyzer.analyze_stock("AAPL", trading_style="SWING_TRADE")

# Check Power Zone
if analysis.ema_power_zone:
    print("✅ Power Zone Active")

# Check Reclaim
if analysis.ema_reclaim:
    print("🔥 EMA Reclaim Confirmed - High Probability Setup")

# DeMarker timing
if analysis.demarker <= 0.30:
    print("🎯 DeMarker Oversold - Entry Zone")

# Fibonacci targets
if analysis.fib_targets:
    print(f"T1: ${analysis.fib_targets['T1_1272']:.2f}")
    print(f"T2: ${analysis.fib_targets['T2_1618']:.2f}")
    print(f"T3: ${analysis.fib_targets['T3_2618']:.2f}")

# Timeframe alignment
if analysis.timeframe_alignment and analysis.timeframe_alignment['aligned']:
    print(f"✅ Timeframes Aligned: {analysis.timeframe_alignment['alignment_score']:.1f}%")

# Sector strength
if analysis.sector_rs and analysis.sector_rs['rs_score'] > 60:
    print(f"💪 Outperforming Sector: RS {analysis.sector_rs['rs_score']:.1f}")
```

### **Testing the New Indicators:**

```powershell
# Run comprehensive test suite
pytest tests/test_technical_indicators.py -v

# Run specific test classes
pytest tests/test_technical_indicators.py::TestEMAPowerZoneAndReclaim -v
pytest tests/test_technical_indicators.py::TestFibonacciExtensions -v
```

---

## 🚨 Phase 3: Advanced Automation Features

Phase 3 adds powerful automation and validation tools to maximize the EMA/Fibonacci system's effectiveness.

### **1. Real-Time Alert System** 🔔

Automatically detect and notify when high-probability setups occur.

**Alert Types:**

- **CRITICAL**: Triple Threat (Reclaim + Aligned + Strong RS), EMA Reclaim
- **HIGH**: Timeframe Alignment, Sector Leaders, Fibonacci Setups
- **MEDIUM**: DeMarker Entry Signals

**Usage:**

```python
from services.alert_system import get_alert_system, SetupDetector, console_callback

# Setup alerts with console notifications
alert_system = get_alert_system()
alert_system.add_callback(console_callback)

# Analyze and generate alerts
detector = SetupDetector(alert_system)
analysis = ComprehensiveAnalyzer.analyze_stock("AAPL", "SWING_TRADE")
alerts = detector.analyze_for_alerts(analysis)

# Get critical alerts
critical = alert_system.get_recent_alerts(priority=AlertPriority.CRITICAL)
```

**Custom Callbacks:**

- Email notifications
- SMS alerts
- Webhook integration (Slack, Discord, Telegram)
- Custom logging

**Alert Log:** All alerts saved to `logs/trading_alerts.json` for review.

### **2. EMA Reclaim + Fibonacci Backtesting** 📊

Validate strategy performance with historical data before trading.

**Features:**

- Configurable filters (Reclaim, Alignment, Sector RS requirements)
- Fibonacci T1/T2/T3 profit-taking
- Dynamic stops below 21 EMA
- Comprehensive metrics: Win rate, profit factor, Sharpe ratio, max drawdown
- Setup-specific performance tracking

**Quick Start:**

```python
from services.backtest_ema_fib import EMAFibonacciBacktester

# Create backtester with filters
bt = EMAFibonacciBacktester(
    initial_capital=10000,
    position_size_pct=0.10,
    require_reclaim=True,
    require_alignment=True,
    use_fibonacci_targets=True
)

# Run backtest
results = bt.backtest("AAPL", "2023-01-01", "2024-10-01")
results.print_summary()

# Optimize parameters
optimization = bt.optimize_parameters("AAPL", "2023-01-01", "2024-10-01")
print("Best Parameters:", optimization['best_params'])
```

**Performance Metrics:**

- Overall: Total trades, win rate, P&L, avg win/loss
- Risk: Max drawdown, Sharpe ratio, profit factor
- Setup-Specific: Win rates for reclaim, aligned, triple threat
- Trade Details: Entry/exit, hold time, exit reason

### **3. Preset Scan Filters** 🔍

Rapidly identify opportunities with pre-configured scanners.

**Available Presets:**

- `TRIPLE_THREAT` - Reclaim + Aligned + Strong RS (highest conviction)
- `EMA_RECLAIM` - EMA reclaim confirmations
- `TIMEFRAME_ALIGNED` - Multi-timeframe agreement
- `SECTOR_LEADERS` - RS > 70 outperformers
- `DEMARKER_PULLBACK` - DeMarker oversold in uptrend
- `FIBONACCI_SETUP` - A-B-C patterns detected
- `HIGH_CONFIDENCE` - Confidence ≥ 85
- `POWER_ZONE` - 8>21 EMA momentum
- `OPTIONS_PREMIUM_SELL` - High IV + Power Zone
- `OPTIONS_DIRECTIONAL` - Low IV + Reclaim/Aligned

**Usage:**

```python
from services.preset_scanners import PresetScanner, ScanPreset, get_high_volume_tech

scanner = PresetScanner()
tickers = get_high_volume_tech()

# Scan with specific preset
results = scanner.scan(tickers, ScanPreset.TRIPLE_THREAT)
scanner.print_results(results, show_details=True)

# Get top 10 opportunities across all presets
top_opps = scanner.get_top_opportunities(tickers, top_n=10)

# Scan all presets
all_results = scanner.scan_all_presets(tickers)
```

**Built-in Watchlists:**

- `get_sp500_tickers()` - S&P 500 stocks
- `get_nasdaq100_tickers()` - NASDAQ 100 stocks
- `get_high_volume_tech()` - High volume tech stocks

### **4. Fibonacci-Based Options Chain Integration** 📈

Auto-populate option strikes and suggest spreads based on Fibonacci levels.

**Features:**

- Automatically finds strikes near Fib A, B, C, T1, T2, T3
- Suggests spreads based on trend and IV context
- Real-time options data from yfinance
- Risk metrics: max profit/loss, breakeven, probability

**Usage:**

```python
from services.options_chain_fib import FibonacciOptionsChain

# Analyze for Fibonacci pattern
analysis = ComprehensiveAnalyzer.analyze_stock("AAPL", "OPTIONS")

if analysis.fib_targets:
    # Get options chain
    fib_chain = FibonacciOptionsChain("AAPL")
  
    # Find strikes near Fibonacci levels (45 DTE)
    fib_strikes = fib_chain.find_strikes_near_fibonacci(
        analysis.fib_targets, 
        target_dte=45
    )
    fib_chain.print_fibonacci_strikes(fib_strikes)
  
    # Get spread suggestions
    spreads = fib_chain.suggest_fibonacci_spreads(
        analysis.fib_targets, 
        analysis, 
        target_dte=45
    )
    fib_chain.print_spread_suggestions(spreads)
  
    # Get best spread by risk/reward
    best = max(spreads, key=lambda s: s.risk_reward_ratio)
```

**Auto-Generated Spreads:**

- **Bull Call Spread (T1)**: Buy ATM, Sell at T1 (127.2%) - Uptrend setups
- **Bull Put Spread (C Support)**: Sell at C, Buy lower - EMA Reclaim + High IV
- **Wide Call Spread (T2)**: Buy ATM, Sell at T2 (161.8%) - Strong uptrend

**Strike Information:**

- Strike price (closest to Fib level)
- Distance from spot (%)
- Bid/Ask spreads
- Open interest (liquidity)
- Greeks (Delta, Theta, Vega)

### **Complete Daily Workflow Example:**

```python
# 1. Scan for high-conviction setups
scanner = PresetScanner()
top_setups = scanner.get_top_opportunities(get_high_volume_tech(), top_n=5)

# 2. Backtest each opportunity
bt = EMAFibonacciBacktester(require_reclaim=True, require_alignment=True)

for setup in top_setups:
    results = bt.backtest(setup.ticker, "2023-01-01", "2024-10-01")
  
    # 3. If backtest is positive, analyze options
    if results.win_rate > 60 and results.profit_factor > 1.5:
        if setup.analysis.fib_targets:
            fib_chain = FibonacciOptionsChain(setup.ticker)
            spreads = fib_chain.suggest_fibonacci_spreads(
                setup.analysis.fib_targets, setup.analysis, 45
            )
          
            # 4. Execute best spread
            if spreads:
                best = max(spreads, key=lambda s: s.risk_reward_ratio)
                print(f"✓ TRADE: {best.strategy_name}")
```

### **Run Complete Demo:**

```powershell
python examples/phase3_advanced_features.py
```

This demonstrates all four features with live examples.

**Documentation:** See `documentation/PHASE3_ADVANCED_FEATURES.md` for complete guide.

---

## 📲 Alert System: My Tickers & Position Monitoring

The platform includes a comprehensive alert system that sends Discord/webhook notifications for both **pre-trade setups** and **active position updates**.

### **Two Alert Types:**

#### **1. Technical Setup Alerts** 🎯

Pre-trade signals based on EMA reclaims, timeframe alignment, and other technical patterns.

**Default**: Alerts for ALL scanned tickers
**Filtered**: Enable `my_tickers_only=True` to only alert for tickers in your **⭐ My Tickers** list (the ones you save in the app)

```python
from services.preset_scanners import PresetScanner
from services.ticker_manager import TickerManager

ticker_manager = TickerManager()

# Only alert for your saved tickers (TSLA, AMC, AMD, etc.)
scanner = PresetScanner(
    my_tickers_only=True,
    ticker_manager=ticker_manager
)

# Scan 500 stocks - only your My Tickers trigger alerts
scanner.scan(sp500_tickers, ScanPreset.TRIPLE_THREAT)
```

**Alert Priorities:**

- **CRITICAL**: Triple Threat, EMA Reclaim
- **HIGH**: Timeframe Aligned, Sector Leaders, Fibonacci Setups
- **MEDIUM**: DeMarker Entry Signals

#### **2. Position Monitoring Alerts** 📊

Real-time updates on ALL your active trades (from Tradier).

```python
from services.position_monitor import get_position_monitor
from services.alert_system import get_alert_system

position_monitor = get_position_monitor(
    alert_system=get_alert_system(),
    tradier_client=tradier_client
)

# Configure thresholds
position_monitor.pnl_alert_threshold = 5.0  # Alert every 5% P&L change
position_monitor.significant_loss_threshold = -10.0  # Alert on 10% loss
position_monitor.significant_gain_threshold = 15.0  # Alert on 15% gain

# Update positions (call periodically)
success, alerts = position_monitor.update_positions()
```

**Position Alert Types:**

- ✅ **Position Opened** - When new trade detected
- ❌ **Position Closed** - Final P&L on exit
- 🎯 **Profit Target Hit** - Price reaches target
- 🛑 **Stop Loss Hit** - Stop triggered
- 📊 **Position Updates** - Regular P&L changes (every 5%)
- ⚠️ **Significant Loss** - Alert on -10% or worse
- 🚀 **Significant Gain** - Alert on +15% or better

### **Managing Your "My Tickers"**

Add/remove tickers via the **⭐ My Tickers** tab in the app, or programmatically:

```python
from services.ticker_manager import TickerManager

tm = TickerManager()

# Add ticker
tm.add_ticker('NVDA', name='NVIDIA', ticker_type='stock')

# Remove ticker
tm.remove_ticker('NVDA')

# View all
my_tickers = tm.get_all_tickers()
print([t['ticker'] for t in my_tickers])  # ['TSLA', 'AMC', 'AMD', 'COIN']
```

### **Discord Integration**

Both alert types send to the same Discord webhook (configured in your `.env`):

```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### **Demo Scripts**

```powershell
# Test position monitoring
python examples/position_monitoring_demo.py --mode monitor

# View your My Tickers setup
python examples/my_tickers_alert_demo.py --mode show
```

### **Recommended Setup**

```python
# Technical alerts: My Tickers ONLY (stocks you're watching)
scanner = PresetScanner(my_tickers_only=True, ticker_manager=ticker_manager)

# Position alerts: ALL trades (everything you have open)
position_monitor = get_position_monitor(alert_system, tradier_client)

# Result:
# ✅ Pre-trade alerts for stocks you're actively following
# ✅ Active trade alerts for all your positions
```

---

## 🎯 ML-Enhanced Trading Workflow (Recommended)

For the **most confident trading decisions**, use the ML-Enhanced Scanner that combines three analysis systems:

### **Triple-Validation System:**

1. **ML Analysis (40% weight)** - Qlib models trained on 158 alpha factors

   - Detects complex patterns across price, volume, momentum, volatility
   - Predicts probability of profitable moves
   - Adapts to changing market conditions
2. **LLM Analysis (35% weight)** - GPT/Claude/Llama reasoning

   - Provides contextual understanding and risk analysis
   - Explains *why* a trade makes sense
   - Identifies hidden risks and catalysts
3. **Quantitative Analysis (25% weight)** - Traditional technical signals

   - RSI, MACD, IV Rank, volume analysis
   - Proven indicators and patterns
   - Fast execution and reliability

### **Your Daily Workflow:**

```powershell
# Run the ML-enhanced daily scan
python examples/ml_enhanced_trading_workflow.py

# Save results to watchlist
python examples/ml_enhanced_trading_workflow.py --save

# Backtest the strategy
python examples/ml_enhanced_trading_workflow.py --backtest
```

### **In Python Code:**

```python
from services.ml_enhanced_scanner import MLEnhancedScanner

# Initialize scanner (auto-detects ML and LLM availability)
scanner = MLEnhancedScanner()

# Get highest confidence options trades
# Only returns trades scoring high on ALL THREE systems
trades = scanner.scan_top_options_with_ml(
    top_n=10,
    min_ensemble_score=70.0  # 70% combined confidence minimum
)

# Analyze top pick
for trade in trades:
    print(f"{trade.ticker}: Combined Score {trade.combined_score:.1f}")
    print(f"  ML: {trade.ml_prediction_score:.1f}")
    print(f"  LLM: {trade.ai_rating * 10:.1f}")
    print(f"  Quant: {trade.score:.1f}")
  
    # Get detailed explanation
    explanation = scanner.get_trade_explanation(trade)
    print(explanation)
```

### **Why This Workflow Works:**

- **Higher Win Rate**: Three independent systems must agree
- **Better Risk Management**: LLM identifies risks ML might miss
- **Adaptive**: ML retrains on recent data, adapting to market changes
- **Explainable**: You know exactly WHY each trade is recommended
- **Backtestable**: Validate strategy performance before live trading

### **Confidence Levels Explained:**

- **VERY HIGH (85+)**: All three systems highly confident - **strongest trades**
- **HIGH (75-84)**: Strong agreement across systems - **good trades**
- **MEDIUM-HIGH (60-74)**: Decent signals but monitor closely
- **Below 60**: Too much uncertainty, avoid or wait

### **Risk Management:**

- Never risk more than 2% of capital per trade
- Size positions inversely to risk level (High Risk = smaller size)
- Use LLM risk analysis to set appropriate stop losses
- Diversify across multiple high-confidence opportunities
- Backtest before deploying new strategies

---

## 🚀 Advanced Opportunity Scanner

The **Advanced Opportunity Scanner** is a powerful tool for discovering top trading opportunities with customizable filters. Perfect for catching buzzing stocks and obscure plays before they rocket!

### **Key Features:**

- **Multiple Scan Types**:
  - 🎯 All Opportunities - Comprehensive scan
  - 📈 Options Plays - High IV and volume setups
  - 💰 Penny Stocks (<$5) - Low-price high-potential plays
  - 💥 Breakouts - EMA reclaims and technical breakouts
  - 🚀 Momentum Plays - Strong price movers
  - 🔥 Buzzing Stocks - Unusual activity detection

- **Advanced Filtering**:
  - Price range ($0.10 to $500+)
  - Volume surge detection (2x, 3x, 5x avg)
  - Momentum filters (% change thresholds)
  - Score and confidence minimums
  - Technical filters (Power Zone, EMA Reclaim, Timeframe Alignment)
  - RSI range selection
  - IV Rank filters for options

- **Buzzing Stock Detection with Social Sentiment (Crawl4ai)**:
  - **Technical Analysis (60%)**: Volume surge, volatility spikes, consecutive moves, gaps
  - **Social Sentiment (40%)**: Reddit (r/wallstreetbets, r/stocks), Twitter/X (via Nitter), StockTwits, financial news
  - **No Login Required**: Public endpoints and mirrors for all sources
  - **JavaScript Rendering**: Full browser crawling for dynamic content
  - **BM25 Relevance Scoring**: Find most relevant discussions automatically
  - **Sentiment Analysis**: Bullish/bearish keyword detection
  - Combined buzz score (0-100) with social + technical indicators
  - Reverse merger candidate detection

- **Penny Stock Risk Analysis**:
  - Reverse split history tracking (3-year lookback)
  - Automatic dilution risk warnings
  - Split frequency severity assessment
  - Critical for sub-$1 micro-pennies

- **Extended Universe**:
  - 200+ tickers including obscure plays
  - Large cap tech, growth stocks, meme stocks
  - EV/clean energy, biotech/pharma
  - Crypto-related, AI/tech emerging
  - Cannabis, small cap high volatility
  - SPACs and special situations

### **Quick Filter Presets:**

- **High Confidence Only** (Score ≥70) - Most reliable setups
- **Ultra-Low Price** (<$1) - Maximum upside potential
- **Penny Stocks** ($1-$5) - Classic penny stock range
- **Volume Surge** (>2x avg) - Strong interest plays
- **Strong Momentum** (>5% change) - Active movers
- **Power Zone Stocks** - EMA 8>21 setups
- **EMA Reclaim Setups** - High-probability entries

### **Usage in App:**

1. Navigate to **🚀 Advanced Scanner** tab
2. Select scan type (Options, Penny Stocks, Buzzing, etc.)
3. Choose a quick filter preset or customize advanced filters
4. Set number of results (5-50)
5. Click **🔍 Scan Markets**
6. Review opportunities with:
   - Score, price, volume metrics
   - Breakout and buzzing indicators
   - Risk/confidence levels
   - Sector and market cap info
7. Export results to CSV for further analysis
8. Add promising tickers to **⭐ My Tickers** for alerts

### **Programmatic Usage:**

```python
from services.advanced_opportunity_scanner import (
    AdvancedOpportunityScanner, 
    ScanFilters, 
    ScanType
)

# Initialize scanner
scanner = AdvancedOpportunityScanner(use_ai=True)

# Create filters
filters = ScanFilters(
    min_price=1.0,
    max_price=5.0,
    min_volume_ratio=2.0,  # 2x average volume
    min_score=65.0,
    require_power_zone=True,
    min_rsi=30,
    max_rsi=70
)

# Scan for penny stock breakouts
opportunities = scanner.scan_opportunities(
    scan_type=ScanType.PENNY_STOCKS,
    top_n=20,
    filters=filters,
    use_extended_universe=True
)

# Display results
for opp in opportunities:
    print(f"{opp.ticker}: Score {opp.score:.1f} | ${opp.price:.2f} ({opp.change_pct:+.1f}%)")
    print(f"  Reason: {opp.reason}")
    if opp.is_breakout:
        print(f"  💥 BREAKOUT: {', '.join(opp.breakout_signals)}")
    if opp.is_buzzing:
        print(f"  🔥 BUZZING: {', '.join(opp.buzz_reasons)}")
```

### **Buzzing Stocks Detection with Social Sentiment:**

The buzzing scanner now uses **Crawl4ai** for comprehensive social sentiment analysis across multiple platforms:

```python
# Scan for stocks showing unusual activity + social buzz
buzzing = scanner.scan_buzzing_stocks(
    top_n=20,
    lookback_days=5,
    min_buzz_score=30.0  # Lower threshold for broader results
)

for stock in buzzing:
    print(f"{stock.ticker}: Buzz Score {stock.buzz_score:.0f}")
    print(f"  {', '.join(stock.buzz_reasons)}")
    # Example reasons:
    # "Volume surge 3.2x average"
    # "📱 25 Reddit mentions (BULLISH)"
    # "🐦 18 Twitter mentions"
    # "💬 12 StockTwits messages (BULLISH)"
    # "📰 8 news articles"
    # "🔥 Trending (score: 78)"
```

### **What Makes a Stock "Buzzing"?**

The buzz detection combines **technical analysis (60%)** + **social sentiment (40%)**:

#### **Technical Buzz (60% weight):**
- **Volume Surge (40 pts)**: Recent volume 2-3x+ average
- **Volatility Spike (30 pts)**: Price swings 1.5-2x+ normal
- **Consecutive Moves (20 pts)**: Multiple days of >2% moves
- **Gap Moves (10 pts)**: Gap up/down >3-5%

#### **Social Sentiment (40% weight) - NEW:**
- **Reddit mentions (0-50 pts)**: 10 popular subreddits (wallstreetbets, stocks, investing, options, StockMarket, pennystocks, Daytrading, swingtrading, wallstreetbetsOGs, thetagang) - searches for both `$TICKER` and `TICKER`
- **Twitter mentions (0-30 pts)**: Via Nitter mirrors (no login required) - searches for `$TICKER` first (standard Twitter format)
- **StockTwits sentiment (0-20 pts)**: Real-time trader sentiment with bullish/bearish tags
- **Financial news (0-20 pts)**: Article volume from Yahoo Finance, Seeking Alpha
- **Sentiment boost (0-20 pts)**: Bullish vs bearish keyword analysis

**Combined Buzz Score Interpretation:**
- **75-100**: Extremely hot - Major technical + social attention
- **60-74**: Very buzzing - Strong interest across sources
- **50-59**: Buzzing - Elevated activity
- **30-49**: Moderate interest - Worth monitoring
- **<30**: Normal activity

### **Social Sentiment Data Sources:**

| Source | Access Method | Search Format | Login Required | Coverage |
|--------|--------------|---------------|----------------|----------|
| **Reddit** | Crawl4ai + BM25 scoring | `$TICKER OR TICKER` | ❌ No | 10 subreddits, 5 checked per scan, 30 posts each |
| **Twitter/X** | Nitter mirrors | `$TICKER OR TICKER` | ❌ No | Multiple mirror instances, 20+ tweets |
| **StockTwits** | Public streams | `TICKER` | ❌ No | Real-time stream, 30+ messages |
| **News** | yfinance API | `TICKER` | ❌ No | Yahoo Finance, 10+ articles |

**Features:**
- ✅ **JavaScript Rendering**: Access dynamic/JS-heavy sites
- ✅ **Stealth Mode**: Bypass bot detection
- ✅ **BM25 Link Scoring**: Find most relevant discussions
- ✅ **Parallel Crawling**: Fast multi-source scraping
- ✅ **Sentiment Analysis**: Bullish/bearish keyword matching
- ✅ **No Login Walls**: Uses public endpoints and mirrors

**Excluded Sources:**
- ❌ Instagram (not relevant for trading)
- ❌ Facebook (minimal trading discussion)

### **Installation:**

The social sentiment analyzer requires **Crawl4ai**:

```powershell
# Install Crawl4ai
pip install crawl4ai

# Setup browser for JS rendering
crawl4ai-setup

# Test installation
python test_crawl4ai.py
```

### **Example Output:**

```
NVDA - Buzz Score: 85.4/100 (VERY HIGH)
├─ Volume surge 3.2x average
├─ High volatility spike
├─ 📱 25 Reddit mentions (BULLISH)
│  └─ "NVDA calls printing 🚀 earnings crush expected"
├─ 🐦 18 Twitter mentions
│  └─ "NVDA breaking ATH, AI demand unstoppable"
├─ 💬 12 StockTwits messages (BULLISH)
│  └─ 83% bullish sentiment
├─ 📰 8 news articles
│  └─ "NVIDIA Earnings Beat, Revenue Guidance Raised"
└─ 🔥 Trending (score: 78)
```

### **Performance:**
- **Single ticker**: 10-20 seconds (comprehensive analysis)
- **5 tickers**: 1-2 minutes
- **20 tickers**: 4-8 minutes

**Note**: Slower than technical-only analysis but provides comprehensive social sentiment data.

### **Reverse Split Tracking (Penny Stocks):**

The scanner automatically detects reverse stock splits in penny stocks, especially critical for sub-$1 micro-pennies. Reverse splits often indicate financial distress and can be warning signs of dilution risk.

**What's Tracked:**
- **Reverse Split History**: Last 3 years of reverse splits with dates and ratios
- **Recent Reverse Splits**: Flags splits within last 12 months
- **Risk Warnings**: Automatic severity assessment based on split frequency

**Warning Levels:**
- ⚠️ **HIGH RISK**: 3+ reverse splits in 3 years - Extreme caution advised
- ⚠️ **CAUTION**: 2 reverse splits in 3 years - High dilution risk
- ⚠️ **Recent**: Reverse split within last year - Monitor closely
- **Previous**: Historical split detected - Note for context

**Example Output:**
```
MULN: Score 65.2 | $0.45 (+8.3%)
  ⚠️ 3 reverse splits in 3y - HIGH RISK
  Split History: 1:10 on 2024-03-15, 1:25 on 2023-08-22, 1:15 on 2022-11-10
```

**Why This Matters:**
- Companies with multiple reverse splits often dilute shareholders repeatedly
- Sub-$1 stocks with split history have higher bankruptcy risk
- Pattern indicates inability to maintain listing requirements
- Useful for risk assessment before entering penny stock positions

**Usage in Scan Results:**
- Prominently displayed in red error boxes for penny stocks
- Included in CSV export with split count and warning
- Filters opportunities by reverse split risk tolerance

### **Reverse Merger Candidate Detection:**

The scanner identifies potential reverse merger candidates using speculation, sentiment analysis, and corporate indicators. Reverse mergers involve shell companies merging with private companies to go public faster than traditional IPOs.

**Detection Algorithm (Score 0-100):**

**1. Shell Company Indicators (40 points)**
- Micro-cap (<$50M) = 20 pts, Small-cap (<$100M) = 10 pts
- Low average volume (<100k) = 10 pts (low liquidity)
- No/minimal revenue = 10 pts (shell characteristic)

**2. Recent Unusual Activity (30 points)**
- Massive volume spike (>5x) = 20 pts
- Unusual volume (>3x) = 10 pts  
- Large price move (>20%) = 10 pts

**3. Sentiment & News Indicators (30 points)**
- High speculation sentiment + micro-cap = 15 pts
- Merger-related news keywords detected = 15 pts
  - Keywords: "merger", "acquisition", "reverse merger", "SPAC", "combination", "transaction", "deal", "takeover", "agreement"
- High news activity (>5 recent items) = 5 pts

**Merger Candidate Threshold:**
- Score ≥50 = Flagged as merger candidate
- Score 50-69 = Possible merger play
- Score 70-100 = Strong merger candidate signals

**Example Output:**
```
PHUN: Score 72.5 | $0.85 (+25.7%)
  🔄 Reverse Merger Candidate (Score: 72)
  🔄 Merger Signals: Micro-cap $45.2M, Massive volume 8.2x, 
      Merger-related news detected, High speculation sentiment
```

**Why This Matters:**
- Reverse mergers can lead to explosive price movements (50-500%+)
- Early detection before merger announcement = profit opportunity
- High speculation activity often precedes major corporate events
- Buzzing stocks + merger signals = potential multi-bagger plays

**Integration with Buzzing Scanner:**
- Merger candidate detection is automatically run on buzzing stocks
- If buzzing stock is also merger candidate, buzz score receives +10 bonus
- Merger candidate signals appear in buzzing reasons list
- Creates "double signal" for highest-probability speculative plays

**Usage:**
```python
# Filter for merger candidates
opportunities = scanner.scan_opportunities(
    scan_type=ScanType.PENNY_STOCKS,
    filters=ScanFilters(max_price=2.0, min_volume_ratio=3.0)
)

# Check for merger candidates
merger_candidates = [o for o in opportunities if o.is_merger_candidate]

for candidate in merger_candidates:
    print(f"{candidate.ticker}: Merger Score {candidate.merger_score:.0f}")
    print(f"  Signals: {', '.join(candidate.merger_signals)}")
```

**Risk Disclaimer:**
- Merger candidates are **highly speculative** plays
- Many shell companies never complete mergers
- High volatility and dilution risk
- Due diligence essential - verify news sources
- Position size should be minimal (1-2% of portfolio max)
- Set tight stop losses (10-15%)

### **Best Practices:**

1. **Start Broad**: Use "All Opportunities" to see overall market
2. **Refine**: Apply quick filters to narrow focus
3. **Cross-Reference**: Compare with your existing analysis
4. **Monitor Buzzing**: Check daily for emerging plays
5. **Backtest**: Validate setups with historical data
6. **Combine with Alerts**: Add top finds to My Tickers for notifications
7. **Export**: Save results for review and comparison

### **Integration with Other Features:**

- **Works with My Tickers**: Add discoveries to watchlist
- **Feeds into Analysis**: Click "Full Analysis" for deep dive
- **Complements ML Scanner**: Use together for maximum confidence
- **Alert Ready**: Set up alerts on discovered opportunities
- **CSV Export**: Export for spreadsheet analysis or backtesting

---

## Design notes & limitations

- The app relies on `yfinance` for market data and news. yfinance may occasionally return incomplete or missing fields (e.g., option chains or news). The code contains fallbacks when data is missing.
- IV Rank/Percentile is estimated/simulated in `TechnicalAnalyzer.calculate_iv_metrics` and should be treated as an approximation. For production IV metrics use a dedicated options data provider with historical IV surfaces.
- LLM integration is optional and can be configured to use different providers. The analyzer expects the LLM to return valid JSON embedded in the response; if parsing fails a helpful error analysis is returned instead.
- Tradier client uses simplified option symbol formatting — you may need to adapt `convert_signal_to_order` to match your broker's exact option symbol conventions.
- The app uses Streamlit's newer features (data_editor, status, toggle). Compatibility shims exist in `app.py` to gracefully degrade when running older Streamlit versions but features may be limited.
- **Qlib integration** is optional and requires separate installation (`pip install pyqlib`). If not installed, the app gracefully falls back to standard analysis methods.

## Security & safety

- Never commit API keys or secrets to version control.
- Keep the app in Paper Trading Mode while configuring and testing.

## Contributing

Open issues or PRs are welcome. If you make changes to pricing or greeks code, please add or update unit tests in `tests/`.

## License

MIT — see LICENSE (if present) for details.

## Contact

If you want help running the app or extending integrations (e.g., add a new LLM provider or broker adapter), open an issue or reach out via the repository.
