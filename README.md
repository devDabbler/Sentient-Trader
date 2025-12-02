# 📈 Sentient Trader Platform

> **AI-powered trading platform for stocks, options, and cryptocurrencies featuring real-time analysis, automated strategies, intelligent risk management, and advanced Solana DEX launch detection.**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![DEX Hunter](https://img.shields.io/badge/DEX%20Hunter-Phase%203-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌟 Overview

**Sentient Trader** is a comprehensive automated trading system that combines **quantitative analysis**, **LLM-based reasoning** (OpenRouter/Groq), **social sentiment** (X/Twitter), and **advanced on-chain verification** for cryptocurrency. 

The platform supports:
- **Stocks & Options**: Paper and live trading via IBKR (Interactive Brokers) and Tradier
- **Cryptocurrency**: 24/7 trading with real-time DEX analysis
- **Solana Tokens**: Production-grade DEX Hunter with on-chain verification

### 🎯 Key Features

#### Trading & Analysis
* **🤖 Triple-Validation System:** Combines ML factors, LLM reasoning, and technical indicators for high-conviction trades
* **📉 Multi-Asset Support:** Trade Stocks, Options (Strategies: Wheel, Spreads), and Crypto (Breakouts, DEX launches)
* **🔬 Entropy Analysis:** Proprietary market noise filtering to avoid choppy conditions
* **🐦 Social Sentiment:** Real-time buzzing stock detection via Crawl4AI (X, Reddit, StockTwits) without API costs
* **🛡️ Risk Management:** Auto-bracket orders, daily loss limits, and PDT-safe modes for small accounts
* **🔔 Smart Alerts:** Discord notifications for earnings, SEC filings, and trade signals

#### DEX Hunter (December 2025) ✅ PRODUCTION READY
* **🔍 On-Chain Verification:** Solana RPC-based token inspection (mint authority, freeze authority, LP status)
* **📊 Holder Distribution Analysis:** Concentration metrics and whale risk detection
* **🏦 LP Status Tracking:** Detects rug pull risks (LP in EOA wallets vs. burned/locked)
* **🎭 Metadata Inspection:** Detects impersonation risk via metadata immutability
* **💰 Cross-Source Price Validation:** Compares DexScreener vs Birdeye for data consistency
* **🚨 Hard Red Flag Enforcement:** Auto-rejects honeypots and unsafe tokens
* **⚡ RPC Load Balancing:** 3-endpoint failover with automatic rate limit handling
* **📈 Multi-Factor Scoring:** Pump potential, velocity, safety, liquidity, and social buzz
* **🐦 X/Twitter Integration:** Real-time social sentiment for trending tokens
* **📢 Multi-Source Discovery:** DexScreener API + Pump.fun integration
* **🎣 Webhook Execution Ready:** High-level placeholders for future bundler integration (Jito, Solayer)

#### Crypto Breakout Service (NEW - December 2025)
* **📊 Multi-Indicator Detection:** Volume spike, EMA crossover, MACD, RSI, Bollinger Bands
* **🪙 Jupiter DEX Cross-Validation:** Real-time price confirmation across Solana DEXs
* **⚡ Arbitrage Detection:** Identifies price spreads between Jupiter and Kraken
* **💧 Liquidity Depth Analysis:** Validates execution viability at multiple price levels
* **🎯 Confidence Scoring:** AI-enhanced technical analysis (when enabled)

#### Stock Intelligence Monitor (ENHANCED - December 2025) ✅ PRODUCTION READY
* **🎯 Multi-Pronged Analysis:** 4-stream detection (Technical + Events + ML + LLM)
  - Technical indicators (RSI, MACD, Bollinger Bands, Volume, Momentum)
  - Event/catalyst detection (Earnings, FDA, SEC filings, News sentiment)
  - ML confidence scoring (Performance, volatility, alignment)
  - LLM meta-analysis (AI reasoning on composite signals)
* **🔍 Stock Discovery Universe:** Auto-discover opportunities beyond watchlist
  - **Mega Caps** - Options-friendly large caps (AAPL, MSFT, etc.)
  - **High Beta Tech** - Volatile tech stocks (PLTR, SOFI, etc.)
  - **Momentum/Meme** - High momentum and meme stocks
  - **EV/Clean Energy** - Electric vehicle and clean energy stocks
  - **Crypto-Related** - Stocks tied to crypto (MARA, RIOT, COIN)
  - **AI Stocks** - Artificial intelligence related stocks
  - **Biotech** - Biotechnology and pharma stocks
  - **Financial** - Banks and financial services
  - **Energy** - Oil and gas stocks
  - **High IV Options** - High implied volatility for options trading
  - **Penny Stocks** - Low-priced stocks under $5
  - All 11 categories independently toggleable via Control Panel
  - **Works after hours** - Uses historical momentum & closing strength, not just intraday
* **⚙️ Service Control Panel Integration:** Full discovery configuration UI
  - **3 Scan Modes:** Watchlist Only | Discovery Only | Both (Watchlist + Discovery)
  - Enable/disable individual discovery categories
  - Adjust universe size per category (10-100 stocks)
  - View discovery statistics and metrics
  - Real-time scan mode indicator
* **📊 Production-Grade Resilience:** Health tracking, circuit breakers, auto-recovery
  - Comprehensive stats tracking (scans, alerts, errors, uptime)
  - Circuit breaker protection (prevents cascading failures)
  - Automatic retry logic with exponential backoff
  - Graceful shutdown with detailed statistics
  - Alert cooldown to prevent notification spam
* **🚀 Smart Caching:** 30-minute TTL per ticker
  - Efficient performance with fresh data
  - Automatic cleanup of old records
  - Watchlist auto-sync with Control Panel

#### Macro Market Filter (NEW - December 2025) ✅ PRODUCTION READY
* **🌐 Multi-Factor Macro Analysis:** Comprehensive market health assessment
  - **SPY/QQQ/IWM Trend Filter:** Major index direction (above/below 20/50/200 SMAs)
  - **VIX Fear Gauge:** Volatility regime detection (LOW/NORMAL/ELEVATED/HIGH/EXTREME)
  - **10Y Treasury Yields:** Interest rate environment (rising/stable/falling)
  - **Dollar Strength (DXY):** Currency impact on multinationals
  - **Sector Rotation:** Defensive vs Growth allocation tracking
  - **Market Breadth:** RSP vs SPY comparison as breadth proxy
  - **Economic Calendar:** Fed events, CPI, NFP proximity detection
* **⏱️ Micro/Intraday Factors:**
  - First hour momentum detection
  - Lunch hour avoidance
  - OpEx week awareness
  - Monday/Friday effects
* **📊 Trading Guidance:**
  - **Score Adjustment:** -30 to +30 points based on macro conditions
  - **Position Size Multiplier:** 25% to 125% based on market regime
  - **Trade Blocking:** Auto-block during FOMC, extreme VIX, crisis conditions
* **🎯 Regime Classification:**
  - **RISK_ON:** Favorable conditions, full position sizes
  - **NEUTRAL:** Mixed signals, standard approach
  - **RISK_OFF:** Caution, reduced exposure
  - **CRISIS:** High volatility, avoid new positions
* **⚙️ Configuration Options:**
  - VIX thresholds customizable (warning/high/extreme)
  - Event blocking toggleable (FOMC/CPI/NFP)
  - Position size multipliers per regime
  - 15-minute cache TTL to reduce API load

---

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone <repo>
cd sentient-trader
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1

# Install dependencies (CPU-only torch recommended for non-GPU servers)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Setup for X/Twitter scraping
pip install crawl4ai && crawl4ai-setup

# Run the UI
streamlit run app.py
```

### Configuration

Create a `.env` file in the root directory:

```bash
# AI & Data
OPENROUTER_API_KEY=sk-or-v1-...
GROQ_API_KEY=gsk_...
FINNHUB_API_KEY=...

# Brokers
# IBKR (Interactive Brokers)
IBKR_PAPER_PORT=7497
IBKR_PAPER_CLIENT_ID=1
IBKR_LIVE_PORT=7496
IBKR_LIVE_CLIENT_ID=2

# Tradier (optional)
TRADIER_PAPER_ACCOUNT_ID=...
TRADIER_PAPER_ACCESS_TOKEN=...

# Crypto
KRAKEN_API_KEY=...
KRAKEN_API_SECRET=...

# Solana RPC Endpoints (DEX Hunter)
SOLANA_RPC_URL=https://solana-mainnet.g.alchemy.com/v2/key1
SOLANA_RPC_URL_2=https://solana-mainnet.g.alchemy.com/v2/key2
SOLANA_RPC_URL_3=https://solana-mainnet.g.alchemy.com/v2/key3

# Alerts
DISCORD_WEBHOOK_URL=...
```

---

## ⚙️ Strategies & Automation

The platform runs multiple background services for continuous analysis and trading.

| Strategy | Description | Status | Config/Service |
|:---------|:-----------|:-------|:--|
| **Warrior Scalping** | Momentum "Gap & Go" (9:30-10:00 AM) | ✅ Active | `config_warrior_scalping.py` |
| **EMA Power Zone** | Swing trading based on 8/21 EMA & DeMarker | ✅ Active | `config_swing_trader.py` |
| **Options Premium** | Wheel strategy and credit spreads | ✅ Active | `config_options_premium.py` |
| **Stock Intelligence** | 🆕 Multi-pronged opportunity detection + discovery | ✅ PRODUCTION | `services/stock_informational_monitor.py` |
| **AI Stock Trader** | 🆕 Position monitoring with broker sync (Tradier/IBKR) | ✅ PRODUCTION | `services/ai_stock_position_manager.py` |
| **Crypto Breakout** | 24/7 Scanner for crypto pairs | ✅ Active | `services/crypto_breakout_service.py` |
| **DEX Hunter** | 🆕 Production Solana token launch detection | ✅ PRODUCTION | `services/dex_launch_hunter.py` |

### Running Services

**Windows:**
```powershell
START_SERVICES.bat              # Start all services
START_STOCK_MONITOR.bat         # Start enhanced stock intelligence monitor (with discovery)
START_STOCK_AI_TRADER.bat       # Start AI stock position manager (monitors your trades)
START_DEX_HUNTER.bat            # Start DEX Hunter only
START_CRYPTO_AI_TRADER.bat      # Start crypto trader
service_control_panel.py        # Streamlit UI for configuring all services
```

---

## 📊 Stock Trading Workflow (Discord + Broker Execution)

The platform now supports a complete **stock trading workflow** from detection → analysis → approval → execution via Discord:

### Workflow Overview

```
Stock Monitor (Detection) 
    ↓ High-confidence alert (score ≥70)
Discord Notification (with buttons/commands)
    ↓ Select analysis type (1/2/3)
AI Analysis (Standard/Multi/Ultimate)
    ↓ Review results, approve trade
Trade Execution (Paper or Live via Tradier/IBKR)
```

### Discord Commands (Reply to Alert)

| Command | Description |
|:--------|:------------|
| `1` or `S` | 🔬 Standard Analysis (single strategy) |
| `2` or `M` | 🎯 Multi-Config Analysis (Long/Short + timeframes) |
| `3` or `U` | 🚀 Ultimate Analysis (ALL combinations) |
| `W` or `WATCH` | Add to watchlist |
| `T` or `TRADE` | Execute BUY trade (after analysis) |
| `SHORT` | Execute SHORT/SELL trade |
| `P` or `PAPER` | Paper trade (test mode) |
| `SIZE` or `SIZING` | 📊 Show AI position sizing recommendation |
| `RISK` | 📊 Show current risk profile |
| `X` or `D` | Dismiss alert |
| `?` or `HELP` | Show all commands |

### Risk Profile & Position Sizing

AI-powered position sizing that automatically calculates optimal trade sizes based on your risk tolerance.

**Features:**
- **Risk Presets**: Conservative (5% max), Moderate (10% max), Aggressive (20% max)
- **AI Sizing**: Adjusts position size based on signal confidence
- **Risk-Based Sizing**: Calculates shares based on stop-loss distance and max risk per trade
- **Portfolio Awareness**: Considers current positions and available capital

**Configure via:**
1. **Service Control Panel** → Risk Profile tab
2. **Discord**: Reply `RISK` to see profile, `SIZE` or `SIZING` for specific calculations

**Discord Position Sizing Example:**
```
📊 Position Sizing for NVDA

💰 Entry: $450.00
🛑 Stop Loss: $427.50 (5.0% risk)

📈 Recommended Position:
   Shares: 22
   Value: $9,900.00
   % of Portfolio: 9.9%

⚠️ Risk:
   Amount at Risk: $495.00
   % of Capital: 0.5%

🎯 Targets (R-multiples):
   1R: $472.50 (+$495.00)
   2R: $495.00 (+$990.00)
   3R: $517.50 (+$1,485.00)
```

### Setting Up Stock Trading

1. **Configure Broker** (`.env` file):
```bash
# For IBKR (Interactive Brokers)
BROKER_TYPE=IBKR
IBKR_PAPER_PORT=7497       # Paper trading port
IBKR_PAPER_CLIENT_ID=1
STOCK_PAPER_MODE=true      # Start with paper trading!

# For Tradier
BROKER_TYPE=TRADIER
TRADIER_API_KEY=your_key
TRADIER_ACCOUNT_ID=your_id
TRADIER_PAPER=true
```

2. **Enable Discord Bot**:
```bash
DISCORD_BOT_TOKEN=your_bot_token
DISCORD_CHANNEL_IDS=channel_id_for_approvals
```

3. **Start Services**:
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start stock monitor + approval bot
python -m windows_services.runners.run_stock_monitor
# Or use the batch file
START_STOCK_MONITOR.bat
```

### Key Components

| Component | File | Description |
|:----------|:-----|:------------|
| Stock Monitor | `services/stock_informational_monitor.py` | Detects opportunities |
| AI Stock Entry | `services/ai_stock_entry_assistant.py` | Analyzes entry timing |
| **Position Manager** | `services/ai_stock_position_manager.py` | **Monitors & manages positions with broker sync** |
| Discord Approval | `services/discord_trade_approval.py` | Handles approval workflow |
| Broker Adapter | `src/integrations/broker_adapter.py` | Unified Tradier/IBKR interface |
| **Runner Script** | `windows_services/runners/run_stock_ai_trader_simple.py` | **Service startup script** |

### Analysis Modes

| Mode | Description | Use Case |
|:-----|:------------|:---------|
| **Standard** | Single strategy, one timeframe | Quick scan, high-volume opportunities |
| **Multi** | Long + Short analysis, multiple timeframes | Swing trades, position sizing |
| **Ultimate** | ALL strategies + directions + timeframes | Deep research, confident entries |

---

## 🤖 AI Stock Position Manager (NEW - December 2025) ✅ PRODUCTION READY

The **AI Stock Position Manager** actively monitors your stock positions (paper and live) and applies intelligent risk management.

### Key Features

* **🔄 Broker Sync:** Automatically syncs with Tradier or IBKR on startup and periodically
* **📊 Position Monitoring:** Tracks all open positions with real-time price updates
* **🛡️ Stop Loss & Take Profit:** Automated position management with configurable thresholds
* **📈 Trailing Stops:** Dynamically adjusts stops as positions move in your favor
* **🎯 Breakeven Protection:** Moves stop to entry price after configurable profit threshold
* **📱 Discord Integration:** Sends alerts for trade recommendations and requires approval before execution

### How It Works

```
Startup
    ↓ Connects to broker (Tradier/IBKR)
    ↓ Syncs all open positions
Monitoring Loop (every 60s)
    ↓ Checks each position for:
    │   - Stop loss triggers
    │   - Take profit triggers
    │   - Breakeven conditions
    │   - Trailing stop adjustments
    ↓ Every 10 cycles: Re-sync with broker
    ↓ Discord alerts for recommendations
```

### Starting the Service

**Windows:**
```powershell
START_STOCK_AI_TRADER.bat
# Or directly:
python windows_services\runners\run_stock_ai_trader_simple.py
```

**Linux (background):**
```bash
nohup python windows_services/runners/run_stock_ai_trader_simple.py > logs/stock_ai_trader_service.log 2>&1 &
```

### Configuration

Set in `.env`:
```bash
# Broker Selection
BROKER_TYPE=TRADIER           # or IBKR

# Trading Mode
STOCK_PAPER_MODE=true         # true = paper, false = LIVE TRADING

# For Tradier
TRADIER_PAPER_ACCOUNT_ID=...
TRADIER_PAPER_ACCESS_TOKEN=...
# For live:
TRADIER_PROD_ACCOUNT_ID=...
TRADIER_PROD_ACCESS_TOKEN=...

# For IBKR
IBKR_PAPER_PORT=7497
IBKR_PAPER_CLIENT_ID=1
IBKR_LIVE_PORT=7496
IBKR_LIVE_CLIENT_ID=2
```

### Sync Behavior

| Event | Action |
|:------|:-------|
| **Startup** | Full sync with broker - imports all positions |
| **Every 10 cycles** | Re-sync to catch manual trades or external changes |
| **Position not in broker** | Removed from AI tracking |
| **New broker position** | Added with default 2% stop / 4% target |

### Service Control Panel

The AI Stock Trader appears in the Service Control Panel under "Stocks" category:
- View real-time status
- Start/stop the service
- Adjust check interval (30s - 5min)
- View logs

**Linux (Systemd):**
```bash
sudo systemctl start sentient-dex-launch      # Start DEX Hunter service
sudo systemctl status sentient-dex-launch     # Check status
tail -f logs/dex_launch_service.log           # View live logs
```

---

## 🆕 DEX Hunter System (Phase 3 Complete)

### What is DEX Hunter?

DEX Hunter is an **advanced token launch detection system** for Solana that combines:
- **On-chain verification** via Solana RPC
- **Risk scoring** based on contract safety metrics
- **Social sentiment** analysis from X/Twitter
- **Multi-source data** aggregation (DexScreener + Pump.fun)

### Phase Completion Status

✅ **Phase 1: Core Solana RPC Integration** (COMPLETE)
- Mint authority & freeze authority inspection
- LP token ownership verification
- Hard red flag enforcement (auto-reject dangerous tokens)

✅ **Phase 2: Holder Distribution Analysis** (COMPLETE)
- On-chain holder concentration metrics
- Whale risk detection (top 1, 5, 10, 20 percentages)
- 90% reduction in RPC calls via optimized strategy

✅ **Phase 3: Enhanced Validation** (COMPLETE)
- On-chain metadata inspection (impersonation risk detection)
- Cross-source price validation (DexScreener vs Birdeye)
- RPC load balancing (3 endpoints with automatic failover)
- Enhanced rate limiting with exponential backoff

### Key Services

```
services/
├── dex_launch_hunter.py          # Main orchestrator
├── solana_mint_inspector.py      # Mint/freeze authority checks
├── solana_lp_analyzer.py         # LP ownership verification
├── solana_holder_analyzer.py     # Holder concentration analysis
├── solana_metadata_inspector.py  # Metadata immutability checks
├── price_validator.py            # Cross-source price validation
├── token_safety_analyzer.py      # Safety scoring hub
├── x_sentiment_service.py        # X/Twitter sentiment
└── launch_announcement_monitor.py # Pump.fun integration
```

### Scoring System

**Composite Score (0-100):**
- **Pump Potential** (0-100): Market metrics and price momentum
- **Velocity Score** (0-100): Price change over timeframes
- **Safety Score** (0-100): Contract safety (mint/freeze/LP/holder checks)
- **Liquidity Score** (0-100): Trading depth and volume
- **Social Buzz** (0-100): X/Twitter sentiment

**Alert Thresholds:**
- ≥ 70: 🚨 CRITICAL priority alert
- ≥ 60: 🔔 HIGH priority alert
- ≥ 30: Gets X/Twitter sentiment enrichment
- < 30: No alert (low score)

### Hard Red Flags (Auto-Reject)

Tokens are automatically blacklisted if:
- ❌ Mint authority retained (can mint infinite tokens)
- ❌ Freeze authority retained (honeypot - users can't sell)
- ❌ LP tokens in EOA wallet (rug pull risk)
- ❌ Already detected as honeypot

### Configuration Options (NEW - December 2025)

**Lenient Mode** - Most new Solana tokens keep mint/freeze authority initially:
```python
from models.dex_models import HunterConfig

config = HunterConfig(
    lenient_solana_mode=True,    # Allow tokens with mint/freeze (most new launches have these)
    discovery_mode="aggressive",  # "conservative", "balanced", or "aggressive"
    min_liquidity_usd=500.0,     # Lower for early launches
    min_composite_score=20.0,    # Show more tokens for manual evaluation
)
```

**Discovery Modes:**
- `aggressive` - Lowest filters, finds most tokens (higher risk)
- `balanced` - Moderate filters (default)
- `conservative` - Strict filters, fewer but safer tokens

**Why Lenient Mode?** Many legitimate meme coins keep mint/freeze authority for the first few hours/days. Strict mode (default before this update) would filter out ~90% of new launches. With lenient mode ON, these tokens are flagged with warnings but not blacklisted.

**Environment Variables (add to .env):**
```bash
# DEX Hunter Settings
DEX_LENIENT_MODE=true       # true = allow risky tokens, false = strict filtering
DEX_DISCOVERY_MODE=aggressive  # aggressive, balanced, or conservative
DEX_MIN_LIQUIDITY=500       # Minimum liquidity in USD
```

### RPC Optimization

- **Two-tier strategy**: `getTokenLargestAccounts` (lightweight) → fallback to `getProgramAccounts`
- **Load balancing**: 3 RPC endpoints with automatic routing
- **Failover**: Automatic endpoint switching on rate limits
- **Batch operations**: `getMultipleAccounts` for efficiency
- **Result**: 90% reduction in RPC calls vs. naive implementation

### Testing

```bash
# Run automated test suite
tests/RUN_DEX_HUNTER_TESTS.bat

# Or run manually
pytest tests/test_dex_hunter_complete.py -v

# Interactive testing with real tokens
python tests/test_dex_hunter_manual.py
```

**Latest Test Results (December 1, 2025):**
```
✅ Mint inspection: PASSING
✅ LP analysis: PASSING
✅ Holder distribution: PASSING
✅ Metadata inspection: PASSING
✅ Price validation: PASSING (88.1/100 consistency)
✅ Rate limiting: HANDLED GRACEFULLY
✅ Real token detection: VERIFIED (honeypot detected)
✅ Systemd service: RUNNING
```

---

## 📊 Advanced Systems

### 1. Entropy Market Filter

Uses Shannon and Approximate Entropy to measure market chaos.

* **< 30 (Structured):** Ideal for trading
* **> 70 (Noisy):** Trading automatically blocked to prevent whipsaws

### 2. Advanced Opportunity Scanner

Finds plays before they rocket using customizable filters:

* **Buzzing Stocks:** Combines volume spikes with social sentiment
* **Reverse Merger:** Detects shell companies and unusual dark pool activity
* **Penny Stock Risk:** Auto-detects dilution history and reverse splits
* **DEX Launches:** Identifies new token launches on Solana with on-chain verification

### 3. ML-Enhanced Analysis

For maximum confidence, run the triple-validation scanner:

```python
from services.ml_enhanced_scanner import MLEnhancedScanner
scanner = MLEnhancedScanner()
# Returns trades only if ML, LLM, and Technicals agree
trades = scanner.scan_top_options_with_ml(min_ensemble_score=70.0)
```

### 4. On-Chain Verification Pipeline

For DEX Hunter token analysis:

```python
from services.dex_launch_hunter import DEXLaunchHunter
from services.token_safety_analyzer import TokenSafetyAnalyzer

hunter = DEXLaunchHunter()
analyzer = TokenSafetyAnalyzer()

# Scan for new launches
tokens = await hunter.scan_launches()

# Analyze with on-chain verification
for token in tokens:
    safety = await analyzer.analyze_token(
        contract_address=token.contract_address,
        chain=token.chain,
        pool_address=token.pool_address
    )
    
    if safety.risk_level == RiskLevel.EXTREME:
        print(f"🚨 Rejected: {token.symbol} - {safety.risk_reasons}")
    else:
        print(f"✅ Score: {safety.safety_score}/100")
```

### 5. Jupiter DEX Price Validation (NEW - December 2025)

For Crypto Breakout Service - cross-validates prices with Solana DEXs:

```python
from clients.jupiter_client import get_jupiter_client

jupiter = get_jupiter_client()

# Get real-time DEX price
jupiter_price = await jupiter.get_price(token_mint="...")

# Check price spreads vs. Kraken
spread_info = await jupiter.check_price_spread(
    token_mint="...",
    reference_price=kraken_price,
    reference_source="Kraken"
)

# Identify arbitrage opportunities
if spread_info['arbitrage_opportunity']:
    print(f"⚡ Arbitrage: Jupiter ${spread_info['jupiter_price']:.6f} "
          f"vs Kraken ${spread_info['reference_price']:.6f}")
```

**Features:**
- Real-time quote fetching from Jupiter Aggregator v6
- Automatic caching (60-second TTL)
- Price spread detection and arbitrage opportunity flagging
- Liquidity depth analysis at multiple price levels

**Environment Variables:**
```bash
# Jupiter configuration (optional, uses defaults if not set)
JUPITER_CACHE_TTL_SECONDS=60
```

### 6. DEX Execution Webhook (NEW - December 2025)

High-level webhook architecture for future bundler integration (Jito, Solayer):

```python
from services.dex_execution_webhook import get_dex_execution_webhook

webhook = get_dex_execution_webhook()

# Queue snipe execution (routes to external bundler service when configured)
success, message, request = await webhook.execute_snipe(
    token_mint="65aP2yHMZ6RxZpXn3iHhfBRnzCpwbZeVDTXAoi1gpump",
    amount_usd=25.0,
    slippage_bps=50,  # 0.5%
    metadata={'source': 'DEX_HUNTER', 'score': 82.5}
)

# Check execution status
status = webhook.get_status(request.request_id)

# Configure external services at runtime
webhook.configure_webhook('snipe', 'https://bundler.example.com/execute')
```

**Ready for Integration With:**
- **Jito Bundles** (Recommended - 45% Solana network coverage)
- **Solayer** (Privacy-focused validator MEV protection)
- **Custom Bundler Services** (Any HTTP webhook endpoint)

**Current Status:** ✅ Placeholders ready, awaiting external service configuration

**Setup Instructions:**
1. Set `DEX_EXECUTION_SNIPE_WEBHOOK` env variable to bundler URL
2. Set `DEX_EXECUTION_ARBITRAGE_WEBHOOK` for arbitrage execution
3. DEX Hunter automatically routes CRITICAL/HIGH priority launches to webhook
4. See `docs/BUNDLER_ECOSYSTEM_MONITORING.md` for integration timeline

---

## 🔌 Broker Integration

### IBKR (Interactive Brokers) ✅ ACTIVE

**Paper Trading:**
- Port: `7497` (TWS application)
- Client ID: `1`
- Config: `config_paper_trading_ibkr.py`

**Live Trading:**
- Port: `7496` (TWS application)
- Client ID: `2`
- Config: `config_live_trading.py`

**Note:** Use TWS (Trader Workstation), not Gateway. Read-only API disabled.

### Tradier (Optional)

- Stocks and Options
- Paper and Live trading
- Config: `config_paper_trading_tradier.py`

### Kraken (Crypto)

- 24/7 spot trading
- API key & secret required
- Used by crypto breakout and DEX hunter services

---

## 📁 Project Structure

```
sentient-trader/
├── app.py                          # Main Streamlit UI
├── config_*.py                     # Strategy configurations
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (not in repo)
│
├── services/ (126+ services)
│   ├── dex_launch_hunter.py        # DEX Hunter main orchestrator
│   ├── solana_*.py                 # Solana on-chain verifications
│   ├── token_safety_analyzer.py    # Safety scoring hub
│   ├── ml_enhanced_scanner.py      # ML-based opportunity detection
│   ├── x_sentiment_service.py      # Twitter sentiment
│   └── [120+ trading services]
│
├── clients/
│   ├── dexscreener_client.py       # DexScreener API
│   ├── kraken_client.py            # Kraken trading
│   ├── supabase_client.py          # Supabase integration
│   └── validators.py               # Address/contract validation
│
├── models/
│   ├── dex_models.py               # TokenLaunch, ContractSafety, HolderDistribution
│   └── [trading models]
│
├── utils/
│   └── solana_rpc_load_balancer.py # RPC endpoint management
│
├── ui/
│   └── [UI components]
│
├── tests/
│   ├── test_dex_hunter_complete.py # Automated pytest suite
│   ├── test_dex_hunter_manual.py   # Interactive testing
│   └── RUN_DEX_HUNTER_TESTS.bat    # Test runner
│
├── windows_services/               # Service runners
├── docs/                           # Documentation
│   └── DEX_HUNTER_REVIEW.md        # Comprehensive DEX Hunter review
│
└── logs/                           # Service logs
```

---

## 🛠️ Service Management

### Control Panel

```bash
python service_control_panel.py
```

This launches an interactive Streamlit UI for:
- Starting/stopping services
- Viewing service status
- Checking logs in real-time
- Configuring service parameters

#### Watchlist Manager (December 2025)

The Control Panel includes a unified **Watchlist Manager** with:

* **📈 Stocks / 🪙 Crypto Separation:** Tickers are organized into dedicated tabs for stocks and crypto, making it easy to manage each asset class independently
* **☁️ Supabase Integration:** Full sync with Supabase database for persistent storage
  - Automatic fetching of all tickers (up to 1000) from your saved watchlist
  - Filter by asset type (stock, penny_stock, crypto)
  - Real-time add/remove with cloud sync
* **⚙️ Service-Specific Watchlists:** Configure which tickers each service monitors
  - Stock Monitor, Crypto Breakout, AI Trader, etc.
  - Quick actions: Select All, Clear, Top 5, Sync from Supabase
  - Custom ticker input for adding new symbols
* **🚫 AI Exclusions:** Manage pairs permanently excluded from AI trading

### Service Status Commands

```bash
# View live logs
VIEW_ALL_LOGS.bat

# Check DEX Hunter logs
tail -f logs/dex_launch_service.log

# Check for errors
grep "ERROR" logs/*.log

# Monitor service memory/CPU
tasklist | findstr python  # Windows
ps aux | grep python      # Linux
```

---

## 📊 Performance & Scale

- **DEX Hunter**: Analyzes 50+ token launches per 5-minute scan cycle
- **Holder Analysis**: 90% reduction in RPC calls via optimized strategy
- **Rate Limiting**: Handles 429 errors gracefully with exponential backoff
- **Data Sources**: DexScreener, Pump.fun, Birdeye, X/Twitter, Solana RPC
- **Supported Chains**: Ethereum, BSC, Solana, Base, and others (via DexScreener)

---

## 📝 Logging & Monitoring

### Log Tags

- `[DEX]` - DexScreener scanning and token analysis
- `[WHALE]` - Smart money/whale wallet tracking
- `[X]` - X/Twitter sentiment enrichment
- `[ALERT]` - Alert generation and notifications
- `[RPC]` - Solana RPC calls
- `[ERROR]` - Error conditions

### Example Log Output

```
🔄 Scan cycle #1 starting...
[DEX] Found 50 pairs from DexScreener
[DEX] Analyzing: PEPE (solana)...
[DEX] 🚨 PEPE: BLACKLISTED (Solana on-chain check failed - freeze authority)
[DEX] Analyzing: FETCH (ethereum)...
[DEX] ✓ FETCH: Score=35.5/100, Risk=MEDIUM
    └─ Pump:40 Velocity:30 Safety:45 Liq:25
    └─ Price=$0.00012 Liq=$12,000 Vol=$3,500
[DEX] Scan complete: Analyzed=20, Failed=5
🐦 Fetching X/Twitter sentiment for tokens with score >= 30...
✓ Active DEX scan completed!
📊 Stats: 5 discovered, 0 high-score alerts
💤 Sleeping 5 minutes until next scan...
```

---

## 🔐 Security Considerations

### Protected Data
- ✅ API keys stored in `.env` (never in code)
- ✅ Broker credentials in config files
- ✅ Sensitive data never logged
- ✅ Trade journal encrypted in SQLite database

### Input Validation
- ✅ Address format validation per blockchain
- ✅ Scam name pattern filtering
- ✅ Honeypot detection (freeze authority, mint authority)
- ✅ Contract safety verification

### Error Handling
- ✅ Graceful degradation on service failures
- ✅ Automatic retry with exponential backoff
- ✅ No data loss on API failures
- ✅ Service recovery on network issues

---

## ⚠️ Disclaimer

**Trading involves significant risk of loss.** This software is provided for **educational and research purposes only**. Always:

1. ✅ Test strategies in **Paper Trading Mode** (`IS_PAPER_TRADING=True`) before risking real capital
2. ✅ Understand the risks of each asset class (stocks, options, crypto, meme coins)
3. ✅ Start with small position sizes
4. ✅ Use stop losses
5. ✅ Never risk more than you can afford to lose
6. ✅ Consult a financial advisor if unsure

**The authors assume no responsibility for trading losses.**

---

## 📞 Support & Documentation

- **DEX Hunter Deep Dive**: See `docs/DEX_HUNTER_REVIEW.md`
- **Project Rules**: See `.cursor/rules.md` or `.windsurf/rules.md`
- **Configuration Examples**: See `config_*.py` files
- **Service Logs**: Check `logs/` directory

---

## 📄 License

MIT License - See LICENSE file for details

---

## ✨ Recent Updates (December 2025)

### Kraken Position Sync UI (December 2, 2025)
- ✅ **Kraken Sync Button:** Control Panel now has "🔄 Sync Kraken Positions" button for AI Crypto Trader
- ✅ **Discord Notifications:** Synced positions announced to Discord with entry price, P&L, stop/target
- ✅ **Auto-Import:** All Kraken positions automatically added to AI monitoring with 5% stop, 10% target
- ✅ **View Positions:** See all monitored crypto positions with real-time P&L in Control Panel
- ✅ **Sync Summary:** Discord notification shows total added, removed, kept positions
- ✅ **Mirrors Stock UI:** Same layout and functionality as Tradier/IBKR broker sync

### Crypto Breakout Trade Execution from Discord (December 2, 2025)
- ✅ **Trade Button Added:** Crypto breakout alerts now have a 🚀 Trade button for direct execution
- ✅ **One-Click Trading:** Click Trade on any crypto alert to execute via AI Crypto Position Manager
- ✅ **Auto Risk Calculation:** Position size calculated from risk profile, 2% stop, 4% target by default
- ✅ **Alert Data Passed:** Price, score, confidence passed to trade execution for optimal sizing
- ✅ **AI Position Monitoring:** Trades automatically monitored with trailing stops, breakeven moves
- ✅ **Mirrors Stock Implementation:** Same Discord-to-execution flow as stock AI trader

### Unified Trade Journal Integration (December 2, 2025)
- ✅ **Stock Trades Now Journaled:** AI Stock Position Manager now logs all trades to the UnifiedTradeJournal (matching crypto)
- ✅ **Entry & Exit Logging:** Both trade entries and exits are recorded with full P&L, R-multiple, and market conditions
- ✅ **Control Panel Integration:** New "Trade Journal" sections in Service Control Panel for both stock and crypto AI traders
- ✅ **Journal Stats:** View total trades, win rate, P&L, AI-managed trade performance from Control Panel
- ✅ **Recent Trades View:** Quick access to last 10 stock or crypto trades with status and P&L
- ✅ **Consistent Style Tracking:** Stock trades now respect the same trading style/strategy as crypto trades
- ✅ **Discord Alerts for Stock Positions:** Now sends Discord notifications for:
  - 📥 Position synced from broker (GME, SOFI, etc. will be announced when imported)
  - 🛑 Stop loss triggered
  - 🎯 Take profit hit
  - 🛡️ Breakeven move executed
  - ✅ Broker sync summary
- ✅ **Synced Position Journaling:** Positions imported from broker are automatically journaled for tracking
- ✅ **Position Status Logging:** Each check cycle logs P&L, current price, and stop/target for all positions

### AI Stock Position Manager (December 2, 2025)
- ✅ **Broker Sync:** New `sync_with_broker()` method syncs positions from Tradier/IBKR on startup and periodically
- ✅ **Position Monitoring:** Monitors all open stock positions (paper and live) with stop loss/take profit management
- ✅ **Trailing Stops:** Automatic trailing stop adjustments as positions move in your favor
- ✅ **Breakeven Protection:** Moves stop to entry price after configurable profit threshold
- ✅ **Runner Script:** New `run_stock_ai_trader_simple.py` for easy service startup
- ✅ **Batch File:** `START_STOCK_AI_TRADER.bat` for Windows quick-start
- ✅ **Service Integration:** Added to Service Control Panel and Service Orchestrator
- ✅ **Trade Journal Integration:** All trades (entry/exit) now logged to UnifiedTradeJournal for tracking and reference

### Discord & Control Panel Fixes (December 2, 2025)
- ✅ **Discord 'Analyze' Button Fixed**: Buttons now use unique IDs per message to prevent "interaction failed" errors
- ✅ **Discord Analysis Results**: Analysis results now automatically send to Discord (enabled by default)
- ✅ **Watchlist Sync**: Stocks/crypto added via Discord now sync to both Supabase AND service watchlists
- ✅ **Crypto Watchlist Seeding**: New "Seed from Config" button when watchlist is empty - populates with top cryptos
- ✅ **Auto-Refresh Enabled**: Analysis results auto-refresh now defaults to ON (15-second intervals)
- ✅ **Improved Results Display**: Cleaner tabs, no duplicates, better labels, compact layout with action counts

### DEX Hunter v3 (Production Ready)
- ✅ Phase 1, 2, 3 complete
- ✅ Systemd service integration
- ✅ Comprehensive test suite (all phases passing)
- ✅ Real token detection verified
- ✅ Running on Linux VPS as continuous service
- ✅ Verbose logging with context tags
- ✅ RPC load balancing with 3 endpoints
- ✅ 90% reduction in RPC calls

### Broker Integration
- ✅ IBKR paper trading fully configured (port 7497)
- ✅ IBKR live trading supported (port 7496)
- ✅ Unified broker adapter
- ✅ Test suite: `test_ibkr.bat`

### Documentation
- ✅ `.cursor/rules.md` - Cursor IDE project roles
- ✅ `.windsurf/rules.md` - Windsurf IDE project roles
- ✅ `docs/DEX_HUNTER_REVIEW.md` - Comprehensive DEX Hunter review

---

**Last Updated**: December 2, 2025  
**Status**: ✅ Production Ready  
**Phases Completed**: 1, 2, 3 ✅
