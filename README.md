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
| **Crypto Breakout** | 24/7 Scanner for crypto pairs | ✅ Active | `services/crypto_breakout_service.py` |
| **DEX Hunter** | 🆕 Production Solana token launch detection | ✅ PRODUCTION | `services/dex_launch_hunter.py` |

### Running Services

**Windows:**
```bash
START_SERVICES.bat              # Start all services
START_DEX_HUNTER.bat            # Start DEX Hunter only
START_CRYPTO_AI_TRADER.bat      # Start crypto trader
START_STOCK_MONITOR.bat         # Start stock monitoring
```

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

This launches an interactive terminal UI for:
- Starting/stopping services
- Viewing service status
- Checking logs in real-time
- Configuring service parameters

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

**Last Updated**: December 1, 2025  
**Status**: ✅ Production Ready  
**Phases Completed**: 1, 2, 3 ✅
