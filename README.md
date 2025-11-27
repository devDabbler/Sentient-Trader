README.md

````markdown
# 📈 Sentient Trader Platform

> **AI-powered trading platform for stocks, options, and cryptocurrencies featuring real-time analysis, automated strategies, and intelligent risk management.**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌟 Overview
Sentient Trader is a comprehensive automated trading system that combines **quantitative analysis**, **LLM-based reasoning** (OpenRouter/Groq), and **social sentiment** (X/Reddit) to identify high-probability setups. It supports paper and live trading via Tradier (Stocks/Options) and Kraken (Crypto).

### Key Features
* **🤖 Triple-Validation System:** Combines ML factors, LLM reasoning, and technical indicators for high-conviction trades.
* **📉 Multi-Asset Support:** Trade Stocks, Options (Strategies: Wheel, Spreads), and Crypto (Breakouts, DEX launches).
* **🔬 Entropy Analysis:** Proprietary market noise filtering to avoid choppy conditions.
* **🐦 Social Sentiment:** Real-time buzzing stock detection via Crawl4AI (X, Reddit, StockTwits) without API costs.
* **🛡️ Risk Management:** Auto-bracket orders, daily loss limits, and PDT-safe modes for small accounts.
* **🔔 Smart Alerts:** Discord notifications for earnings, SEC filings, and trade signals.

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
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt

# Setup for X/Twitter scraping
pip install crawl4ai && crawl4ai-setup

# Run the UI
streamlit run app.py
````

### Configuration

Create a `.env` file in the root directory:

```bash
# AI & Data
OPENROUTER_API_KEY=sk-or-v1-...
GROQ_API_KEY=gsk_...
FINNHUB_API_KEY=...

# Trading (Tradier & Kraken)
TRADIER_PAPER_ACCOUNT_ID=...
TRADIER_PAPER_ACCESS_TOKEN=...
KRAKEN_API_KEY=...
KRAKEN_API_SECRET=...

# Alerts
DISCORD_WEBHOOK_URL=...
```

---

## ⚙️ Strategies & Automation

The platform runs multiple background services for continuous analysis.

| Strategy                   | Description                                | Config File                                     |
| :------------------------- | :----------------------------------------- | :---------------------------------------------- |
| **Warrior Scalping** | Momentum "Gap & Go" (9:30-10:00 AM)        | `config_warrior_scalping.py`                  |
| **EMA Power Zone**   | Swing trading based on 8/21 EMA & DeMarker | `config_swing_trader.py`                      |
| **Options Premium**  | Wheel strategy and credit spreads          | `config_options_premium.py`                   |
| **Crypto Breakout**  | 24/7 Scanner for crypto pairs              | `windows_services/crypto_breakout_service.py` |
| **DEX Hunter**       | New token launch detection                 | `windows_services/dex_launch_service.py`      |

**To run background automation:**

```bash
python windows_services/runners/run_autotrader_background.py
```

---

## 📊 Advanced Systems

### 1\. Entropy Market Filter

Uses Shannon and Approximate Entropy to measure market chaos.

* **\< 30 (Structured):** Ideal for trading.
* **\> 70 (Noisy):** Trading is automatically blocked to prevent whipsaws.

### 2\. Advanced Opportunity Scanner

Finds plays before they rocket using customizable filters:

* **Buzzing Stocks:** Combines volume spikes with social sentiment.
* **Reverse Merger:** Detects shell companies and unusual dark pool activity.
* **Penny Stock Risk:** Auto-detects dilution history and reverse splits.

### 3\. ML-Enhanced Analysis

For maximum confidence, run the triple-validation scanner:

```python
from services.ml_enhanced_scanner import MLEnhancedScanner
scanner = MLEnhancedScanner()
# Returns trades only if ML, LLM, and Technicals agree
trades = scanner.scan_top_options_with_ml(min_ensemble_score=70.0)
```

---

## ⚠️ Disclaimer

**Trading involves significant risk.** This software is for educational purposes. Always test strategies in **Paper Trading Mode** (`IS_PAPER_TRADING=True`) before risking real capital.

## License

MIT

```


```
