"""
Configuration for Background Auto-Trader - IBKR VERSION
Customize your trading bot settings here
Last updated: 2025-11-05 06:14:43

This is the IBKR-specific version of config_paper_trading.py
Automatically loaded when DEFAULT_BROKER=IBKR is set
"""

# ==============================================================================
# BROKER CONFIGURATION
# ==============================================================================

# Broker: "TRADIER" or "IBKR"
BROKER_TYPE = "IBKR"  # Interactive Brokers (this file is for IBKR)

# Market data source: "IBKR", "YFINANCE", or "HYBRID" (for IBKR broker only)
MARKET_DATA_SOURCE = "HYBRID"  # Recommended: Try IBKR first, fallback to yfinance if delayed

# ==============================================================================
# TRADING CONFIGURATION
# ==============================================================================

# Trading Mode: "SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"
TRADING_MODE = "SCALPING"

# Scan Interval (minutes)
SCAN_INTERVAL_MINUTES = 5

# Minimum Confidence % (only execute signals above this)
MIN_CONFIDENCE = 65

# ==============================================================================
# CAPITAL MANAGEMENT
# ==============================================================================

# Total capital allocated to auto-trading
TOTAL_CAPITAL = 1000.0  # $1,000

# Reserve cash percentage (kept aside, not used for trading)
RESERVE_CASH_PCT = 5.0  # 5.0% of $1,000 = $50 reserved

# Maximum capital utilization (% of usable capital that can be deployed)
MAX_CAPITAL_UTILIZATION_PCT = 95.0  # Max 95.0% of usable capital in positions

# ==============================================================================
# RISK MANAGEMENT
# ==============================================================================

MAX_DAILY_ORDERS = 15
MAX_POSITION_SIZE_PCT = 15.0  # Max % per position
RISK_PER_TRADE_PCT = 0.02  # 2.0% risk per trade
MAX_DAILY_LOSS_PCT = 0.04  # 4.0% max daily loss

# Bracket Orders (Stop-Loss & Take-Profit)
USE_BRACKET_ORDERS = True
SCALPING_TAKE_PROFIT_PCT = 1.0
SCALPING_STOP_LOSS_PCT = 0.5

# PDT-Safe Cash Management
USE_SETTLED_FUNDS_ONLY = False
CASH_BUCKETS = 3
T_PLUS_SETTLEMENT_DAYS = 2

# ==============================================================================
# TICKER SELECTION
# ==============================================================================

# Use Smart Scanner (finds best tickers automatically)
USE_SMART_SCANNER = True

# Your Custom Watchlist (used only if USE_SMART_SCANNER = False)
WATCHLIST = ['AFRM', 'AMC', 'AMD', 'BB', 'CHPT', 'COP', 'CVU', 'EVGO', 'FCEL', 'GOGY', 'MAIA', 'MARA', 'NOK', 'OXY', 'PINS', 'PLTR', 'PLUG', 'RIOT', 'RIVN', 'RUN', 'SOFI', 'SRFM', 'TLRY', 'UROY', 'WFC']

# ==============================================================================
# AI-POWERED HYBRID MODE (1-2 KNOCKOUT COMBO) 🥊
# ==============================================================================

# Enable ML-Enhanced Scanner for triple validation (40% ML + 35% LLM + 25% Quant)
USE_ML_ENHANCED_SCANNER = True  # RECOMMENDED: Superior trade quality

# Enable AI Pre-Trade Validation (final risk check before execution)
USE_AI_VALIDATION = False  # DISABLED: Too conservative, blocking all trades

# Minimum ensemble score for ML-Enhanced Scanner (0-100)
MIN_ENSEMBLE_SCORE = 60  # Only trades passing all 3 systems with 70%+ score

# Minimum AI validation confidence (0-1.0)
MIN_AI_VALIDATION_CONFIDENCE = 0.6  # AI must be 70%+ confident to approve

# NOTE: When both are enabled, you get the 1-2 KNOCKOUT COMBO:
#   PUNCH 1: ML-Enhanced Scanner filters trades (triple validation)
#   PUNCH 2: AI Validator performs final risk check
#   Result: Only the highest quality, lowest risk trades execute!

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

# Short Selling (ONLY works in paper trading)
ALLOW_SHORT_SELLING = False

# Multi-Agent System
USE_AGENT_SYSTEM = False

