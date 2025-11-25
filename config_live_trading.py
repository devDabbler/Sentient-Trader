"""
Configuration for Background Auto-Trader
Customize your trading bot settings here
Last updated: 2025-11-03 15:22:58
"""

# ==============================================================================
# BROKER & TRADING CONFIGURATION
# ==============================================================================

# Broker Selection: "TRADIER" or "IBKR" (Interactive Brokers)
# Determines which broker API to use for order execution
BROKER_TYPE = ""  # Options: "TRADIER" or "IBKR"

# Market Data Source (for IBKR only)
# "IBKR" - Use IBKR's data (real-time if you have subscriptions)
# "YFINANCE" - Use yfinance for real-time data (free alternative)
# "HYBRID" - Try IBKR first, fallback to yfinance if needed
MARKET_DATA_SOURCE = "IBKR"  # Options: "IBKR", "YFINANCE", "HYBRID" (use IBKR for live trading)

# ==============================================================================
# TRADING CONFIGURATION
# ==============================================================================

# Trading Mode: "SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"
TRADING_MODE = "STOCKS"

# Scan Interval (minutes)
SCAN_INTERVAL_MINUTES = 5

# Minimum Confidence % (only execute signals above this)
MIN_CONFIDENCE = 75

# ==============================================================================
# CAPITAL MANAGEMENT
# ==============================================================================

# Total capital allocated to auto-trading
TOTAL_CAPITAL = 500.0  # $500

# Reserve cash percentage (kept aside, not used for trading)
RESERVE_CASH_PCT = 10.0  # 10.0% = $50 reserved

# Maximum capital utilization (% of usable capital that can be deployed)
MAX_CAPITAL_UTILIZATION_PCT = 85.0  # Max 85.0% of usable capital in positions

# ==============================================================================
# RISK MANAGEMENT
# ==============================================================================

MAX_DAILY_ORDERS = 3
MAX_POSITION_SIZE_PCT = 15.0  # Max % per position
RISK_PER_TRADE_PCT = 0.01  # 1.0% risk per trade
MAX_DAILY_LOSS_PCT = 0.02  # 2.0% max daily loss

# Bracket Orders (Stop-Loss & Take-Profit)
USE_BRACKET_ORDERS = True
SCALPING_TAKE_PROFIT_PCT = 1.5
SCALPING_STOP_LOSS_PCT = 0.75

# PDT-Safe Cash Management
USE_SETTLED_FUNDS_ONLY = True
CASH_BUCKETS = 3
T_PLUS_SETTLEMENT_DAYS = 2

# ==============================================================================
# TICKER SELECTION
# ==============================================================================

# Use Smart Scanner (finds best tickers automatically)
USE_SMART_SCANNER = True

# Your Custom Watchlist (used only if USE_SMART_SCANNER = False)
WATCHLIST = ['AFRM', 'AMC', 'AMD', 'BB', 'CHPT', 'COP', 'CVU', 'EVGO', 'FCEL', 'GOGY', 'MAIA', 'MARA', 'NOK', 'OXY', 'PINS', 'PLTR', 'PLUG', 'RIOT', 'RIVN', 'RUN', 'SOFI', 'SRFM', 'TLRY', 'UROY', 'WFC']

# Long-Term Holdings (CRITICAL: These will NEVER be sold by the auto-trader)
# Add tickers here that you want to hold long-term
# These positions will be completely ignored by:
# - Position Exit Monitor (no auto-sells)
# - Auto-trader trading logic (won't be traded)
# - All automated risk management systems
LONG_TERM_HOLDINGS = ['BXP']  # Your long-term holds - DO NOT SELL

# ==============================================================================
# AI-POWERED HYBRID MODE (1-2 KNOCKOUT COMBO) 🥊
# ==============================================================================

# Enable ML-Enhanced Scanner for triple validation (40% ML + 35% LLM + 25% Quant)
USE_ML_ENHANCED_SCANNER = True  # RECOMMENDED: Superior trade quality

# Enable AI Pre-Trade Validation (final risk check before execution)
USE_AI_VALIDATION = True  # RECOMMENDED: Blocks high-risk trades

# Minimum ensemble score for ML-Enhanced Scanner (0-100)
MIN_ENSEMBLE_SCORE = 75  # Only trades passing all 3 systems with 70%+ score

# Minimum AI validation confidence (0-1.0)
MIN_AI_VALIDATION_CONFIDENCE = 0.75  # AI must be 70%+ confident to approve

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
