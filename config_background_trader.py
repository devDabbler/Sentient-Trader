"""
Configuration for Background Auto-Trader
Customize your trading bot settings here
Last updated: 2025-11-01 (Auto-updated)

⚠️ IMPORTANT: For WARRIOR_SCALPING mode, use config_warrior_scalping.py instead!
This file is for general SCALPING/STOCKS/OPTIONS modes.

The background runner (run_autotrader_background.py) will automatically detect
and use config_warrior_scalping.py if TRADING_MODE="WARRIOR_SCALPING" is set there.
"""

# ==============================================================================
# TRADING CONFIGURATION
# ==============================================================================

# Trading Mode: "SCALPING", "STOCKS", "OPTIONS", "WARRIOR_SCALPING", "ALL"
# ⚠️ For WARRIOR_SCALPING, edit config_warrior_scalping.py instead!
TRADING_MODE = "SCALPING"

# Scan Interval (minutes)
SCAN_INTERVAL_MINUTES = 5

# Minimum Confidence % (only execute signals above this)
MIN_CONFIDENCE = 65

# Risk Management
MAX_DAILY_ORDERS = 3
MAX_POSITION_SIZE_PCT = 15.0
RISK_PER_TRADE_PCT = 0.01
MAX_DAILY_LOSS_PCT = 0.02

# Bracket Orders (Stop-Loss & Take-Profit)
USE_BRACKET_ORDERS = True
SCALPING_TAKE_PROFIT_PCT = 2.0
SCALPING_STOP_LOSS_PCT = 1.0

# PDT-Safe Cash Management
USE_SETTLED_FUNDS_ONLY = True
CASH_BUCKETS = 3
T_PLUS_SETTLEMENT_DAYS = 2
RESERVE_CASH_PCT = 0.05

# Capital Management
TOTAL_CAPITAL = 10000.0
MAX_CAPITAL_UTILIZATION_PCT = 80.0

# Trading Hours (Eastern Time)
TRADING_START_HOUR = 9
TRADING_START_MINUTE = 30
TRADING_END_HOUR = 15
TRADING_END_MINUTE = 30

# ==============================================================================
# TICKER SELECTION
# ==============================================================================

# Use Smart Scanner (finds best tickers automatically)
# ✅ RECOMMENDED: Set to True to automatically discover opportunities
# ❌ Set to False to use your custom watchlist below
USE_SMART_SCANNER = True  # Changed to True for better performance

# Your Custom Watchlist (used only if USE_SMART_SCANNER = False)
WATCHLIST = ['AFRM', 'AMC', 'AMD', 'BB', 'CHPT', 'COP', 'CVU', 'EVGO', 'FCEL', 'GOGY', 'MAIA', 'MARA', 'NOK', 'OXY', 'PINS', 'PLTR', 'PLUG', 'RIOT', 'RIVN', 'RUN', 'SOFI', 'SRFM', 'TLRY', 'UROY', 'WFC']

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

# Short Selling (ONLY works in paper trading)
ALLOW_SHORT_SELLING = False

# Multi-Agent System
USE_AGENT_SYSTEM = False

# Test Mode (for testing when market is closed)
TEST_MODE = False
