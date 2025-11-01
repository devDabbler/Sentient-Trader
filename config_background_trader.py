"""
Configuration for Background Auto-Trader
Customize your trading bot settings here
Last updated: 2025-10-31 12:20:41
"""

# ==============================================================================
# TRADING CONFIGURATION
# ==============================================================================

# Trading Mode: "SCALPING", "STOCKS", "OPTIONS", "ALL"
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

# ==============================================================================
# TICKER SELECTION
# ==============================================================================

# Use Smart Scanner (finds best tickers automatically)
USE_SMART_SCANNER = False

# Your Custom Watchlist (used only if USE_SMART_SCANNER = False)
WATCHLIST = ['AFRM', 'AMC', 'AMD', 'BB', 'CHPT', 'COP', 'CVU', 'EVGO', 'FCEL', 'GOGY', 'MAIA', 'MARA', 'NOK', 'OXY', 'PINS', 'PLTR', 'PLUG', 'RIOT', 'RIVN', 'RUN', 'SOFI', 'SRFM', 'TLRY', 'UROY', 'WFC']

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

# Short Selling (ONLY works in paper trading)
ALLOW_SHORT_SELLING = False

# Multi-Agent System
USE_AGENT_SYSTEM = False
