"""
Configuration for Background Auto-Trader
Customize your trading bot settings here
Last updated: 2025-11-11 08:13:20
"""

# ==============================================================================
# TRADING CONFIGURATION
# ==============================================================================

# Trading Mode: "SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"
TRADING_MODE = "WARRIOR_SCALPING"

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
RESERVE_CASH_PCT = 5.0  # 5.0% = $50 reserved

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
USE_AI_VALIDATION = False  # RECOMMENDED: Blocks high-risk trades

# Minimum ensemble score for ML-Enhanced Scanner (0-100)
MIN_ENSEMBLE_SCORE = 60  # Only trades passing all 3 systems with 70%+ score

# Minimum AI validation confidence (0-1.0)
MIN_AI_VALIDATION_CONFIDENCE = 0.7  # AI must be 70%+ confident to approve

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

# ==============================================================================
# FRACTIONAL SHARES (IBKR ONLY) 📊
# ==============================================================================

# Enable fractional share trading for expensive stocks
USE_FRACTIONAL_SHARES = False

# Auto-use fractional shares for stocks above this price
FRACTIONAL_PRICE_THRESHOLD = 100.0  # $100

# Dollar amount limits for fractional trades
FRACTIONAL_MIN_AMOUNT = 50.0  # Min $50.00 per trade
FRACTIONAL_MAX_AMOUNT = 1000.0  # Max $1000.00 per trade

# NOTE: Fractional shares only work with Interactive Brokers (IBKR)
# - Automatically uses fractional shares for stocks above threshold
# - Allows precise dollar-based position sizing (e.g., $250 in NVDA)
# - Better diversification with limited capital
