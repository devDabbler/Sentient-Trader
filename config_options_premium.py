"""
Multi-Strategy Trading Configuration - Options Premium Selling
Focus: Generate consistent income through premium selling (Wheel, Iron Condors, Credit Spreads)
"""

# ==============================================================================
# OPTIONS PREMIUM SELLING CONFIGURATION
# ==============================================================================

# Trading Mode
TRADING_MODE = "OPTIONS"

# Scan Interval (daily for options)
SCAN_INTERVAL_MINUTES = 1440  # Once per day

# Minimum Confidence %
MIN_CONFIDENCE = 65

# Risk Management
MAX_OPEN_POSITIONS = 5  # Max concurrent option positions
MAX_POSITION_SIZE_PCT = 20.0  # Max 20% per position
RISK_PER_TRADE_PCT = 0.02
MAX_DAILY_LOSS_PCT = 0.05

# ==============================================================================
# STRATEGY PREFERENCES
# ==============================================================================

# Preferred strategies (in priority order)
PREFERRED_STRATEGIES = [
    "WHEEL_STRATEGY",        # Highest priority - consistent income
    "CASH_SECURED_PUT",      # Entry into Wheel
    "COVERED_CALL",          # If assigned stock
    "IRON_CONDOR",           # Neutral high IV plays
    "CREDIT_SPREAD",         # Directional premium selling
]

# Wheel Strategy Settings
WHEEL_ENABLED = True
WHEEL_PUT_DELTA = 0.30  # Sell puts at 30 delta (~70% prob of profit)
WHEEL_CALL_DELTA = 0.30  # Sell calls at 30 delta
WHEEL_MIN_PREMIUM_PCT = 1.0  # Minimum 1% premium per month
WHEEL_DTE_RANGE = (30, 45)  # 30-45 days to expiration

# Iron Condor Settings
IRON_CONDOR_ENABLED = True
IRON_CONDOR_MIN_IV_RANK = 60  # Only trade in high IV (>60% IV rank)
IRON_CONDOR_PUT_DELTA = 0.16  # Sell puts at 16 delta (~84% prob of profit)
IRON_CONDOR_CALL_DELTA = 0.16  # Sell calls at 16 delta
IRON_CONDOR_WING_WIDTH = 5  # $5 wing width
IRON_CONDOR_MIN_CREDIT = 0.30  # Min $0.30 credit per spread
IRON_CONDOR_DTE_RANGE = (30, 45)

# Credit Spread Settings
CREDIT_SPREAD_ENABLED = True
CREDIT_SPREAD_DELTA = 0.30  # Short strike delta
CREDIT_SPREAD_WIDTH = 5  # $5 spread width
CREDIT_SPREAD_MIN_CREDIT = 0.50  # Min $0.50 credit
CREDIT_SPREAD_DTE_RANGE = (30, 45)

# ==============================================================================
# EXIT MANAGEMENT
# ==============================================================================

# Profit Taking
PROFIT_TARGET_PCT = 50  # Close at 50% of max profit
CLOSE_BEFORE_EARNINGS = True  # Close positions before earnings
CLOSE_DTE_THRESHOLD = 7  # Close at 7 DTE if not profitable

# Stop Loss
STOP_LOSS_MULTIPLIER = 2.0  # Close if loss = 2x credit received
ADJUST_AT_LOSS_PCT = 100  # Adjust position at 100% loss (1x credit)

# Rolling Rules
ROLL_ENABLED = True
ROLL_WHEN_TESTED = True  # Roll when strike is tested
ROLL_MIN_CREDIT = 0.10  # Minimum $0.10 credit to roll
ROLL_DTE_EXTENSION = 21  # Roll out 21 days

# ==============================================================================
# TICKER SELECTION - Focus on High Quality, Liquid Stocks
# ==============================================================================

# Use Smart Scanner
USE_SMART_SCANNER = False

# Watchlist (stocks you're willing to own for Wheel)
WATCHLIST = [
    # Large Cap Tech (stable, liquid)
    'AAPL',  # Apple - high volume, consistent
    'MSFT',  # Microsoft - stable premium
    'AMD',   # AMD - good volatility
    
    # High IV Plays
    'SOFI',  # SoFi - high IV, growth story
    'PLTR',  # Palantir - high IV, institutional interest
    
    # Dividend Stocks (good for Wheel)
    'WFC',   # Wells Fargo - dividend + premium
    'JPM',   # JP Morgan - stable banking
    
    # ETFs (low risk, consistent premium)
    'SPY',   # S&P 500 - liquid, tight spreads
    'QQQ',   # Nasdaq - tech exposure
    'IWM',   # Russell 2000 - higher IV
]

# ==============================================================================
# STOCK SELECTION CRITERIA
# ==============================================================================

# IV Rank Requirements
MIN_IV_RANK = 40  # Minimum 40 IV rank for premium selling
IDEAL_IV_RANK = 60  # Ideal 60+ IV rank

# Fundamental Filters (for Wheel - stocks you want to own)
MIN_MARKET_CAP_M = 1000  # $1B minimum market cap
MIN_AVG_VOLUME = 1_000_000  # 1M+ average volume
MIN_PRICE = 10  # $10 minimum price
MAX_PRICE = 200  # $200 maximum price (capital efficiency)

# Technical Filters
TREND_PREFERENCE = ["UPTREND", "SIDEWAYS"]  # Avoid downtrends for Wheel
MIN_SUPPORT_STRENGTH = 0.7  # 70% support strength at strike

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

ALLOW_SHORT_SELLING = False
USE_AGENT_SYSTEM = False

# Options-specific
USE_SPREAD_ONLY = True  # Only trade spreads (defined risk)
ALLOW_NAKED_OPTIONS = False  # Never sell naked options
MAX_BUYING_POWER_USAGE_PCT = 50  # Use max 50% buying power

# Settlement
USE_SETTLED_FUNDS_ONLY = True
CASH_BUCKETS = 2
T_PLUS_SETTLEMENT_DAYS = 1  # T+1 for options
RESERVE_CASH_PCT = 0.20  # Reserve 20% for assignments

# Monitoring
CHECK_POSITIONS_INTERVAL_MINUTES = 60  # Check every hour
ALERT_ON_ASSIGNMENT = True
ALERT_ON_TESTED_STRIKE = True
