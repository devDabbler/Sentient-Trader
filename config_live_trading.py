"""
Configuration for Live Trading Background Trader
CONSERVATIVE settings for real money trading with $500 allocation
"""

# Watchlist (Smart Scanner will override this if enabled)
WATCHLIST = [
    'AFRM', 'AMC', 'AMD', 'BB', 'CHPT', 'COP', 'CVU', 'EVGO', 'FCEL', 'GOGY', 
    'MAIA', 'MARA', 'NOK', 'OXY', 'PINS', 'PLTR', 'PLUG', 'RIOT', 'RIVN', 'RUN', 
    'SOFI', 'SRFM', 'TLRY', 'UROY', 'WFC'
]

# Trading Mode: "SCALPING", "STOCKS", "OPTIONS", "ALL"
TRADING_MODE = "SCALPING"

# Scan Interval (minutes) - SLOWER FOR LIVE TRADING
SCAN_INTERVAL_MINUTES = 10  # More deliberate scanning

# Minimum Confidence % (only execute signals above this) - HIGHER FOR LIVE
MIN_CONFIDENCE = 75  # More selective for real money

# Capital Management - LIVE TRADING WITH $500 ALLOCATION
TOTAL_CAPITAL = 500.0  # LIVE: $500 allocation
RESERVE_CASH_PCT = 20.0  # Keep $100 in reserve (20%)
MAX_CAPITAL_UTILIZATION_PCT = 60.0  # Max $300 deployed (60%)

# Risk Management - VERY CONSERVATIVE FOR LIVE TRADING
MAX_DAILY_ORDERS = 3  # Max 3 trades per day
MAX_POSITION_SIZE_PCT = 15.0  # Max $75 per position (15% of $500)
RISK_PER_TRADE_PCT = 0.01  # Risk only $5 per trade (1%)
MAX_DAILY_LOSS_PCT = 0.02  # Stop if down $10 in a day (2%)

# Bracket Orders (Stop-Loss & Take-Profit) - TIGHTER FOR LIVE
USE_BRACKET_ORDERS = True
SCALPING_TAKE_PROFIT_PCT = 1.5  # More conservative take profit
SCALPING_STOP_LOSS_PCT = 0.75  # Tighter stop loss

# PDT-Safe Cash Management - CRITICAL FOR LIVE TRADING
USE_SETTLED_FUNDS_ONLY = True
CASH_BUCKETS = 3
T_PLUS_SETTLEMENT_DAYS = 2
RESERVE_CASH_PCT = 20.0  # Ensure this is defined for compatibility

# Smart Scanner - DISABLED FOR LIVE TRADING (use custom watchlist)
USE_SMART_SCANNER = False  # Use your specific watchlist instead
SMART_SCANNER_MIN_VOLUME = 2000000  # Higher volume requirement
SMART_SCANNER_MIN_PRICE = 5.0  # Avoid penny stocks
SMART_SCANNER_MAX_PRICE = 200.0  # Avoid very expensive stocks
SMART_SCANNER_MAX_STOCKS = 15  # Fewer stocks to focus on

# Advanced Settings - CONSERVATIVE FOR LIVE TRADING
ALLOW_SHORT_SELLING = False  # No short selling for live trading
MAX_CONSECUTIVE_LOSSES = 2  # Stop after 2 losses
COOLDOWN_AFTER_LOSS_MINUTES = 60  # Longer cooldown

# Agent System Settings
USE_AGENT_SYSTEM = False  # Keep simple for live trading
