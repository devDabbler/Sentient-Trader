"""
Configuration for Paper Trading Background Trader
Safe environment for testing strategies without real money
"""

# Watchlist (Smart Scanner will override this if enabled)
WATCHLIST = [
    'AFRM', 'AMC', 'AMD', 'BB', 'CHPT', 'COP', 'CVU', 'EVGO', 'FCEL', 'GOGY', 
    'MAIA', 'MARA', 'NOK', 'OXY', 'PINS', 'PLTR', 'PLUG', 'RIOT', 'RIVN', 'RUN', 
    'SOFI', 'SRFM', 'TLRY', 'UROY', 'WFC'
]

# Trading Mode: "SCALPING", "STOCKS", "OPTIONS", "ALL"
TRADING_MODE = "SCALPING"

# Scan Interval (minutes)
SCAN_INTERVAL_MINUTES = 5

# Minimum Confidence % (only execute signals above this)
MIN_CONFIDENCE = 65

# Capital Management - PAPER TRADING (AGGRESSIVE FOR TESTING)
TOTAL_CAPITAL = 99481.07  # Full paper account balance
RESERVE_CASH_PCT = 10.0  # Keep 10% in reserve
MAX_CAPITAL_UTILIZATION_PCT = 80.0  # Max 80% deployed

# Risk Management - AGGRESSIVE FOR PAPER TESTING
MAX_DAILY_ORDERS = 15  # More trades for testing
MAX_POSITION_SIZE_PCT = 5.0  # 5% per position
RISK_PER_TRADE_PCT = 0.02  # 2% risk per trade
MAX_DAILY_LOSS_PCT = 0.04  # 4% daily loss limit

# Bracket Orders (Stop-Loss & Take-Profit)
USE_BRACKET_ORDERS = True
SCALPING_TAKE_PROFIT_PCT = 1.0  # Take profit at 1% instead of 2%
SCALPING_STOP_LOSS_PCT = 0.5  # Tighter stop loss

# PDT-Safe Cash Management
USE_SETTLED_FUNDS_ONLY = True
CASH_BUCKETS = 3
T_PLUS_SETTLEMENT_DAYS = 2

# Smart Scanner (finds hot stocks automatically)
USE_SMART_SCANNER = False  # Use custom watchlist for consistency
SMART_SCANNER_MIN_VOLUME = 1000000
SMART_SCANNER_MIN_PRICE = 1.0
SMART_SCANNER_MAX_PRICE = 500.0
SMART_SCANNER_MAX_STOCKS = 25

# Advanced Settings
ALLOW_SHORT_SELLING = False  # Safer for paper trading
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_AFTER_LOSS_MINUTES = 30

# Trading Hours (Eastern Time)
TRADING_START_HOUR = 9
TRADING_START_MINUTE = 30
TRADING_END_HOUR = 15  # 3:30 PM ET (30 min before close for safety)
TRADING_END_MINUTE = 55  # Extended to 3:55 PM ET (5 min before close)

# Additional Required Settings
USE_AGENT_SYSTEM = False  # Keep simple for paper trading
