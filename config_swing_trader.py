"""
Multi-Strategy Trading Configuration - Swing Trading Component
Complements scalping with medium-term positions (1-5 days)
"""

# ==============================================================================
# SWING TRADING CONFIGURATION
# ==============================================================================

# Trading Mode
TRADING_MODE = "SWING_TRADE"

# Scan Interval (minutes) - less frequent than scalping
SCAN_INTERVAL_MINUTES = 30  # Every 30 minutes

# Minimum Confidence % 
MIN_CONFIDENCE = 70  # Higher than scalping for quality setups

# Risk Management (more conservative than scalping)
MAX_DAILY_ORDERS = 5  # Fewer trades, bigger positions
MAX_POSITION_SIZE_PCT = 15.0  # Larger position size per trade
RISK_PER_TRADE_PCT = 0.03  # 3% risk per trade
MAX_DAILY_LOSS_PCT = 0.06  # 6% max daily loss

# Swing Trading Targets (wider than scalping)
USE_BRACKET_ORDERS = True
SWING_TAKE_PROFIT_PCT = 10.0  # 10% target vs 2% scalping
SWING_STOP_LOSS_PCT = 4.0  # 4% stop vs 1% scalping

# Technical Filters (quality setups only)
REQUIRE_EMA_RECLAIM = True  # Must have EMA reclaim
REQUIRE_POWER_ZONE = True   # Must be in power zone
MIN_RSI = 40                # Not overbought
MAX_RSI = 65                # Not overbought
TREND_FILTER = ["UPTREND", "STRONG UPTREND"]  # Only uptrends

# Volume Requirements
MIN_VOLUME_RATIO = 1.5  # At least 1.5x average volume

# PDT-Safe Cash Management
USE_SETTLED_FUNDS_ONLY = True
CASH_BUCKETS = 2  # Fewer buckets, larger positions
T_PLUS_SETTLEMENT_DAYS = 2
RESERVE_CASH_PCT = 0.10  # Reserve 10% cash

# ==============================================================================
# TICKER SELECTION - Same as Scalping for Consistency
# ==============================================================================

# Use Smart Scanner
USE_SMART_SCANNER = False

# Watchlist (focus on trending stocks)
WATCHLIST = [
    'PLTR',  # Strong trends, institutional interest
    'SOFI',  # High growth, good swings
    'AMD',   # Tech sector leader
    'RIOT', 'MARA',  # Crypto plays
    'RIVN',  # EV volatility
    'PLUG', 'FCEL',  # Clean energy swings
    'AFRM',  # Fintech momentum
]

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

ALLOW_SHORT_SELLING = False
USE_AGENT_SYSTEM = False

# Swing-specific settings
HOLD_MIN_HOURS = 4  # Minimum hold time (avoid whipsaws)
HOLD_MAX_DAYS = 7   # Maximum hold time (exit stale positions)
TRAIL_STOP_AFTER_PCT = 5.0  # Trail stop after 5% profit
TRAIL_STOP_DISTANCE_PCT = 2.0  # Trail 2% below high
