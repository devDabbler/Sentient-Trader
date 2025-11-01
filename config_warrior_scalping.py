"""
Warrior Trading Scalping Configuration
Based on Ross Cameron's Gap & Go approach

Settings:
- Price filter: $2-$20
- Gap filter: 4-10% premarket gap
- Volume filter: 2-3x average volume
- Trading window: 9:30 AM - 10:00 AM ET
- Profit target: 2% (scale out)
- Stop loss: 1% (low of breakout candle)
- Max 10 positions, $100 max loss/day
- NEW: Market-wide scanning support (S&P 500, NASDAQ 100, custom)
"""

# ====================================================================
# MARKET-WIDE SCANNING SETTINGS (NEW)
# ====================================================================

# Enable market-wide premarket gap scanning
# Set to False to use WATCHLIST only (original behavior)
USE_MARKET_WIDE_SCAN = False  # Start with False for safety, change to True when ready

# Stock universe to scan
# Options: "WATCHLIST", "SP500", "NASDAQ100", "ALL", "CUSTOM"
SCAN_UNIVERSE = "SP500"

# Maximum number of gappers to return from scan
# Limits processing time and focuses on best setups
MAX_SCAN_RESULTS = 50

# Batch processing chunk size
# Process tickers in chunks to avoid API rate limits
SCAN_CHUNK_SIZE = 50

# Data source preferences
USE_TRADIER_QUOTES = True  # Use Tradier for real-time quotes (recommended)
USE_YFINANCE_HISTORICAL = True  # Use yfinance for previous closes (free)

# Premarket scanning schedule
PREMARKET_SCAN_START_TIME = "04:00"  # 4:00 AM ET (when to start scanning)
PREMARKET_SCAN_INTERVAL = 5  # Rescan every 5 minutes until market open
PREMARKET_SCAN_END_TIME = "09:30"  # Stop scanning at market open

# Custom universe file (if SCAN_UNIVERSE = "CUSTOM")
CUSTOM_UNIVERSE_FILE = "data/custom_universe.txt"

# ====================================================================
# WATCHLIST (Used when USE_MARKET_WIDE_SCAN = False)
# ====================================================================

# Watchlist (focus on liquid, gap-prone stocks)
WATCHLIST = [
    'AAPL', 'AMD', 'TSLA', 'NVDA', 'PLTR', 'SOFI', 'RIVN',
    'MARA', 'RIOT', 'NOK', 'AMC', 'GME', 'SNAP', 'HOOD',
    'NIO', 'LCID', 'PLUG', 'FCEL', 'TLRY', 'SNDL', 'AFRM',
    'PINS', 'RBLX', 'DASH', 'UBER', 'LYFT'
]

# ====================================================================
# TRADING MODE & SCAN SETTINGS
# ====================================================================

# Trading Mode
TRADING_MODE = "WARRIOR_SCALPING"

# Scan Interval (minutes) - More frequent during morning window
SCAN_INTERVAL_MINUTES = 1  # Check every minute during 9:30-10:00 AM window

# Smart Scanner (Advanced)
# Set to True to use Advanced Scanner to find optimal tickers
# Set to False to use market-wide scan or watchlist
USE_SMART_SCANNER = False  # Disabled when using market-wide scan

# Minimum Confidence %
MIN_CONFIDENCE = 70  # Slightly lower than standard scalping for more opportunities

# Risk Management (Warrior Trading style)
MAX_DAILY_ORDERS = 10  # Max 10 positions per day (Warrior Trading rule)
MAX_POSITION_SIZE_PCT = 3.0  # 3% per position (tight risk management)
RISK_PER_TRADE_PCT = 0.01  # 1% risk per trade
MAX_DAILY_LOSS_PCT = 0.01  # Max $100 loss/day on $10k account (1%)
MAX_DAILY_LOSS_DOLLARS = 100.0  # Absolute dollar limit

# Warrior Trading Targets
USE_BRACKET_ORDERS = True
WARRIOR_TAKE_PROFIT_PCT = 2.0  # 2% profit target (scale out)
WARRIOR_STOP_LOSS_PCT = 1.0  # 1% stop loss (low of breakout candle)

# ====================================================================
# GAP & GO FILTERS (Warrior Trading Criteria)
# ====================================================================

# Gap & Go Filters
MIN_GAP_PCT = 2.0  # Minimum 2% premarket gap (lowered for market-wide scan)
MAX_GAP_PCT = 20.0  # Maximum 20% premarket gap (increased for market-wide scan)
MIN_PRICE = 2.0  # Minimum stock price $2
MAX_PRICE = 20.0  # Maximum stock price $20
MIN_VOLUME_RATIO = 1.5  # Minimum 1.5x average volume (lowered for market-wide scan)
MAX_VOLUME_RATIO = 10.0  # Maximum 10x average volume (increased for market-wide scan)

# Trading Window (Eastern Time)
TRADING_START_HOUR = 9
TRADING_START_MINUTE = 30
TRADING_END_HOUR = 10  # Focus on 9:30-10:00 AM (Warrior Trading sweet spot)
TRADING_END_MINUTE = 0

# PDT-Safe Cash Management
USE_SETTLED_FUNDS_ONLY = True
CASH_BUCKETS = 3
T_PLUS_SETTLEMENT_DAYS = 2
RESERVE_CASH_PCT = 10.0  # Keep 10% in reserve

# Capital Management
TOTAL_CAPITAL = 10000.0  # Default $10k account
MAX_CAPITAL_UTILIZATION_PCT = 80.0  # Max 80% deployed

# Advanced Settings
PAPER_TRADING = True  # ✅ PAPER TRADING MODE (change to False for live trading)
ALLOW_SHORT_SELLING = False  # Warrior Trading focuses on long setups
USE_AGENT_SYSTEM = False  # Keep simple for now
USE_SMART_SCANNER = False  # Use custom watchlist

# Setup Preferences
PREFERRED_SETUPS = [
    "GAP_AND_GO",  # Primary strategy
    "MICRO_PULLBACK",
    "RED_TO_GREEN",
    "BULL_FLAG"
]

# Max positions during morning window
MAX_POSITIONS_DURING_WINDOW = 5  # Max 5 concurrent positions in 9:30-10:00 window

