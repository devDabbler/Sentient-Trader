"""
Configuration for Stock Informational Mode
Monitors stocks without executing trades - alerts only
Last updated: 2025-11-23
"""

# ==============================================================================
# MODE CONFIGURATION
# ==============================================================================

# Trading Mode: Set to INFORMATIONAL for alerts-only (no trading)
TRADING_MODE = "INFORMATIONAL"

# Scan Interval (minutes) - less frequent than trading modes to save costs
SCAN_INTERVAL_MINUTES = 30

# Auto-execute trades (MUST be False for informational mode)
AUTO_EXECUTE = False

# Enable Discord/Alert System
ENABLE_ALERTS = True

# Alert on opportunities even without positions
ALERT_ON_ALL_OPPORTUNITIES = True

# ==============================================================================
# LLM OPTIMIZATION (Cost-Efficient Settings)
# ==============================================================================

# LLM Priority Level: LOW for informational scanning
LLM_PRIORITY = "LOW"

# Use aggressive caching to reduce API calls
USE_CACHE_AGGRESSIVELY = True

# Cache TTL (seconds) - longer for informational mode
CACHE_TTL_SECONDS = 900  # 15 minutes

# Primary LLM Provider
PROVIDER_PREFERENCE = "openrouter"

# Default model (cost-efficient option)
DEFAULT_MODEL = "openai/gpt-4o-mini"

# Rate limiting (requests per minute)
MAX_LLM_REQUESTS_PER_MINUTE = 30

# ==============================================================================
# MONITORING SCOPE
# ==============================================================================

# Use Smart Scanner to find opportunities
USE_SMART_SCANNER = True

# Your Custom Watchlist (used if USE_SMART_SCANNER = False)
WATCHLIST = [
    'AFRM', 'AMC', 'AMD', 'AMTD', 'ARBK', 'BB', 'BTBT', 'CHPT', 'COP', 'CVU',
    'ENPH', 'EVGO', 'FCEL', 'GOGY', 'HKD', 'MAIA', 'MARA', 'NOK', 'OXY', 'PINS',
    'PLTR', 'PLUG', 'RIOT', 'RIVN', 'RUN', 'SGMO', 'SI', 'SOFI', 'SRFM', 'TLRY',
    'UROY', 'VXRT', 'WFC', 'WKHS'
]

# Market cap filters
MIN_MARKET_CAP = 500_000_000  # $500M minimum
MAX_MARKET_CAP = 50_000_000_000  # $50B maximum

# Volume filters
MIN_AVG_VOLUME = 1_000_000  # 1M shares minimum
MIN_RELATIVE_VOLUME = 1.2  # 20% above average

# ==============================================================================
# ALERT CONFIGURATION
# ==============================================================================

# Alert priority thresholds
ALERT_PRIORITY_CRITICAL = [
    "triple_threat",      # All 3 timeframes aligned
    "ema_reclaim",        # EMA reclaim with volume
    "position_loss_10",   # 10%+ unrealized loss
    "earnings_tomorrow"   # Earnings in 1-3 days
]

ALERT_PRIORITY_HIGH = [
    "timeframe_aligned",  # 2+ timeframes aligned
    "sector_leader",      # Leading its sector
    "pnl_change_5",      # 5%+ P&L change
    "merger_candidate"   # Potential merger/acquisition
]

ALERT_PRIORITY_MEDIUM = [
    "demarker_signal",   # DeMarker oversold in uptrend
    "fibonacci_target",  # Near Fibonacci extension level
    "position_update",   # Regular position monitoring
    "earnings_week"      # Earnings in 4-7 days
]

ALERT_PRIORITY_LOW = [
    "informational",     # General market updates
    "earnings_month"     # Earnings 15-30 days out
]

# ==============================================================================
# ANALYSIS SETTINGS
# ==============================================================================

# EMA Fibonacci System
USE_EMA_FIBONACCI_SYSTEM = True

# Timeframes to analyze
TIMEFRAMES = ["1wk", "1d", "4h"]

# Multi-factor validation
USE_ML_ENHANCED_SCANNER = True  # ML model for scoring
USE_AI_VALIDATION = True        # LLM for qualitative analysis

# Minimum ensemble score to trigger alert (0-100)
MIN_ENSEMBLE_SCORE = 60

# Minimum confidence for alerts (0-1.0)
MIN_ALERT_CONFIDENCE = 0.65

# ==============================================================================
# EVENT MONITORING
# ==============================================================================

# Monitor SEC filings
MONITOR_SEC_FILINGS = True

# Monitor earnings calendar
MONITOR_EARNINGS = True

# Monitor news sentiment
MONITOR_NEWS_SENTIMENT = True

# Monitor social sentiment (Reddit, Twitter, StockTwits)
MONITOR_SOCIAL_SENTIMENT = True

# Monitor economic calendar
MONITOR_ECONOMIC_CALENDAR = True

# Detect reverse merger candidates
DETECT_MERGER_CANDIDATES = True

# Detect penny stock risks (reverse splits, etc.)
DETECT_PENNY_STOCK_RISKS = True

# ==============================================================================
# CAPITAL TRACKING (Reference Only - No Trading)
# ==============================================================================

# Reference capital for position sizing calculations in alerts
REFERENCE_CAPITAL = 25000.0

# Position size for "what if" scenarios in alerts
REFERENCE_POSITION_SIZE_PCT = 5.0

# Risk per trade for "what if" scenarios
REFERENCE_RISK_PER_TRADE_PCT = 2.0

# ==============================================================================
# DISCORD INTEGRATION
# ==============================================================================

# Send alerts to Discord
USE_DISCORD_ALERTS = True

# Discord webhook URL (set in .env as DISCORD_WEBHOOK_URL)
# DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."

# Alert channel preferences
DISCORD_CHANNEL_CRITICAL = "critical-alerts"
DISCORD_CHANNEL_HIGH = "high-priority"
DISCORD_CHANNEL_MEDIUM = "opportunities"
DISCORD_CHANNEL_LOW = "informational"

# Rich embeds with charts and links
USE_RICH_EMBEDS = True

# Include TradingView chart links
INCLUDE_CHART_LINKS = True

# Include SEC filing links
INCLUDE_SEC_LINKS = True

# ==============================================================================
# LOGGING & STORAGE
# ==============================================================================

# Log all opportunities to file
LOG_OPPORTUNITIES = True
OPPORTUNITIES_LOG_PATH = "logs/stock_opportunities.json"

# Log all alerts
LOG_ALERTS = True
ALERTS_LOG_PATH = "logs/stock_alerts.json"

# Retain logs for N days
LOG_RETENTION_DAYS = 30

# ==============================================================================
# PERFORMANCE OPTIMIZATION
# ==============================================================================

# Batch API requests where possible
USE_BATCH_REQUESTS = True

# Vectorized calculations for large datasets
USE_VECTORIZED_OPERATIONS = True

# Max concurrent analysis tasks
MAX_CONCURRENT_ANALYSIS = 5

# Timeout for LLM requests (seconds)
LLM_REQUEST_TIMEOUT = 60

# ==============================================================================
# BACKTEST INTEGRATION (Optional)
# ==============================================================================

# Enable backtest mode for strategy validation
ENABLE_BACKTEST_MODE = False

# Backtest period (days)
BACKTEST_PERIOD_DAYS = 90

# Minimum metrics for strategy approval
MIN_WIN_RATE = 0.60  # 60%
MIN_PROFIT_FACTOR = 1.5

# ==============================================================================
# NOTES
# ==============================================================================

# This configuration is optimized for:
# - Cost efficiency (low LLM usage, aggressive caching)
# - Comprehensive monitoring (all event types)
# - Alert-based workflow (no trading execution)
# - Multi-factor validation (ML + LLM + Quant)
# - Discord integration for real-time notifications

# Best used for:
# - Monitoring markets without deploying capital
# - Learning and paper trading
# - Validating strategies before live trading
# - Tracking opportunities across multiple tickers
# - Staying informed on positions and market events
