"""
Configuration for Cryptocurrency Trading (Kraken Integration)
Customize your crypto trading bot settings here
Last updated: 2025-11-04

⚠️ IMPORTANT CRYPTO CONSIDERATIONS:
- 24/7 Market: No trading hours restrictions (crypto never sleeps!)
- Higher Volatility: Crypto is typically 3-5x more volatile than stocks
- Different Fee Structure: Maker/taker fees vs stock commissions
- No PDT Rules: Day trading restrictions don't apply to crypto
- Settlement: Instant settlement, no T+2 waiting period
"""

# ==============================================================================
# BROKER & TRADING CONFIGURATION
# ==============================================================================

# Broker Selection: "KRAKEN" for cryptocurrency trading
BROKER_TYPE = "KRAKEN"

# Trading Mode: "CRYPTO_SCALPING", "CRYPTO_SWING", "CRYPTO_MOMENTUM", "CRYPTO_ALL"
TRADING_MODE = "CRYPTO_SCALPING"

# Scan Interval (minutes) - More frequent for volatile crypto markets
SCAN_INTERVAL_MINUTES = 3  # Every 3 minutes (vs 5 for stocks)

# Minimum Confidence % (only execute signals above this)
MIN_CONFIDENCE = 70  # Higher threshold due to crypto volatility

# ==============================================================================
# CAPITAL MANAGEMENT
# ==============================================================================

# Total capital allocated to crypto trading
TOTAL_CAPITAL = 100.0  # $100 starting capital

# Reserve cash percentage (kept aside for opportunities and safety)
RESERVE_CASH_PCT = 15.0  # 15% reserve (higher than stocks due to volatility)

# Maximum capital utilization (% of usable capital that can be deployed)
MAX_CAPITAL_UTILIZATION_PCT = 85.0  # 85% max deployment

# ==============================================================================
# RISK MANAGEMENT (Crypto-Specific)
# ==============================================================================

# Daily Trading Limits
MAX_DAILY_ORDERS = 20  # More trades possible in 24/7 market
MAX_POSITION_SIZE_PCT = 12.0  # Max % per position (slightly lower for crypto volatility)
RISK_PER_TRADE_PCT = 0.015  # 1.5% risk per trade (tighter for volatility)
MAX_DAILY_LOSS_PCT = 0.05  # 5% max daily loss

# Crypto-Specific Stop Loss & Take Profit
USE_BRACKET_ORDERS = True
CRYPTO_SCALPING_TAKE_PROFIT_PCT = 3.0  # 3% target (higher than stocks)
CRYPTO_SCALPING_STOP_LOSS_PCT = 1.5   # 1.5% stop loss (tighter for volatility)

# Swing Trading (for lower time frame positions)
CRYPTO_SWING_TAKE_PROFIT_PCT = 8.0    # 8% target for swing trades
CRYPTO_SWING_STOP_LOSS_PCT = 3.5      # 3.5% stop loss for swing trades

# Momentum Trading (for strong trending moves)
CRYPTO_MOMENTUM_TAKE_PROFIT_PCT = 12.0  # 12% target for momentum plays
CRYPTO_MOMENTUM_STOP_LOSS_PCT = 5.0     # 5% stop loss for momentum

# Trailing Stop Loss (recommended for crypto)
USE_TRAILING_STOP = True
TRAILING_STOP_PCT = 2.0  # Trail by 2% from peak

# ==============================================================================
# CRYPTO-SPECIFIC FEATURES
# ==============================================================================

# No PDT restrictions in crypto!
USE_SETTLED_FUNDS_ONLY = False  # Instant settlement in crypto
CASH_BUCKETS = 1  # No need for cash rotation
T_PLUS_SETTLEMENT_DAYS = 0  # Instant settlement

# Time-of-Day Trading (even though 24/7, some hours are better)
ENABLE_TIME_BASED_TRADING = True
PREFERRED_TRADING_HOURS = {
    # UTC times when crypto markets are most active
    'PEAK_VOLUME_START': 13,  # 1 PM UTC (8 AM EST - US market open)
    'PEAK_VOLUME_END': 21,    # 9 PM UTC (4 PM EST - US market close)
    'ASIA_SESSION_START': 0,   # Midnight UTC (9 AM JST)
    'ASIA_SESSION_END': 8,     # 8 AM UTC (5 PM JST)
}

# Trade 24/7 or only during peak hours?
TRADE_24_7 = False  # Set to True for round-the-clock trading
TRADE_WEEKENDS = True  # Crypto markets are active on weekends

# Volatility-Based Position Sizing
USE_VOLATILITY_ADJUSTMENT = True
HIGH_VOLATILITY_MULTIPLIER = 0.7  # Reduce position size by 30% in high volatility
LOW_VOLATILITY_MULTIPLIER = 1.2   # Increase position size by 20% in low volatility

# ==============================================================================
# CRYPTO ASSET SELECTION
# ==============================================================================

# Use Smart Scanner (finds best crypto opportunities automatically)
USE_SMART_SCANNER = True

# Trading Pairs - Major Cryptocurrencies (Kraken format)
MAJOR_PAIRS = [
    'BTC/USD',   # Bitcoin
    'ETH/USD',   # Ethereum
    'SOL/USD',   # Solana
    'XRP/USD',   # Ripple
    'ADA/USD',   # Cardano
    'AVAX/USD',  # Avalanche
    'DOT/USD',   # Polkadot
    'MATIC/USD', # Polygon
    'LINK/USD',  # Chainlink
    'ATOM/USD',  # Cosmos
]

# Mid-Cap Altcoins (higher risk/reward)
ALTCOIN_PAIRS = [
    'UNI/USD',   # Uniswap
    'AAVE/USD',  # Aave
    'MKR/USD',   # Maker
    'COMP/USD',  # Compound
    'CRV/USD',   # Curve
    'SNX/USD',   # Synthetix
    'ALGO/USD',  # Algorand
    'DOGE/USD',  # Dogecoin
    'LTC/USD',   # Litecoin
    'BCH/USD',   # Bitcoin Cash
]

# Layer 2 & Scaling Solutions
L2_PAIRS = [
    'OP/USD',    # Optimism
    'ARB/USD',   # Arbitrum
    'MATIC/USD', # Polygon (also in majors)
]

# Your Custom Watchlist (used if USE_SMART_SCANNER = False)
CRYPTO_WATCHLIST = MAJOR_PAIRS + ALTCOIN_PAIRS[:5]  # Top 10 majors + 5 altcoins

# Asset Class Weights (when using diversified strategy)
ASSET_ALLOCATION = {
    'MAJOR_PAIRS': 70.0,   # 70% in BTC, ETH, SOL, etc.
    'ALTCOINS': 25.0,      # 25% in mid-cap alts
    'L2_PAIRS': 5.0        # 5% in Layer 2 plays
}

# Minimum Trade Size (in USD)
MIN_TRADE_SIZE_USD = 20.0  # Kraken minimum for most pairs (~$10-20)

# Maximum Trade Size (in USD) - For testing phase
MAX_TRADE_SIZE_USD = 100.0  # Start small! Increase after 20+ successful trades

# ==============================================================================
# AI-POWERED HYBRID MODE (1-2 KNOCKOUT COMBO) 🥊
# ==============================================================================

# Enable ML-Enhanced Scanner for triple validation (40% ML + 35% LLM + 25% Quant)
USE_ML_ENHANCED_SCANNER = True  # RECOMMENDED: Superior trade quality

# Enable AI Capital Advisor (dynamic position sizing)
USE_AI_CAPITAL_ADVISOR = True  # RECOMMENDED: Optimizes for crypto volatility

# Best-Pick-Only Mode (focus on highest confidence when capital is limited)
BEST_PICK_ONLY_MODE = True  # Critical for small capital accounts

# Enable AI Pre-Trade Validation (final risk check before execution)
USE_AI_VALIDATION = True  # RECOMMENDED: Extra important for volatile crypto

# Minimum ensemble score for ML-Enhanced Scanner (0-100)
MIN_ENSEMBLE_SCORE = 65  # Slightly higher for crypto (vs 60 for stocks)

# Minimum AI validation confidence (0-1.0)
MIN_AI_VALIDATION_CONFIDENCE = 0.65  # Higher threshold for crypto

# ==============================================================================
# CRYPTO TECHNICAL INDICATORS (Adapted for Crypto Markets)
# ==============================================================================

# RSI Settings (crypto tends to have more extreme RSI values)
RSI_OVERSOLD = 25  # Lower threshold (vs 30 for stocks)
RSI_OVERBOUGHT = 75  # Higher threshold (vs 70 for stocks)

# Moving Averages (faster EMAs work better for crypto)
USE_EMA = True  # Exponential moving average (more responsive)
EMA_SHORT = 8   # 8-period EMA (vs 9 for stocks)
EMA_MEDIUM = 20  # 20-period EMA (vs 21 for stocks)
EMA_LONG = 50    # 50-period EMA

# MACD Settings (standard but with crypto interpretation)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands (wider bands for crypto volatility)
BB_PERIOD = 20
BB_STD_DEV = 2.5  # 2.5 standard deviations (vs 2.0 for stocks)

# Volume Analysis
USE_VOLUME_ANALYSIS = True
VOLUME_SPIKE_THRESHOLD = 2.5  # 2.5x average volume

# ==============================================================================
# CRYPTO SENTIMENT ANALYSIS
# ==============================================================================

# Social Media Sentiment (more important for crypto)
USE_SOCIAL_SENTIMENT = True
TWITTER_WEIGHT = 0.4   # Twitter/X is huge for crypto
REDDIT_WEIGHT = 0.3    # r/cryptocurrency, r/bitcoin
DISCORD_WEIGHT = 0.3   # Crypto Discord communities

# On-Chain Metrics (blockchain data)
USE_ONCHAIN_METRICS = False  # Set to True if you have access to on-chain data
WHALE_ALERT_WEIGHT = 0.5    # Large wallet movements

# Fear & Greed Index
USE_FEAR_GREED_INDEX = True  # Crypto-specific sentiment indicator

# ==============================================================================
# STRATEGY-SPECIFIC SETTINGS
# ==============================================================================

# CRYPTO_SCALPING Strategy (for quick in-and-out trades)
SCALPING_TIMEFRAME = '5min'  # 5-minute candles
SCALPING_HOLD_TIME_MINUTES = 15  # Hold for ~15 minutes
SCALPING_MIN_VOLATILITY = 2.0  # Minimum 2% daily volatility

# CRYPTO_SWING Strategy (for multi-day positions)
SWING_TIMEFRAME = '1h'  # 1-hour candles
SWING_HOLD_TIME_HOURS = 48  # Hold for ~2 days
SWING_MIN_TREND_STRENGTH = 0.6  # Minimum trend strength score

# CRYPTO_MOMENTUM Strategy (for strong trending moves)
MOMENTUM_TIMEFRAME = '15min'  # 15-minute candles
MOMENTUM_MIN_VOLUME_RATIO = 3.0  # Must have 3x+ normal volume
MOMENTUM_MIN_PRICE_CHANGE = 5.0  # Must have moved 5%+ in session

# ==============================================================================
# SAFETY & CIRCUIT BREAKERS
# ==============================================================================

# Max Drawdown Circuit Breaker
ENABLE_DRAWDOWN_PROTECTION = True
MAX_DRAWDOWN_PCT = 15.0  # Stop trading if down 15% from peak

# Flash Crash Protection
ENABLE_FLASH_CRASH_PROTECTION = True
FLASH_CRASH_THRESHOLD_PCT = 10.0  # 10% drop in 5 minutes triggers halt

# Correlation Protection (don't over-concentrate)
MAX_CORRELATED_POSITIONS = 3  # Max 3 positions in highly correlated assets

# Exchange Status Monitoring
CHECK_EXCHANGE_STATUS = True  # Verify Kraken is operational before trading

# ==============================================================================
# NOTIFICATIONS & LOGGING
# ==============================================================================

# Discord Notifications (recommended for 24/7 monitoring)
ENABLE_DISCORD_NOTIFICATIONS = True
NOTIFY_ON_TRADE = True
NOTIFY_ON_STOP_LOSS = True
NOTIFY_ON_TAKE_PROFIT = True
NOTIFY_ON_ERROR = True

# Alert Thresholds
ALERT_ON_LARGE_MOVE_PCT = 5.0  # Alert if position moves 5%+
ALERT_ON_HIGH_VOLATILITY = True

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

# Short Selling / Margin Trading (requires Kraken Pro account)
ALLOW_SHORT_SELLING = False  # Keep False until you're experienced with crypto
ALLOW_MARGIN_TRADING = False  # Keep False for safety

# Leverage (DO NOT USE unless you know what you're doing!)
MAX_LEVERAGE = 1.0  # 1.0 = no leverage (RECOMMENDED)

# Multi-Agent System
USE_MULTI_AGENT_SYSTEM = True  # Multiple AI agents for better decisions

# Backtesting Mode
BACKTESTING_MODE = False  # Set to True to test strategies without real orders

# Paper Trading Mode
# ⚠️ NOTE: Kraken does not have an official paper trading mode
# For testing, use VERY SMALL positions ($20-50) and treat as "learning capital"
PAPER_TRADING = False  # No paper trading available on Kraken

# Safety for Testing with Real Money
TEST_MODE = True  # Enables extra confirmations and logging
MIN_TRADE_SIZE_USD = 20.0  # Start with $20 minimum trades for testing
MAX_TRADE_SIZE_USD = 100.0  # Cap at $100 per trade during testing phase

# ==============================================================================
# FEE CONFIGURATION (Kraken Fee Structure)
# ==============================================================================

# Kraken fees depend on 30-day volume
# Default: 0.26% maker / 0.16% taker for <$50k volume
MAKER_FEE_PCT = 0.16  # Maker fee percentage
TAKER_FEE_PCT = 0.26  # Taker fee percentage

# Factor fees into profit targets
ADJUST_FOR_FEES = True
FEE_BUFFER_PCT = 0.5  # Add 0.5% buffer to profit targets for fees

# ==============================================================================
# TRADING SCHEDULE (Optional time restrictions)
# ==============================================================================

# Even though crypto is 24/7, you may want to restrict trading times
TRADING_SCHEDULE = {
    'MONDAY': {'enabled': True, 'hours': 'all'},
    'TUESDAY': {'enabled': True, 'hours': 'all'},
    'WEDNESDAY': {'enabled': True, 'hours': 'all'},
    'THURSDAY': {'enabled': True, 'hours': 'all'},
    'FRIDAY': {'enabled': True, 'hours': 'all'},
    'SATURDAY': {'enabled': True, 'hours': 'peak'},  # Peak hours only on weekends
    'SUNDAY': {'enabled': True, 'hours': 'peak'}
}

# ==============================================================================
# NOTES & RECOMMENDATIONS
# ==============================================================================

"""
🎯 RECOMMENDED STRATEGIES FOR CRYPTO (by risk level):

LOW RISK (Conservative):
- CRYPTO_SWING with major pairs (BTC, ETH)
- 8% take profit, 3.5% stop loss
- Hold for 2-3 days
- Focus on established coins
- Capital: $1000+

MEDIUM RISK (Balanced):
- CRYPTO_SCALPING with major + mid-cap alts
- 3% take profit, 1.5% stop loss
- Hold for 15-30 minutes
- Mix of BTC, ETH, SOL, AVAX
- Capital: $500+

HIGH RISK (Aggressive):
- CRYPTO_MOMENTUM with altcoins
- 12% take profit, 5% stop loss
- Hold during strong trends
- Focus on news-driven moves
- Capital: $200+ (but risky!)

🛡️ SAFETY CHECKLIST:
1. ✅ Start with TEST_MODE = True
2. ✅ Use MIN_TRADE_SIZE_USD = 20.0 initially
3. ✅ Keep MAX_LEVERAGE = 1.0 (no leverage)
4. ✅ Enable USE_AI_VALIDATION = True
5. ✅ Set RESERVE_CASH_PCT = 15%+
6. ✅ Use BRACKET_ORDERS = True (always!)
7. ✅ Enable DRAWDOWN_PROTECTION
8. ✅ Set up Discord notifications
9. ✅ Only invest what you can afford to lose
10. ✅ Start with $100-200 "learning capital"

📊 CRYPTO vs STOCKS - KEY DIFFERENCES:
- Volatility: 3-5x higher in crypto
- Market Hours: 24/7 (no pre-market/after-hours)
- Settlement: Instant (no T+2 wait)
- Fees: Maker/taker model (0.16-0.26% on Kraken)
- Liquidity: Variable (check orderbook depth)
- Regulation: Less regulated, higher risk
- News Impact: Social media drives prices more
- Technical Analysis: Works well but moves faster

🚀 GETTING STARTED:
1. Set up Kraken Pro account
2. Generate API keys (see KRAKEN_SETUP_GUIDE.md)
3. Deposit $100-200 as "learning capital"
4. Set TEST_MODE = True and MIN_TRADE_SIZE_USD = 20.0
5. Begin with major pairs (BTC, ETH) only
6. Use conservative settings initially
7. Make 5-10 small test trades ($20-30 each)
8. Review results after each trading session
9. Gradually increase to $50-100 per trade
10. Only scale up after 20+ successful trades
"""
