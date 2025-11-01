"""
Auto-Trader Background Service
Runs the auto-trader continuously without the Streamlit UI
"""

import logging
import time
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging with UTF-8 encoding for file, ASCII-safe for console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/autotrader_background.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the auto-trader continuously"""
    logger.info("=" * 80)
    logger.info("Auto-Trader Background Service Started")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("=" * 80)
    
    try:
        # Import required modules
        from services.auto_trader import AutoTrader, AutoTraderConfig
        from src.integrations.tradier_client import TradierClient
        from services.ai_trading_signals import AITradingSignalGenerator
        
        # Load configuration based on trading mode
        logger.info("Loading configuration...")
        
        is_paper_trading = os.getenv('IS_PAPER_TRADING', 'true').lower() == 'true'
        
        # Load appropriate config file based on trading mode
        # Priority order:
        # 1. config_warrior_scalping.py (if TRADING_MODE="WARRIOR_SCALPING")
        # 2. config_background_trader.py (general scalping/stocks/options)
        # 3. config_paper_trading.py / config_live_trading.py (legacy)
        # 4. Fallback defaults
        
        cfg = None
        config_source = None
        
        # Try warrior scalping config first
        try:
            import config_warrior_scalping as warrior_cfg
            if hasattr(warrior_cfg, 'TRADING_MODE') and warrior_cfg.TRADING_MODE == "WARRIOR_SCALPING":
                cfg = warrior_cfg
                config_source = "config_warrior_scalping.py"
                logger.info("⚔️ Using WARRIOR SCALPING configuration (config_warrior_scalping.py)")
        except (ImportError, AttributeError) as e:
            logger.debug(f"Warrior scalping config not found or invalid: {e}")
        
        # Try background trader config
        if cfg is None:
            try:
                import config_background_trader as bg_cfg
                cfg = bg_cfg
                config_source = "config_background_trader.py"
                logger.info(f"📋 Using BACKGROUND TRADER configuration (config_background_trader.py)")
                logger.info(f"   Mode: {cfg.TRADING_MODE}")
            except ImportError as e:
                logger.debug(f"Background trader config not found: {e}")
        
        # Fall back to paper/live trading config (legacy)
        if cfg is None:
            try:
                if is_paper_trading:
                    import config_paper_trading as cfg
                    config_source = "config_paper_trading.py"
                    logger.info("📝 Using PAPER TRADING configuration (config_paper_trading.py)")
                else:
                    import config_live_trading as cfg
                    config_source = "config_live_trading.py"
                    logger.info("💰 Using LIVE TRADING configuration (config_live_trading.py)")
                    logger.info(f"💰 Live trading capital: ${cfg.TOTAL_CAPITAL}")
                    logger.info(f"💰 Max position size: ${cfg.TOTAL_CAPITAL * cfg.MAX_POSITION_SIZE_PCT / 100:.2f}")
            except ImportError as e:
                logger.debug(f"Paper/Live trading configs not found: {e}")
        
        # Fallback to hardcoded defaults
        if cfg is None:
            class cfg:
                TRADING_MODE = "SCALPING"
                SCAN_INTERVAL_MINUTES = 15
                MIN_CONFIDENCE = 75.0
                MAX_DAILY_ORDERS = 10
                MAX_POSITION_SIZE_PCT = 20.0
                USE_BRACKET_ORDERS = True
                SCALPING_TAKE_PROFIT_PCT = 2.0
                SCALPING_STOP_LOSS_PCT = 1.0
                USE_SETTLED_FUNDS_ONLY = True
                RISK_PER_TRADE_PCT = 0.02
                MAX_DAILY_LOSS_PCT = 0.04
                USE_SMART_SCANNER = True
                WATCHLIST = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
                ALLOW_SHORT_SELLING = False
                USE_AGENT_SYSTEM = False
                CASH_BUCKETS = 3
                T_PLUS_SETTLEMENT_DAYS = 2
                RESERVE_CASH_PCT = 0.05
                TOTAL_CAPITAL = 10000.0
                MAX_CAPITAL_UTILIZATION_PCT = 80.0
                TRADING_START_HOUR = 9
                TRADING_START_MINUTE = 30
                TRADING_END_HOUR = 15
                TRADING_END_MINUTE = 30
                TEST_MODE = False
            config_source = "hardcoded defaults"
            logger.warning("⚠️ No config file found! Using hardcoded defaults")
        
        is_paper_trading = os.getenv('IS_PAPER_TRADING', 'true').lower() == 'true'
        
        # Initialize Tradier client (using trading mode manager)
        logger.info(f"Connecting to Tradier ({'Paper' if is_paper_trading else 'Live'} Trading)...")
        from src.integrations.trading_config import TradingMode
        trading_mode = TradingMode.PAPER if is_paper_trading else TradingMode.PRODUCTION
        tradier_client = TradierClient(trading_mode=trading_mode)
        
        # Initialize AI signal generator (gets config from environment)
        logger.info("Initializing AI signal generator...")
        signal_generator = AITradingSignalGenerator()
        
        # Watchlist configuration
        watchlist = cfg.WATCHLIST
        use_smart_scanner = cfg.USE_SMART_SCANNER
        
        logger.info("=" * 80)
        logger.info("📊 CONFIGURATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📁 Config Source: {config_source}")
        logger.info(f"🎯 Trading Mode: {cfg.TRADING_MODE}")
        logger.info(f"🧠 Smart Scanner: {'ENABLED' if use_smart_scanner else 'DISABLED'}")
        if not use_smart_scanner:
            logger.info(f"📋 Watchlist: {len(watchlist)} tickers - {', '.join(watchlist[:10])}{'...' if len(watchlist) > 10 else ''}")
        else:
            logger.info(f"📋 Watchlist: IGNORED (Smart Scanner will auto-discover tickers)")
        logger.info(f"🎲 Min Confidence: {cfg.MIN_CONFIDENCE}%")
        logger.info(f"⏱️  Scan Interval: {cfg.SCAN_INTERVAL_MINUTES} minutes")
        logger.info(f"💰 Capital: ${getattr(cfg, 'TOTAL_CAPITAL', 10000.0):,.2f}")
        logger.info(f"🎯 Max Position Size: {cfg.MAX_POSITION_SIZE_PCT}%")
        logger.info(f"📝 Paper Trading: {is_paper_trading}")
        logger.info("=" * 80)
        
        # Configure auto-trader
        # Handle both SCALPING_* and WARRIOR_* attribute names
        take_profit_pct = getattr(cfg, 'WARRIOR_TAKE_PROFIT_PCT', None) or getattr(cfg, 'SCALPING_TAKE_PROFIT_PCT', 2.0)
        stop_loss_pct = getattr(cfg, 'WARRIOR_STOP_LOSS_PCT', None) or getattr(cfg, 'SCALPING_STOP_LOSS_PCT', 1.0)
        
        config = AutoTraderConfig(
            enabled=True,
            scan_interval_minutes=cfg.SCAN_INTERVAL_MINUTES,
            min_confidence=cfg.MIN_CONFIDENCE,
            max_daily_orders=cfg.MAX_DAILY_ORDERS,
            max_position_size_pct=cfg.MAX_POSITION_SIZE_PCT,
            use_bracket_orders=cfg.USE_BRACKET_ORDERS,
            trading_mode=cfg.TRADING_MODE,
            scalping_take_profit_pct=take_profit_pct,
            scalping_stop_loss_pct=stop_loss_pct,
            paper_trading=is_paper_trading,
            # Capital Management (NEW)
            total_capital=getattr(cfg, 'TOTAL_CAPITAL', 10000.0),
            reserve_cash_pct=getattr(cfg, 'RESERVE_CASH_PCT', 10.0),
            max_capital_utilization_pct=getattr(cfg, 'MAX_CAPITAL_UTILIZATION_PCT', 80.0),
            # PDT-safe settings
            use_settled_funds_only=cfg.USE_SETTLED_FUNDS_ONLY,
            risk_per_trade_pct=cfg.RISK_PER_TRADE_PCT,
            max_daily_loss_pct=cfg.MAX_DAILY_LOSS_PCT,
            allow_short_selling=cfg.ALLOW_SHORT_SELLING,
            use_agent_system=cfg.USE_AGENT_SYSTEM,
            cash_buckets=cfg.CASH_BUCKETS,
            t_plus_settlement_days=cfg.T_PLUS_SETTLEMENT_DAYS,
            # Trading Hours
            trading_start_hour=getattr(cfg, 'TRADING_START_HOUR', 9),
            trading_start_minute=getattr(cfg, 'TRADING_START_MINUTE', 30),
            trading_end_hour=getattr(cfg, 'TRADING_END_HOUR', 15),
            trading_end_minute=getattr(cfg, 'TRADING_END_MINUTE', 30),
            # Test mode (for testing when market is closed)
            test_mode=getattr(cfg, 'TEST_MODE', False),
        )
        
        # If WARRIOR_SCALPING mode, attach config attributes for price filtering
        if cfg.TRADING_MODE == "WARRIOR_SCALPING":
            # Attach warrior scalping config attributes to the config object
            # These are used by the WarriorTradingDetector for filtering
            config.MIN_GAP_PCT = getattr(cfg, 'MIN_GAP_PCT', 2.0)
            config.MAX_GAP_PCT = getattr(cfg, 'MAX_GAP_PCT', 20.0)
            config.MIN_PRICE = getattr(cfg, 'MIN_PRICE', 2.0)
            config.MAX_PRICE = getattr(cfg, 'MAX_PRICE', 20.0)
            config.MIN_VOLUME_RATIO = getattr(cfg, 'MIN_VOLUME_RATIO', 1.5)
            config.MAX_VOLUME_RATIO = getattr(cfg, 'MAX_VOLUME_RATIO', 10.0)
            config.USE_MARKET_WIDE_SCAN = getattr(cfg, 'USE_MARKET_WIDE_SCAN', False)
            config.USE_YFINANCE_HISTORICAL = getattr(cfg, 'USE_YFINANCE_HISTORICAL', True)
            logger.info(f"⚔️ WARRIOR_SCALPING mode: Price range ${config.MIN_PRICE}-${config.MAX_PRICE}, Gap {config.MIN_GAP_PCT}-{config.MAX_GAP_PCT}%")
        
        # Create auto-trader
        trader = AutoTrader(
            config=config,
            tradier_client=tradier_client,
            signal_generator=signal_generator,
            watchlist=watchlist,
            use_smart_scanner=use_smart_scanner
        )
        
        logger.info("Auto-trader initialized successfully")
        logger.info(f"Mode: {config.trading_mode}")
        logger.info(f"Scan interval: {config.scan_interval_minutes} minutes")
        logger.info(f"Min confidence: {config.min_confidence}%")
        logger.info("Starting auto-trader loop...")
        
        # Start the trader
        trader.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("\nReceived interrupt signal, shutting down gracefully...")
        if 'trader' in locals():
            trader.stop()
        logger.info("Auto-trader stopped")
        
    except Exception as e:
        logger.error(f"Fatal error in auto-trader: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

