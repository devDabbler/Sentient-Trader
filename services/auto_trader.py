"""
Automated Trading Service
Monitors tickers and automatically executes trades based on AI signals
"""

import logging
import time
import asyncio
from datetime import datetime, time as dt_time
from typing import List, Dict, Optional
import threading
from dataclasses import dataclass
from services.cash_manager import CashManager, CashManagerConfig
from services.trade_state_manager import TradeStateManager

logger = logging.getLogger(__name__)

# Define custom exception for API errors
class APIError(Exception):
    pass

# Define custom exception for data processing errors
class DataProcessingError(Exception):
    pass

@dataclass
class AutoTraderConfig:
    """Configuration for auto-trader"""
    enabled: bool = False
    scan_interval_minutes: int = 15  # How often to scan for signals
    min_confidence: float = 75.0  # Minimum confidence to auto-execute
    max_daily_orders: int = 10
    max_position_size_pct: float = 5.0  # Max % of total capital per trade
    trading_start_hour: int = 9  # 9:30 AM ET
    trading_start_minute: int = 30
    trading_end_hour: int = 15  # 3:30 PM ET (close before market close)
    trading_end_minute: int = 30
    use_bracket_orders: bool = True  # Use stop-loss/take-profit
    risk_tolerance: str = "MEDIUM"
    paper_trading: bool = True  # Safety: start with paper trading
    trading_mode: str = "STOCKS"  # STOCKS, OPTIONS, SCALPING, WARRIOR_SCALPING, SLOW_SCALPER, MICRO_SWING, ALL
    scalping_take_profit_pct: float = 2.0  # For scalping mode
    scalping_stop_loss_pct: float = 1.0  # For scalping mode
    # Capital Management (NEW)
    total_capital: float = 10000.0  # YOUR TOTAL ACCOUNT BALANCE
    reserve_cash_pct: float = 10.0  # Keep 10% in reserve
    max_capital_utilization_pct: float = 80.0  # Max 80% of usable capital deployed
    # PDT-safe cash/risk controls
    use_settled_funds_only: bool = True
    cash_buckets: int = 3
    t_plus_settlement_days: int = 2
    risk_per_trade_pct: float = 0.02
    max_daily_loss_pct: float = 0.04
    max_consecutive_losses: int = 2
    # Multi-agent mode (new architecture for SLOW_SCALPER/MICRO_SWING)
    use_agent_system: bool = False  # Enable multi-agent architecture
    # Short selling support (paper trading only)
    # WARNING: Requires margin account in real trading. Only enable if you understand the risks.
    allow_short_selling: bool = False  # Enable short selling in paper trading mode (DISABLED by default)
    test_mode: bool = False  # Enable test mode to bypass market hours check (for testing when market is closed)


class AutoTrader:
    """Automated trading service"""
    
    def __init__(self, config: AutoTraderConfig, tradier_client, signal_generator, watchlist: List[str], use_smart_scanner: bool = False):
        """
        Initialize auto-trader
        
        Args:
            config: AutoTraderConfig settings
            tradier_client: TradierClient instance
            signal_generator: AITradingSignalGenerator instance
            watchlist: List of ticker symbols to monitor
            use_smart_scanner: If True, use Advanced Scanner to find best tickers for strategy
        """
        self.config = config
        self.tradier_client = tradier_client
        self.signal_generator = signal_generator
        self.watchlist = watchlist
        self.use_smart_scanner = use_smart_scanner
        
        self.is_running = False
        self.thread = None
        self.daily_orders = 0
        self.last_reset_date = datetime.now().date()
        self.execution_history = []
        self._cash_manager: Optional[CashManager] = None
        self._daily_realized_pnl: float = 0.0
        self._consecutive_losses: int = 0
        
        # Trade state manager (persistent state across restarts)
        self.state_manager = TradeStateManager()
        
        # Capital Manager (NEW - tracks total capital and allocations)
        from services.capital_manager import get_capital_manager
        self._capital_manager = get_capital_manager(
            total_capital=config.total_capital,
            max_position_pct=config.max_position_size_pct,
            reserve_cash_pct=config.reserve_cash_pct
        )
        logger.info(f"💰 Capital Manager initialized: ${config.total_capital:,.2f} total, {config.reserve_cash_pct}% reserve")
        
        # Short position tracking (for paper trading)
        # Format: {symbol: {'quantity': int, 'entry_price': float, 'entry_time': datetime}}
        self._short_positions: Dict[str, Dict] = {}
        
        # Multi-agent orchestrator (for SLOW_SCALPER/MICRO_SWING modes)
        self._orchestrator = None
        self._agent_loop = None
        self._agent_thread = None
        
        # Sync state with broker on startup
        self._sync_state_on_startup()
        
        logger.info(f"AutoTrader initialized with {len(watchlist)} tickers, smart_scanner={use_smart_scanner}, agent_system={config.use_agent_system}")
    
    def _sync_state_on_startup(self):
        """Sync our state manager with actual broker positions on startup"""
        try:
            success, positions = self.tradier_client.get_positions()
            if not success:
                raise APIError("Failed to retrieve positions")
            if not positions:
                logger.info("📊 No open positions found at startup")
                return
            
            logger.info(f"🔄 Syncing state with {len(positions)} broker positions...")
            self.state_manager.sync_with_broker(positions)
            
            # Log what we found
            for symbol, trade in self.state_manager.get_all_open_positions().items():
                logger.info(f"  ✅ Tracking {symbol}: {trade.side} {trade.quantity} @ ${trade.entry_price:.2f}")
        except APIError as e:
            logger.error(f"❌ API Error syncing state on startup: {e}")
        except Exception as e:
            logger.error(f"❌ Error syncing state on startup: {e}")
    
    def start(self):
        """Start the auto-trader in background thread"""
        if self.is_running:
            logger.warning("AutoTrader already running")
            return False
        
        self.is_running = True
        
        # Use agent system for SLOW_SCALPER/MICRO_SWING modes if enabled
        if self.config.use_agent_system and self.config.trading_mode in ['SLOW_SCALPER', 'MICRO_SWING']:
            logger.info("Starting multi-agent system for PDT-safe trading...")
            return self._start_agent_system()
        else:
            # Traditional signal-based approach
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            logger.info("🤖 AutoTrader started (traditional mode)")
            return True
    
    def stop(self):
        """Stop the auto-trader"""
        self.is_running = False
        
        # Stop agent system if running
        if self._orchestrator:
            self._stop_agent_system()
        
        # Stop traditional thread if running
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("🛑 AutoTrader stopped")
    
    def _run_loop(self):
        """Main trading loop"""
        logger.info("AutoTrader loop started")
        
        while self.is_running:
            try:
                # Reset daily counters if new day
                self._reset_daily_counters()
                
                # Check if within trading hours
                if not self._is_trading_hours():
                    logger.info("Outside trading hours, sleeping...")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Check daily limits
                if self.daily_orders >= self.config.max_daily_orders:
                    logger.info(f"Daily order limit reached ({self.config.max_daily_orders})")
                    time.sleep(300)  # Sleep 5 minutes
                    continue
                
                # Scan for signals
                logger.info(f"🔍 Scanning {len(self.watchlist)} tickers for signals...")
                scan_start_time = time.time()
                signals = self._scan_for_signals()
                scan_duration = time.time() - scan_start_time
                
                # Send Discord scan alert (if enabled)
                try:
                    scan_info = {
                        'tickers_found': self.watchlist if not self.use_smart_scanner else self._get_smart_watchlist(),
                        'signals_count': len(signals) if signals else 0,
                        'duration_seconds': scan_duration
                    }
                    self._send_discord_scan_alert(scan_info)
                except Exception as e:
                    logger.debug(f"Error sending scan alert: {e}")
                
                # Execute high-confidence signals
                if signals:
                    logger.info(f"Found {len(signals)} signals")
                    for signal in signals:
                        # Log signal details
                        logger.info(f"📊 {signal.symbol}: {signal.signal} signal, confidence={signal.confidence:.1f}%, min_required={self.config.min_confidence}%")
                        
                        # Check if signal is actionable (BUY or SELL)
                        if signal.signal not in ['BUY', 'SELL']:
                            logger.info(f"⏸️ {signal.symbol}: Skipping {signal.signal} signal (only BUY/SELL are executed)")
                            continue
                        
                        # Check confidence threshold
                        if signal.confidence >= self.config.min_confidence:
                            logger.info(f"✅ {signal.symbol}: Signal meets criteria, executing...")
                            self._execute_signal(signal)
                        else:
                            logger.info(f"❌ {signal.symbol}: Confidence {signal.confidence:.1f}% below threshold {self.config.min_confidence}%")
                else:
                    logger.info("No signals generated in this scan")
                
                # Sleep until next scan
                sleep_seconds = self.config.scan_interval_minutes * 60
                logger.info(f"Sleeping for {self.config.scan_interval_minutes} minutes...")
                time.sleep(sleep_seconds)
                
            except Exception as e:
                logger.error(f"Error in auto-trader loop: {e}", exc_info=True)
                time.sleep(60)  # Sleep 1 minute on error
    
    def _reset_daily_counters(self):
        """Reset counters at start of new day"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_orders = 0
            self._daily_realized_pnl = 0.0
            self._consecutive_losses = 0
            self.last_reset_date = today
            logger.info("Daily counters reset")
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours (Eastern Time)"""
        # Test mode: bypass market hours check
        if self.config.test_mode:
            logger.info("🧪 Test mode enabled: Bypassing market hours check")
            return True
        
        from datetime import timezone, timedelta
        
        # Get current time in ET (UTC-5 for EST, UTC-4 for EDT)
        # Using a simple approach: assume ET = UTC-5 (adjust for DST if needed)
        now_utc = datetime.now(timezone.utc)
        
        # Convert to ET (UTC-4 during EDT, UTC-5 during EST)
        # Simple heuristic: use UTC-4 from March-November, UTC-5 otherwise
        month = now_utc.month
        is_edt = 3 <= month <= 11  # Approximate DST period
        et_offset = timedelta(hours=-4 if is_edt else -5)
        now_et = now_utc + et_offset
        
        # Check if weekend
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        start_time = dt_time(self.config.trading_start_hour, self.config.trading_start_minute)
        end_time = dt_time(self.config.trading_end_hour, self.config.trading_end_minute)
        current_time_et = now_et.time()
        
        in_hours = start_time <= current_time_et <= end_time
        
        if not in_hours:
            logger.debug(f"Outside trading hours: ET time is {current_time_et.strftime('%H:%M')}, market hours are {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}")
        
        return in_hours
    
    def _get_smart_watchlist(self) -> List[str]:
        """
        Use Enhanced Multi-Source Discovery to find optimal tickers
        Combines: Technical scanner + Sentiment + Market screeners
        """
        try:
            # Check if enhanced discovery is enabled
            use_enhanced = getattr(self.config, 'USE_ENHANCED_DISCOVERY', True)
            
            if use_enhanced:
                # Use Enhanced Ticker Discovery (multi-source)
                from services.enhanced_ticker_discovery import get_enhanced_discovery
                
                logger.info(f"🚀 Enhanced Multi-Source Discovery enabled for {self.config.trading_mode}")
                
                # Define fallback universe for this strategy
                strategy_universes = {
                    "SCALPING": [
                        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
                        'PLTR', 'SOFI', 'RIVN', 'PLUG', 'NOK', 'AMC', 'GME', 'MARA',
                        'RIOT', 'COIN', 'HOOD', 'SNAP', 'UBER', 'LYFT', 'NIO', 'LCID'
                    ],
                    "WARRIOR_SCALPING": [
                        'AAPL', 'AMD', 'TSLA', 'NVDA', 'PLTR', 'SOFI', 'RIVN',
                        'MARA', 'RIOT', 'NOK', 'AMC', 'GME', 'SNAP', 'HOOD',
                        'NIO', 'LCID', 'PLUG', 'FCEL', 'TLRY', 'SNDL', 'AFRM',
                        'PINS', 'RBLX', 'DASH', 'UBER', 'LYFT'
                    ],
                    "STOCKS": [
                        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
                        'NFLX', 'DIS', 'PLTR', 'SOFI', 'COIN', 'RBLX', 'ABNB', 'DASH',
                        'SHOP', 'SNOW', 'CRWD', 'ZS', 'DDOG', 'NIO', 'RIVN', 'PLUG'
                    ],
                }
                
                fallback_universe = strategy_universes.get(self.config.trading_mode, strategy_universes["WARRIOR_SCALPING"])
                
                # Get enhanced discovery system
                discovery = get_enhanced_discovery(
                    tradier_client=self.tradier_client,
                    min_confidence=60.0
                )
                
                # Discover tickers using multiple sources
                discovered_tickers = discovery.discover_tickers(
                    strategy=self.config.trading_mode,
                    use_sentiment=getattr(self.config, 'USE_SENTIMENT_DISCOVERY', True),
                    use_screeners=getattr(self.config, 'USE_SCREENER_DISCOVERY', True),
                    use_social=getattr(self.config, 'USE_SOCIAL_DISCOVERY', False),
                    max_tickers=20,
                    fallback_universe=fallback_universe
                )
                
                if discovered_tickers:
                    logger.info(f"✅ Enhanced Discovery found {len(discovered_tickers)} high-confidence tickers")
                    return discovered_tickers
                else:
                    logger.warning("⚠️ Enhanced Discovery found no tickers, using fallback universe")
                    return fallback_universe[:15]
            
            else:
                # Fall back to original Smart Scanner (technical only)
                from services.advanced_opportunity_scanner import AdvancedOpportunityScanner, ScanType
                
                strategy_universes = {
                    "SCALPING": [
                        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
                        'PLTR', 'SOFI', 'RIVN', 'PLUG', 'NOK', 'AMC', 'GME', 'MARA',
                        'RIOT', 'COIN', 'HOOD', 'SNAP', 'UBER', 'LYFT', 'NIO', 'LCID'
                    ],
                    "WARRIOR_SCALPING": [
                        'AAPL', 'AMD', 'TSLA', 'NVDA', 'PLTR', 'SOFI', 'RIVN',
                        'MARA', 'RIOT', 'NOK', 'AMC', 'GME', 'SNAP', 'HOOD',
                        'NIO', 'LCID', 'PLUG', 'FCEL', 'TLRY', 'SNDL', 'AFRM',
                        'PINS', 'RBLX', 'DASH', 'UBER', 'LYFT'
                    ],
                    "STOCKS": [
                        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
                        'NFLX', 'DIS', 'PLTR', 'SOFI', 'COIN', 'RBLX', 'ABNB', 'DASH'
                    ],
                }
                
                custom_universe = strategy_universes.get(self.config.trading_mode, strategy_universes["WARRIOR_SCALPING"])
                
                scan_config = {
                    "SCALPING": {"scan_type": ScanType.MOMENTUM, "trading_style": "SCALP", "top_n": 10},
                    "WARRIOR_SCALPING": {"scan_type": ScanType.MOMENTUM, "trading_style": "SCALP", "top_n": 10},
                    "STOCKS": {"scan_type": ScanType.ALL, "trading_style": "SWING_TRADE", "top_n": 15},
                }
                
                config = scan_config.get(self.config.trading_mode, scan_config["WARRIOR_SCALPING"])
                
                logger.info(f"🔍 Smart Scanner: Scanning {len(custom_universe)} curated tickers for {self.config.trading_mode} strategy...")
                
                scanner = AdvancedOpportunityScanner(use_ai=False)
                opportunities = scanner.scan_opportunities(
                    scan_type=config["scan_type"],
                    trading_style=config["trading_style"],
                    top_n=config["top_n"],
                    custom_tickers=custom_universe,
                    use_extended_universe=False
                )
                
                smart_tickers = [opp.ticker for opp in opportunities if opp.score >= 60]
                
                if smart_tickers:
                    logger.info(f"✅ Smart Scanner found {len(smart_tickers)} optimal tickers: {', '.join(smart_tickers[:5])}...")
                    return smart_tickers
                else:
                    logger.warning("⚠️ Smart Scanner found no tickers, falling back to watchlist")
                    return self.watchlist
                
        except Exception as e:
            logger.error(f"Error in smart discovery: {e}", exc_info=True)
            return self.watchlist
    
    def _scan_for_signals(self) -> List:
        """Scan watchlist for trading signals"""
        try:
            # Use Warrior Trading detector for WARRIOR_SCALPING mode
            if self.config.trading_mode == "WARRIOR_SCALPING":
                return self._scan_warrior_trading_signals()
            
            from analyzers.comprehensive import ComprehensiveAnalyzer
            
            # Use smart scanner if enabled, otherwise use watchlist
            tickers_to_scan = self._get_smart_watchlist() if self.use_smart_scanner else self.watchlist
            
            logger.info(f"🔍 Scanning {len(tickers_to_scan)} tickers for signals...")
            
            # Collect data for all symbols
            technical_data_dict = {}
            news_data_dict = {}
            sentiment_data_dict = {}
            
            for symbol in tickers_to_scan:
                try:
                    analysis = ComprehensiveAnalyzer.analyze_stock(symbol, "OPTIONS")
                    
                    if analysis:
                        technical_data_dict[symbol] = {
                            'price': analysis.price,
                            'change_pct': analysis.change_pct,
                            'rsi': analysis.rsi,
                            'macd_signal': analysis.macd_signal,
                            'trend': analysis.trend,
                            'volume': analysis.volume,
                            'avg_volume': analysis.avg_volume,
                            'support': analysis.support,
                            'resistance': analysis.resistance,
                            'iv_rank': analysis.iv_rank
                        }
                        news_data_dict[symbol] = analysis.recent_news
                        sentiment_data_dict[symbol] = {
                            'score': analysis.sentiment_score,
                            'signals': analysis.sentiment_signals
                        }
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Get account balance and current positions
            account_balance = 10000.0
            settled_cash = None
            current_positions = []  # List of symbols you currently own
            
            try:
                success, bal_data = self.tradier_client.get_account_balance()
                if not success:
                    raise APIError("Failed to retrieve account balance")
                if not isinstance(bal_data, dict):
                    raise DataProcessingError("Invalid account balance data")
                
                b = bal_data.get('balances', {})
                settled_cash = float(b.get('cash_available', b.get('total_cash', 10000.0)))
                account_balance = float(b.get('total_equity', settled_cash or 10000.0))
                
                # SYNC CAPITAL MANAGER WITH LIVE BALANCE (NEW)
                if self._capital_manager:
                    # Update capital manager's total based on live broker balance
                    broker_total = account_balance
                    config_total = self.config.total_capital
                    
                    # If broker balance differs significantly from config, log it
                    diff_pct = abs(broker_total - config_total) / config_total * 100 if config_total > 0 else 0
                    if diff_pct > 5:  # More than 5% difference
                        logger.warning(f"💰 Broker balance (${broker_total:,.2f}) differs from config (${config_total:,.2f}) by {diff_pct:.1f}%")
                        logger.info(f"💰 Using LIVE broker balance: ${broker_total:,.2f}")
                        # Update capital manager's total to match broker
                        self._capital_manager.total_capital = broker_total
                        self._capital_manager.usable_capital = broker_total * (1 - self._capital_manager.reserve_cash_pct / 100)
                
            except APIError as e:
                logger.error(f"API Error getting account balance: {e}")
            except DataProcessingError as e:
                logger.error(f"Error processing account balance data: {e}")
            except Exception as e:
                logger.error(f"Error getting account balance: {e}")
            
            # Get current positions to inform AI
            try:
                success, positions = self.tradier_client.get_positions()
                if not success:
                    raise APIError("Failed to retrieve positions")
                
                if not positions or positions == []:
                    logger.info("📊 No open positions found")
                    current_positions = []  # Empty list, not early return!
                else:
                    current_positions = [pos.get('symbol') for pos in positions if pos.get('symbol')]
                    logger.info(f"📊 Current positions: {current_positions}")
            except APIError as e:
                logger.error(f"API Error getting positions: {e}")
            except Exception as e:
                logger.error(f"Error getting positions: {e}")

            if self._cash_manager is None:
                cm_cfg = CashManagerConfig(
                    initial_settled_cash=float(settled_cash or account_balance),
                    num_buckets=self.config.cash_buckets,
                    t_plus_days=self.config.t_plus_settlement_days,
                    use_settled_only=self.config.use_settled_funds_only,
                )
                self._cash_manager = CashManager(cm_cfg)
            
            # Generate signals with position awareness
            signals = self.signal_generator.batch_analyze(
                symbols=self.watchlist,
                technical_data_dict=technical_data_dict,
                news_data_dict=news_data_dict,
                sentiment_data_dict=sentiment_data_dict,
                account_balance=account_balance,
                risk_tolerance=self.config.risk_tolerance,
                current_positions=current_positions  # Pass current holdings to AI
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error scanning for signals: {e}")
            return []
    
    def _scan_warrior_trading_signals(self) -> List:
        """Scan for Warrior Trading Gap & Go signals"""
        try:
            from services.warrior_trading_detector import WarriorTradingDetector
            from services.ai_trading_signals import TradingSignal
            
            # Initialize detector with config
            detector = WarriorTradingDetector(
                min_gap_pct=self.config.MIN_GAP_PCT if hasattr(self.config, 'MIN_GAP_PCT') else 2.0,
                max_gap_pct=self.config.MAX_GAP_PCT if hasattr(self.config, 'MAX_GAP_PCT') else 20.0,
                min_price=self.config.MIN_PRICE if hasattr(self.config, 'MIN_PRICE') else 2.0,
                max_price=self.config.MAX_PRICE if hasattr(self.config, 'MAX_PRICE') else 20.0,
                min_volume_ratio=self.config.MIN_VOLUME_RATIO if hasattr(self.config, 'MIN_VOLUME_RATIO') else 1.5,
                max_volume_ratio=self.config.MAX_VOLUME_RATIO if hasattr(self.config, 'MAX_VOLUME_RATIO') else 10.0,
                config=self.config,
                watchlist=self.watchlist
            )
            
            # Check if Smart Scanner is enabled (takes priority)
            if self.use_smart_scanner:
                logger.info("🧠 Smart Scanner enabled: Finding best opportunities automatically...")
                tickers_to_scan = self._get_smart_watchlist()
                if not tickers_to_scan:
                    logger.warning("⚠️ Smart Scanner found no tickers, falling back to watchlist")
                    tickers_to_scan = self.watchlist
                else:
                    logger.info(f"✅ Smart Scanner found {len(tickers_to_scan)} optimal tickers: {', '.join(tickers_to_scan[:10])}{'...' if len(tickers_to_scan) > 10 else ''}")
            # Check if market-wide scan is enabled
            elif getattr(self.config, 'USE_MARKET_WIDE_SCAN', False):
                # Market-wide scan: discover tickers automatically
                logger.info("🌍 Market-wide scan enabled, discovering gappers...")
                gappers = detector.scan_market_for_gappers(
                    tradier_client=self.tradier_client,
                    use_yfinance=getattr(self.config, 'USE_YFINANCE_HISTORICAL', True)
                )
                
                # Extract ticker symbols from results
                tickers_to_scan = [g['ticker'] for g in gappers]
                
                # Log discovered gappers
                if gappers:
                    logger.info(f"✅ Market scan found {len(gappers)} qualified gappers")
                    for g in gappers[:5]:  # Show top 5
                        logger.info(f"  {g['ticker']}: {g['gap_pct']:+.2f}% gap, "
                                  f"{g['volume_ratio']:.1f}x volume, ${g['current_price']:.2f}")
                else:
                    logger.info("No gappers found in market scan")
                    return []
            else:
                # Traditional approach: use watchlist
                logger.info(f"📋 Using watchlist: {len(self.watchlist)} tickers: {', '.join(self.watchlist[:10])}{'...' if len(self.watchlist) > 10 else ''}")
            
            # Filter tickers by price range first (for WARRIOR_SCALPING)
            min_price = getattr(self.config, 'MIN_PRICE', 2.0)
            max_price = getattr(self.config, 'MAX_PRICE', 20.0)
            
            # Filter watchlist by price range before scanning
            if min_price or max_price:
                logger.info(f"⚔️ Filtering watchlist by price range: ${min_price}-${max_price}")
                try:
                    # Get current prices for watchlist tickers and filter
                    filtered_tickers = []
                    for ticker in tickers_to_scan:
                        try:
                            success, quote = self.tradier_client.get_quote(ticker)
                            if success and quote:
                                price = float(quote.get('last', 0) or quote.get('bid', 0) or 0)
                                if min_price <= price <= max_price:
                                    filtered_tickers.append(ticker)
                                    logger.debug(f"  ✅ {ticker}: ${price:.2f} (in range)")
                                else:
                                    logger.debug(f"  ❌ {ticker}: ${price:.2f} (outside ${min_price}-${max_price} range)")
                            else:
                                # If we can't get quote, include it and let detector filter it
                                filtered_tickers.append(ticker)
                        except Exception as e:
                            logger.debug(f"Error checking price for {ticker}: {e}")
                            # Include it and let detector filter it
                            filtered_tickers.append(ticker)
                    
                    if filtered_tickers:
                        tickers_to_scan = filtered_tickers
                        logger.info(f"✅ Filtered to {len(filtered_tickers)} tickers in price range ${min_price}-${max_price}: {', '.join(filtered_tickers[:10])}{'...' if len(filtered_tickers) > 10 else ''}")
                    else:
                        logger.warning(f"⚠️ No tickers in price range ${min_price}-${max_price}, using original watchlist")
                except Exception as e:
                    logger.warning(f"Error filtering watchlist by price: {e}, using original watchlist")
            
            # Continue with existing scan_for_setups logic
            # Scan for setups during trading window (9:30-10:00 AM), or bypass for test mode
            bypass_window = getattr(self.config, 'test_mode', False)
            if bypass_window:
                logger.info("🧪 Test mode: Bypassing trading window check")
            
            warrior_signals = detector.scan_for_setups(
                tickers=tickers_to_scan,
                trading_window_start=dt_time(
                    getattr(self.config, 'TRADING_START_HOUR', 9),
                    getattr(self.config, 'TRADING_START_MINUTE', 30)
                ),
                trading_window_end=dt_time(
                    getattr(self.config, 'TRADING_END_HOUR', 10),
                    getattr(self.config, 'TRADING_END_MINUTE', 0)
                ),
                bypass_window_check=bypass_window
            )
            
            if not warrior_signals:
                logger.debug("No Warrior Trading setups found")
                return []
            
            # Convert Warrior Trading signals to TradingSignal format
            signals = []
            for warrior_signal in warrior_signals:
                # Check confidence threshold
                if warrior_signal.confidence < self.config.min_confidence:
                    logger.debug(f"{warrior_signal.ticker}: Warrior setup confidence {warrior_signal.confidence:.1f}% below threshold {self.config.min_confidence}%")
                    continue
                
                # Create TradingSignal object
                signal = TradingSignal(
                    symbol=warrior_signal.ticker,
                    signal='BUY',  # Warrior Trading focuses on long setups
                    confidence=warrior_signal.confidence,
                    entry_price=warrior_signal.entry_price,
                    target_price=warrior_signal.profit_target,
                    stop_loss=warrior_signal.stop_loss,
                    reasoning=f"Warrior {warrior_signal.setup_type.value}: {warrior_signal.reasoning}",
                    timestamp=warrior_signal.timestamp
                )
                
                signals.append(signal)
                logger.info(f"⚔️ Warrior Trading signal: {warrior_signal.ticker} - {warrior_signal.setup_type.value} (confidence: {warrior_signal.confidence:.1f}%)")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error scanning for Warrior Trading signals: {e}", exc_info=True)
            return []
    
    def _execute_signal(self, signal):
        """Execute a trading signal"""
        try:
            logger.info(f"🎯 Executing signal: {signal.symbol} {signal.signal} (confidence: {signal.confidence}%)")
            
            # Check current positions with retry handling
            success, positions = self.tradier_client.get_positions()
            if not success:
                raise APIError("Failed to retrieve positions")
            
            has_position = False
            position_quantity = 0
            
            if positions:
                for pos in positions:
                    if pos.get('symbol') == signal.symbol:
                        has_position = True
                        position_quantity = int(pos.get('quantity', 0))
                        break
            
            # Check for short positions (tracked internally for paper trading)
            has_short_position = signal.symbol in self._short_positions
            short_quantity = self._short_positions.get(signal.symbol, {}).get('quantity', 0)
            
            # Validate SELL orders
            if signal.signal == 'SELL':
                if has_position:
                    # Closing a long position
                    logger.info(f"✅ SELL signal validated - closing {position_quantity} share LONG position in {signal.symbol}")
                elif has_short_position:
                    # Already short - skip adding to short
                    logger.info(f"Already have SHORT position in {signal.symbol} ({short_quantity} shares), skipping additional SHORT")
                    return
                elif self.config.allow_short_selling and self.config.paper_trading:
                    # Opening a short position
                    logger.info(f"✅ SELL signal validated - opening SHORT position in {signal.symbol} (paper trading)")
                else:
                    # Not allowed to short
                    logger.warning(f"❌ Cannot SELL {signal.symbol} - no existing long position and short selling is disabled")
                    return
            
            # Validate BUY orders
            if signal.signal == 'BUY':
                if has_short_position:
                    # Covering a short position
                    logger.info(f"✅ BUY signal validated - covering {short_quantity} share SHORT position in {signal.symbol}")
                elif has_position or self.state_manager.has_open_position(signal.symbol):
                    # Already long - skip adding to position
                    logger.info(f"Already have LONG position in {signal.symbol} ({position_quantity} shares), skipping additional BUY")
                    return
                else:
                    # Opening a long position
                    logger.info(f"✅ BUY signal validated - opening LONG position in {signal.symbol}")
                    
                    # CHECK CAPITAL AVAILABILITY (NEW)
                    if self._capital_manager:
                        available = self._capital_manager.get_available_capital()
                        utilization = self._capital_manager.get_utilization_pct()
                        
                        if utilization >= self.config.max_capital_utilization_pct:
                            logger.warning(f"⚠️ Capital utilization too high ({utilization:.1f}% >= {self.config.max_capital_utilization_pct}%), skipping {signal.symbol}")
                            return
                        
                        if available <= 0:
                            logger.warning(f"⚠️ No capital available (${available:.2f}), skipping {signal.symbol}")
                            return
                        
                        logger.info(f"💰 Capital check: ${available:,.2f} available ({utilization:.1f}% utilization)")
            
            # Guardrails: daily loss and consecutive losses
            if self._daily_realized_pnl <= -abs(self.config.max_daily_loss_pct) * 100.0:
                logger.warning("Daily loss limit reached, skipping new trades today")
                return
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                logger.warning("Max consecutive losses reached, pausing entries")
                return

            # Adjust for scalping mode
            if self.config.trading_mode == "SCALPING":
                # Use tighter stops and targets for scalping
                if signal.entry_price:
                    if signal.signal == 'BUY':
                        # For BUY (LONG): profit is higher, stop is lower
                        signal.target_price = signal.entry_price * (1 + self.config.scalping_take_profit_pct / 100)
                        signal.stop_loss = signal.entry_price * (1 - self.config.scalping_stop_loss_pct / 100)
                    else:  # SELL
                        if has_position:
                            # Closing a LONG position - DON'T USE BRACKET ORDER (just place simple sell)
                            # Set targets to None so bracket order won't be used
                            signal.target_price = None
                            signal.stop_loss = None
                            logger.info(f"📊 Scalping mode (SELL/CLOSING LONG): Entry=${signal.entry_price:.2f} - Using simple SELL order (no bracket)")
                        else:
                            # Opening a SHORT position - target is LOWER (profit when price drops), stop is HIGHER (stop when price rises)
                            signal.target_price = signal.entry_price * (1 - self.config.scalping_take_profit_pct / 100)
                            signal.stop_loss = signal.entry_price * (1 + self.config.scalping_stop_loss_pct / 100)
                            position_type = "SHORT"
                            logger.info(f"📊 Scalping mode ({signal.signal}/{position_type}): Entry=${signal.entry_price:.2f}, Target=${signal.target_price:.2f}, Stop=${signal.stop_loss:.2f}")
            
            # Adjust for Warrior Trading scalping mode
            elif self.config.trading_mode == "WARRIOR_SCALPING":
                # Warrior Trading: 2% profit, 1% stop (Gap & Go strategy)
                if signal.entry_price:
                    if signal.signal == 'BUY':
                        # For BUY (LONG): 2% profit target, 1% stop loss
                        signal.target_price = signal.entry_price * 1.02  # 2% target
                        signal.stop_loss = signal.entry_price * 0.99  # 1% stop
                        logger.info(f"⚔️ Warrior Trading (BUY): Entry=${signal.entry_price:.2f}, Target=${signal.target_price:.2f}, Stop=${signal.stop_loss:.2f}")
                    else:  # SELL
                        if has_position:
                            # Closing a LONG position
                            signal.target_price = None
                            signal.stop_loss = None
                            logger.info(f"⚔️ Warrior Trading (SELL/CLOSING): Entry=${signal.entry_price:.2f} - Using simple SELL order")
            
            # PDT-safe position sizing using settled funds
            # Skip position sizing for closing orders (use position quantity instead)
            if signal.signal == 'SELL' and has_position:
                # Closing a LONG position - use existing position quantity
                signal.position_size = position_quantity
                logger.info(f"📊 Closing {position_quantity} shares of {signal.symbol}")
            elif signal.signal == 'BUY' and has_short_position:
                # Covering a SHORT position - use existing short quantity
                signal.position_size = short_quantity
                logger.info(f"📊 Covering {short_quantity} shares SHORT of {signal.symbol}")
            elif signal.entry_price and signal.stop_loss and self._cash_manager:
                try:
                    bal_success, bal_data = self.tradier_client.get_account_balance()
                    if not bal_success:
                        raise APIError("Failed to retrieve account balance")
                    if not isinstance(bal_data, dict):
                        raise DataProcessingError("Invalid account balance data")
                    
                    settled_cash = None
                    total_equity = 10000.0
                    b = bal_data.get('balances', {})
                    settled_cash = float(b.get('cash_available', b.get('total_cash', 10000.0)))
                    total_equity = float(b.get('total_equity', settled_cash or 10000.0))
                    settled_cash = self._cash_manager.get_settled_cash(settled_cash)

                    bucket_idx = self._cash_manager.select_active_bucket()
                    bucket_cash = self._cash_manager.bucket_target_cash(settled_cash, bucket_idx)

                    risk_pct = self.config.risk_per_trade_pct
                    # If down >2% on day, halve risk
                    if self._daily_realized_pnl <= -2.0:
                        risk_pct = max(0.005, risk_pct / 2.0)

                    shares_by_risk = self._cash_manager.compute_position_size_by_risk(
                        account_equity=total_equity,
                        risk_perc=risk_pct,
                        entry_price=signal.entry_price,
                        stop_price=signal.stop_loss,
                    )
                    affordable = self._cash_manager.clamp_to_settled_cash(
                        shares=shares_by_risk,
                        entry_price=signal.entry_price,
                        settled_cash=bucket_cash,
                        reserve_pct=self.config.reserve_cash_pct,
                    )
                    
                    # Use capital manager max position size (NEW)
                    max_position_dollars = self._capital_manager.get_max_position_size() if self._capital_manager else (total_equity * (self.config.max_position_size_pct / 100.0))
                    max_by_pct = int(max_position_dollars // signal.entry_price)
                    
                    # Also check available capital (NEW)
                    if self._capital_manager:
                        available_capital = self._capital_manager.get_available_capital()
                        max_by_available = int(available_capital // signal.entry_price)
                        max_by_pct = min(max_by_pct, max_by_available)
                        logger.info(f"📊 Position sizing: risk-based={shares_by_risk}, affordable={affordable}, max_pct={max_by_pct}, avail_cap={max_by_available}")
                    
                    final_shares = max(0, min(affordable, max_by_pct))
                    if final_shares <= 0:
                        logger.info("No settled cash or capital available for this entry; skipping")
                        return
                    signal.position_size = final_shares
                except APIError as e:
                    logger.error(f"API Error sizing position with settled funds: {e}")
                except DataProcessingError as e:
                    logger.error(f"Error processing account balance data: {e}")
                except Exception as e:
                    logger.error(f"Error sizing position with settled funds: {e}")
                    return

            # Place order
            # For closing positions, use simple market/limit order (no bracket)
            if (signal.signal == 'SELL' and has_position) or (signal.signal == 'BUY' and has_short_position):
                # Closing a position - use simple market order for immediate execution
                logger.info(f"📤 Placing simple MARKET order to close position: {signal.signal} {signal.position_size} {signal.symbol}")
                
                order_data = {
                    'class': 'equity',
                    'symbol': signal.symbol,
                    'side': signal.signal.lower(),
                    'quantity': str(signal.position_size),
                    'type': 'market',
                    'duration': 'day',
                    'tag': f"AUTOCLOSE{self.config.trading_mode}{datetime.now().strftime('%Y%m%d%H%M%S')}"
                }
                
                success, result = self.tradier_client.place_order(order_data)
                
            # For opening positions, use bracket order if enabled
            elif self.config.use_bracket_orders and signal.entry_price and signal.target_price and signal.stop_loss:
                # Determine duration based on mode
                duration = 'day' if self.config.trading_mode in ["SCALPING", "WARRIOR_SCALPING"] else 'gtc'
                
                logger.info(f"📤 Placing BRACKET order to open position: {signal.signal} {signal.position_size} {signal.symbol}")
                
                success, result = self.tradier_client.place_bracket_order(
                    symbol=signal.symbol,
                    side='buy' if signal.signal == 'BUY' else 'sell',
                    quantity=signal.position_size,
                    entry_price=signal.entry_price,
                    take_profit_price=signal.target_price,
                    stop_loss_price=signal.stop_loss,
                    duration=duration,
                    tag=f"AUTO{self.config.trading_mode}{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
            else:
                # Simple market order
                success, result = self.tradier_client.place_equity_order(
                    symbol=signal.symbol,
                    side='buy' if signal.signal == 'BUY' else 'sell',
                    quantity=signal.position_size,
                    order_type='market',
                    duration='day'
                )
            
            if success:
                self.daily_orders += 1
                execution_record = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': signal.symbol,
                    'signal': signal.signal,
                    'confidence': signal.confidence,
                    'quantity': signal.position_size,
                    'entry_price': signal.entry_price,
                    'target_price': signal.target_price,
                    'stop_loss': signal.stop_loss,
                    'trading_mode': self.config.trading_mode,
                    'result': result
                }
                self.execution_history.append(execution_record)
                
                # Record trade in state manager
                try:
                    # Determine if this is opening or closing a position
                    if (signal.signal == 'SELL' and has_position) or (signal.signal == 'BUY' and has_short_position):
                        # Closing a position
                        self.state_manager.record_trade_closed(
                            symbol=signal.symbol,
                            exit_price=signal.entry_price,
                            reason=f"Signal (confidence: {signal.confidence}%)"
                        )
                        
                        # RELEASE CAPITAL (NEW)
                        if self._capital_manager:
                            # Calculate P&L
                            pnl = 0.0
                            if has_position and positions:
                                for pos in positions:
                                    if pos.get('symbol') == signal.symbol:
                                        cost_basis = float(pos.get('cost_basis', 0))
                                        current_value = signal.entry_price * signal.position_size
                                        pnl = current_value - cost_basis
                                        break
                            
                            self._capital_manager.release_capital(signal.symbol, pnl=pnl)
                            
                            # Store P/L for Discord notification
                            if not hasattr(self, '_last_pnl'):
                                self._last_pnl = {}
                            self._last_pnl[signal.symbol] = pnl
                            
                            logger.info(f"💰 Released capital for {signal.symbol} (P&L: ${pnl:+,.2f})")
                            logger.info(f"📊 Capital status: ${self._capital_manager.get_available_capital():,.2f} available ({self._capital_manager.get_utilization_pct():.1f}% utilization)")
                    else:
                        # Opening a position
                        # Extract order IDs from result
                        order_id = None
                        bracket_ids = None
                        if isinstance(result, dict):
                            order_id = result.get('id') or result.get('order', {}).get('id')
                            # For bracket orders, Tradier returns multiple order IDs
                            if 'orders' in result:
                                orders = result['orders'].get('order', [])
                                if isinstance(orders, list) and len(orders) >= 3:
                                    # Typically: [entry, take_profit, stop_loss]
                                    bracket_ids = [orders[1].get('id'), orders[2].get('id')]
                        
                        self.state_manager.record_trade_opened(
                            symbol=signal.symbol,
                            side=signal.signal,
                            quantity=signal.position_size,
                            entry_price=signal.entry_price,
                            order_id=str(order_id) if order_id else None,
                            bracket_order_ids=bracket_ids,
                            reason=f"Signal (confidence: {signal.confidence}%)"
                        )
                        
                        # ALLOCATE CAPITAL (NEW)
                        if self._capital_manager and signal.entry_price and signal.position_size:
                            capital_allocated = signal.entry_price * signal.position_size
                            strategy = self.config.trading_mode
                            allocation = self._capital_manager.allocate_capital(
                                ticker=signal.symbol,
                                strategy=strategy,
                                position_size_pct=(capital_allocated / self.config.total_capital * 100),
                                entry_price=signal.entry_price,
                                quantity=signal.position_size
                            )
                            if allocation:
                                logger.info(f"💰 Allocated ${allocation.capital_allocated:,.2f} for {signal.symbol} ({strategy})")
                                logger.info(f"📊 Capital status: ${self._capital_manager.get_available_capital():,.2f} available ({self._capital_manager.get_utilization_pct():.1f}% utilization)")
                except Exception as e:
                    logger.error(f"❌ Error recording trade in state manager: {e}")
                
                # Track short positions for paper trading
                if self.config.paper_trading and self.config.allow_short_selling:
                    if signal.signal == 'SELL' and not has_position:
                        # Opening a short position
                        self._short_positions[signal.symbol] = {
                            'quantity': signal.position_size,
                            'entry_price': signal.entry_price,
                            'entry_time': datetime.now()
                        }
                        logger.info(f"📊 Tracking SHORT position: {signal.position_size} shares of {signal.symbol} @ ${signal.entry_price:.2f}")
                    elif signal.signal == 'BUY' and has_short_position:
                        # Covering a short position
                        if signal.symbol in self._short_positions:
                            short_entry = self._short_positions[signal.symbol]['entry_price']
                            # Calculate P&L for short (profit when price goes down)
                            short_pnl = (short_entry - signal.entry_price) * signal.position_size
                            self._daily_realized_pnl += short_pnl
                            logger.info(f"💰 Covered SHORT: {signal.symbol} - P&L: ${short_pnl:.2f} (Entry: ${short_entry:.2f}, Cover: ${signal.entry_price:.2f})")
                            del self._short_positions[signal.symbol]
                
                # Journal settlement info (approximate on entry)
                try:
                    if self._cash_manager and signal.entry_price and signal.position_size:
                        side = 'BUY' if signal.signal == 'BUY' else 'SELL'
                        fr = self._cash_manager.record_fill(
                            symbol=signal.symbol,
                            side=side,
                            quantity=int(signal.position_size),
                            price=float(signal.entry_price),
                            fees=0.0,
                        )
                        execution_record['settlement_date'] = fr.settlement_date.isoformat()
                        execution_record['settled_cash_after'] = self._cash_manager.get_settled_cash()
                except Exception:
                    pass
                
                # Send Discord notification
                try:
                    self._send_discord_notification(signal, execution_record, has_position, has_short_position)
                except Exception as e:
                    logger.error(f"Failed to send Discord notification: {e}")
                
                logger.info(f"✅ Order placed successfully for {signal.symbol}")
            else:
                logger.error(f"❌ Failed to place order for {signal.symbol}: {result}")
                
        except APIError as e:
            logger.error(f"API Error executing signal for {signal.symbol}: {e}")
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}", exc_info=True)
    
    def _send_discord_notification(self, signal, execution_record, has_position, has_short_position):
        """Send Discord notification for trade execution - Enhanced for Warrior Scalping"""
        import os
        import requests
        
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if not webhook_url:
            logger.debug('DISCORD_WEBHOOK_URL not set. Skipping Discord notification.')
            return
        
        # Determine action type and get P/L for closed positions
        pnl = None
        pnl_pct = None
        if (signal.signal == 'SELL' and has_position) or (signal.signal == 'BUY' and has_short_position):
            action = "CLOSED"
            emoji = "📤"
            
            # Get P/L from the stored value (most reliable)
            if hasattr(self, '_last_pnl') and signal.symbol in self._last_pnl:
                pnl = self._last_pnl[signal.symbol]
                # Calculate P/L percentage if we have entry price
                if signal.entry_price and signal.position_size:
                    cost_basis = signal.entry_price * signal.position_size
                    if cost_basis > 0:
                        pnl_pct = (pnl / cost_basis) * 100
                # Clean up the stored value after using it
                del self._last_pnl[signal.symbol]
            
            # Set color based on P/L
            if pnl is not None:
                color = 65280 if pnl > 0 else 16711680  # Green for profit, Red for loss
            else:
                color = 15844367  # Orange if P/L unknown
        else:
            action = "OPENED"
            emoji = "⚔️" if self.config.trading_mode == "WARRIOR_SCALPING" else "📥"
            color = 15158332  # Red/Orange for Warrior Scalping entries
        
        # Enhanced title for Warrior Scalping
        if self.config.trading_mode == "WARRIOR_SCALPING":
            if action == "OPENED":
                title = f'{emoji} WARRIOR SCALPING: {signal.symbol} GAP & GO!'
            else:
                title = f'{emoji} CLOSED: {signal.symbol} {"🎉 PROFIT" if pnl and pnl > 0 else "💔 LOSS"}'
        else:
            title = f'{emoji} {action} {signal.signal} Position: {signal.symbol}'
        
        # Build enhanced embed
        embed = {
            'title': title,
            'description': f"**{signal.signal}** signal executed for **{signal.symbol}**",
            'color': color,
            'timestamp': datetime.now().isoformat(),
            'fields': [],
            'footer': {
                'text': f"Mode: {self.config.trading_mode} | Paper Trading: {self.config.paper_trading}"
            }
        }
        
        # Add fields based on action
        if action == "OPENED":
            # Entry information
            embed['fields'].extend([
                {
                    'name': '💵 Entry Price',
                    'value': f"${signal.entry_price:.2f}" if signal.entry_price else "Market",
                    'inline': True
                },
                {
                    'name': '📦 Shares',
                    'value': str(signal.position_size),
                    'inline': True
                },
                {
                    'name': '💰 Position Size',
                    'value': f"${signal.entry_price * signal.position_size:,.2f}" if signal.entry_price and signal.position_size else "N/A",
                    'inline': True
                },
                {
                    'name': '🎯 Target',
                    'value': f"${signal.target_price:.2f} (+{((signal.target_price / signal.entry_price - 1) * 100):.1f}%)" if signal.target_price and signal.entry_price else "N/A",
                    'inline': True
                },
                {
                    'name': '🛑 Stop Loss',
                    'value': f"${signal.stop_loss:.2f} ({((signal.stop_loss / signal.entry_price - 1) * 100):.1f}%)" if signal.stop_loss and signal.entry_price else "N/A",
                    'inline': True
                },
                {
                    'name': '⚖️ Risk/Reward',
                    'value': f"{((signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss)):.2f}:1" if signal.entry_price and signal.stop_loss and signal.target_price else "N/A",
                    'inline': True
                },
                {
                    'name': '📊 Confidence',
                    'value': f"{signal.confidence:.1f}%",
                    'inline': True
                },
                {
                    'name': '🕐 Time',
                    'value': datetime.now().strftime('%I:%M:%S %p ET'),
                    'inline': True
                },
                {
                    'name': '📍 Status',
                    'value': "🟢 LIVE" if not self.config.paper_trading else "📝 PAPER",
                    'inline': True
                }
            ])
        else:
            # Exit information
            embed['fields'].extend([
                {
                    'name': '💵 Exit Price',
                    'value': f"${signal.entry_price:.2f}" if signal.entry_price else "Market",
                    'inline': True
                },
                {
                    'name': '📦 Shares',
                    'value': str(signal.position_size),
                    'inline': True
                },
                {
                    'name': '🕐 Time',
                    'value': datetime.now().strftime('%I:%M:%S %p ET'),
                    'inline': True
                }
            ])
            
            # Add P/L if available
            if pnl is not None:
                pnl_emoji = "💰" if pnl > 0 else "💸"
                pnl_sign = "+" if pnl > 0 else ""
                embed['fields'].extend([
                    {
                        'name': f'{pnl_emoji} Profit/Loss',
                        'value': f"${pnl_sign}{pnl:.2f}",
                        'inline': True
                    },
                    {
                        'name': '📈 P/L %',
                        'value': f"{pnl_sign}{pnl_pct:.2f}%" if pnl_pct else "N/A",
                        'inline': True
                    },
                    {
                        'name': '📍 Status',
                        'value': "🟢 LIVE" if not self.config.paper_trading else "📝 PAPER",
                        'inline': True
                    }
                ])
        
        # Add reasoning if available (Warrior Trading setup info)
        if hasattr(signal, 'reasoning') and signal.reasoning:
            embed['fields'].append({
                'name': '💡 Setup Details',
                'value': signal.reasoning[:1024],  # Discord limit
                'inline': False
            })
        
        payload = {'embeds': [embed]}
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f'✅ Discord notification sent for {signal.symbol}')
        except requests.exceptions.RequestException as e:
            logger.error(f'❌ Failed to send Discord notification: {e}')
    
    def _send_discord_scan_alert(self, scan_info: Dict):
        """Send Discord notification for scan results - helpful for monitoring"""
        import os
        import requests
        
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if not webhook_url:
            return
        
        # Only send scan alerts if configured (to avoid spam)
        send_scan_alerts = os.getenv('DISCORD_SEND_SCAN_ALERTS', 'false').lower() == 'true'
        if not send_scan_alerts:
            return
        
        # Build scan summary embed
        tickers_found = scan_info.get('tickers_found', [])
        signals_found = scan_info.get('signals_count', 0)
        scan_duration = scan_info.get('duration_seconds', 0)
        
        if signals_found > 0:
            # Found signals - send alert
            color = 15158332  # Orange/Red
            title = f"⚔️ {signals_found} Warrior Setup{'s' if signals_found > 1 else ''} Found!"
            description = f"Smart Scanner detected **{len(tickers_found)} qualified tickers** and found **{signals_found} tradeable setup{'s' if signals_found > 1 else ''}**"
        else:
            # No signals - only send every 5 scans to avoid spam
            if not hasattr(self, '_scan_count'):
                self._scan_count = 0
            self._scan_count += 1
            
            if self._scan_count % 5 != 0:  # Only send every 5th scan
                return
            
            color = 3447003  # Blue
            title = "🔍 Scan Complete - No Setups"
            description = f"Scanned {len(tickers_found)} tickers, no qualified Gap & Go setups found"
        
        embed = {
            'title': title,
            'description': description,
            'color': color,
            'timestamp': datetime.now().isoformat(),
            'fields': [
                {
                    'name': '📊 Tickers Scanned',
                    'value': ', '.join(tickers_found[:10]) + ('...' if len(tickers_found) > 10 else ''),
                    'inline': False
                },
                {
                    'name': '⏱️ Scan Duration',
                    'value': f"{scan_duration:.1f}s",
                    'inline': True
                },
                {
                    'name': '🕐 Time',
                    'value': datetime.now().strftime('%I:%M:%S %p ET'),
                    'inline': True
                },
                {
                    'name': '📍 Status',
                    'value': "🟢 LIVE" if not self.config.paper_trading else "📝 PAPER",
                    'inline': True
                }
            ],
            'footer': {
                'text': f"Mode: {self.config.trading_mode} | Next scan in {self.config.scan_interval_minutes}min"
            }
        }
        
        payload = {'embeds': [embed]}
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.debug(f'✅ Discord scan alert sent')
        except requests.exceptions.RequestException as e:
            logger.debug(f'Failed to send Discord scan alert: {e}')
    
    def get_status(self) -> Dict:
        """Get current status of auto-trader"""
        return {
            'is_running': self.is_running,
            'daily_orders': self.daily_orders,
            'max_daily_orders': self.config.max_daily_orders,
            'watchlist_size': len(self.watchlist),
            'last_reset_date': self.last_reset_date.isoformat(),
            'execution_history_count': len(self.execution_history),
            'in_trading_hours': self._is_trading_hours(),
            'short_positions': len(self._short_positions),
            'short_positions_details': [
                {
                    'symbol': symbol,
                    'quantity': pos['quantity'],
                    'entry_price': pos['entry_price'],
                    'entry_time': pos['entry_time'].isoformat()
                }
                for symbol, pos in self._short_positions.items()
            ],
            'config': {
                'scan_interval_minutes': self.config.scan_interval_minutes,
                'min_confidence': self.config.min_confidence,
                'use_bracket_orders': self.config.use_bracket_orders,
                'paper_trading': self.config.paper_trading,
                'allow_short_selling': self.config.allow_short_selling,
                'test_mode': self.config.test_mode
            }
        }
    
    def get_execution_history(self) -> List[Dict]:
        """Get history of executed trades"""
        return self.execution_history
    
    def _start_agent_system(self) -> bool:
        """
        Start the multi-agent orchestration system.
        Used for SLOW_SCALPER and MICRO_SWING modes.
        """
        try:
            from services.agents.orchestrator import AgentOrchestrator
            import asyncio
            
            # Get account balance
            success, bal_data = self.tradier_client.get_account_balance()
            if not success:
                raise APIError("Failed to retrieve account balance")
            if not isinstance(bal_data, dict):
                raise DataProcessingError("Invalid account balance data")
            
            settled_cash = None
            total_equity = 10000.0
            b = bal_data.get('balances', {})
            settled_cash = float(b.get('cash_available', b.get('total_cash', 10000.0)))
            total_equity = float(b.get('total_equity', settled_cash or 10000.0))
            
            # Create orchestrator
            self._orchestrator = AgentOrchestrator(
                symbols=self.watchlist,
                tradier_client=self.tradier_client,
                initial_settled_cash=settled_cash,
                account_equity=total_equity,
                cash_buckets=self.config.cash_buckets,
                t_plus_days=self.config.t_plus_settlement_days
            )
            
            # Run in separate thread with asyncio
            def agent_thread_func():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._agent_loop = loop
                try:
                    loop.run_until_complete(self._orchestrator.run())
                except Exception as e:
                    logger.error(f"Agent system error: {e}", exc_info=True)
                finally:
                    loop.close()
            
            self._agent_thread = threading.Thread(target=agent_thread_func, daemon=True)
            self._agent_thread.start()
            
            logger.info("🤖 Multi-agent system started successfully")
            return True
            
        except APIError as e:
            logger.error(f"API Error starting agent system: {e}")
        except DataProcessingError as e:
            logger.error(f"Error processing account balance data: {e}")
        except Exception as e:
            logger.error(f"Failed to start agent system: {e}", exc_info=True)
            self.is_running = False
            return False
    
    def _stop_agent_system(self):
        """Stop the multi-agent system"""
        try:
            if self._orchestrator and self._agent_loop:
                # Schedule stop in agent loop
                asyncio.run_coroutine_threadsafe(
                    self._orchestrator.stop(),
                    self._agent_loop
                )
                
                # Wait for thread to finish
                if self._agent_thread:
                    self._agent_thread.join(timeout=10)
                
                logger.info("Multi-agent system stopped")
        except Exception as e:
            logger.error(f"Error stopping agent system: {e}")
    
    def get_agent_status(self) -> Optional[Dict]:
        """Get status from agent orchestrator (if running)"""
        if self._orchestrator:
            try:
                return self._orchestrator.get_status()
            except Exception as e:
                logger.error(f"Error getting agent status: {e}")
                return None
        return None
    
    def get_agent_journal_stats(self, days: int = 30) -> Optional[Dict]:
        """Get journal statistics from agent orchestrator (if running)"""
        if self._orchestrator:
            try:
                stats = self._orchestrator.get_journal_stats(days=days)
                return {
                    'total_trades': stats.total_trades,
                    'win_rate': stats.win_rate,
                    'total_pnl': stats.total_pnl,
                    'avg_win': stats.avg_win,
                    'avg_loss': stats.avg_loss,
                    'profit_factor': stats.profit_factor,
                    'avg_r_multiple': stats.avg_r_multiple,
                    'avg_hold_time_minutes': stats.avg_hold_time_minutes,
                    'setup_stats': stats.setup_stats
                }
            except Exception as e:
                logger.error(f"Error getting journal stats: {e}")
                return None
        return None


def create_auto_trader(
    tradier_client,
    signal_generator,
    watchlist: List[str],
    config: Optional[AutoTraderConfig] = None,
    use_smart_scanner: bool = False
) -> AutoTrader:
    """
    Create and configure auto-trader
    
    Args:
        tradier_client: TradierClient instance
        signal_generator: AITradingSignalGenerator instance
        watchlist: List of tickers to monitor
        config: Optional AutoTraderConfig (uses defaults if None)
        use_smart_scanner: Use Advanced Scanner to find optimal tickers
        
    Returns:
        AutoTrader instance
    """
    if config is None:
        config = AutoTraderConfig()
    
    return AutoTrader(config, tradier_client, signal_generator, watchlist, use_smart_scanner)
