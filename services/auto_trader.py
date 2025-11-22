"""
Automated Trading Service
Monitors tickers and automatically executes trades based on AI signals
"""

from loguru import logger
import time
import asyncio
from datetime import datetime, time as dt_time
from typing import List, Dict, Optional
import threading
from dataclasses import dataclass
from services.cash_manager import CashManager, CashManagerConfig
from services.trade_state_manager import TradeStateManager
from services.position_exit_monitor import PositionExitMonitor


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
    # AI-Powered Hybrid Mode (1-2 KNOCKOUT COMBO)
    use_ml_enhanced_scanner: bool = False  # Enable ML-Enhanced Scanner (40% ML + 35% LLM + 25% Quant)
    use_ai_validation: bool = False  # Enable AI pre-trade validation (secondary knockout check)
    use_ai_capital_advisor: bool = True  # Enable AI Capital Advisor for dynamic position sizing
    best_pick_only_mode: bool = True  # Focus on highest confidence trade when capital is limited
    min_ensemble_score: float = 70.0  # Minimum ensemble score for ML-Enhanced Scanner (0-100)
    min_ai_validation_confidence: float = 0.7  # Minimum AI validation confidence (0-1)
    # Price Filters (for small capital accounts)
    min_stock_price: Optional[float] = None  # Minimum stock price filter
    max_stock_price: Optional[float] = None  # Maximum stock price filter
    # Position Exit Monitoring (CRITICAL SAFETY FEATURE)
    enable_position_monitoring: bool = True  # Monitor and auto-close positions
    position_check_interval_seconds: int = 30  # How often to check positions (30s recommended)
    enable_trailing_stops: bool = True  # Enable trailing stop feature
    enable_time_limits: bool = True  # Enable time-based position exits
    enable_break_even_stops: bool = True  # Move stop to breakeven after profit threshold
    max_position_hold_minutes: int = 480  # Max hold time (8 hours default)
    # Long-Term Holdings Protection (CRITICAL SAFETY FEATURE)
    long_term_holdings: Optional[List[str]] = None  # Tickers to never sell (e.g., ['BXP', 'AAPL'])
    # AI Entry Timing (NEW - Stock Entry Assistant)
    use_ai_entry_timing: bool = False  # Enable AI entry analysis before execution
    enable_auto_entry: bool = False  # Auto-execute when monitored conditions met
    min_ai_entry_confidence: float = 85.0  # Min confidence for immediate entry
    # Fractional Shares (NEW - IBKR Fractional Share Support)
    use_fractional_shares: bool = False  # Enable fractional share trading (IBKR only)
    fractional_price_threshold: float = 100.0  # Auto-use fractional for stocks above this price
    fractional_min_amount: float = 50.0  # Minimum dollar amount per fractional trade
    fractional_max_amount: float = 1000.0  # Maximum dollar amount per fractional trade


class AutoTrader:
    """Automated trading service"""
    
    def __init__(self, config: AutoTraderConfig, broker_client, signal_generator, watchlist: List[str], use_smart_scanner: bool = False):
        """
        Initialize auto-trader
        
        Args:
            config: AutoTraderConfig settings
            broker_client: Broker client instance (TradierClient, IBKRClient, or BrokerAdapter)
            signal_generator: AITradingSignalGenerator instance
            watchlist: List of ticker symbols to monitor
            use_smart_scanner: If True, use Advanced Scanner to find best tickers for strategy
        """
        self.config = config
        self.broker_client = broker_client
        # Keep tradier_client as alias for backward compatibility
        self.tradier_client = broker_client
        self.signal_generator = signal_generator
        self.watchlist = watchlist
        self.use_smart_scanner = use_smart_scanner
        
        self.is_running = False
        self.thread = None
        self.daily_orders = 0
        self.last_reset_date = datetime.now().date()
        self.execution_history = []
        self._daily_realized_pnl: float = 0.0
        self._consecutive_losses: int = 0
        
        # Initialize Cash Manager for PDT-safe position sizing (ALWAYS initialize, even if not using settled funds)
        self._cash_manager: Optional[CashManager] = None
        try:
            from services.cash_manager import CashManager, CashManagerConfig
            cm_cfg = CashManagerConfig(
                initial_settled_cash=config.total_capital,
                num_buckets=config.cash_buckets if config.use_settled_funds_only else 1,  # Single bucket if not using settlement
                t_plus_days=config.t_plus_settlement_days,
                use_settled_only=config.use_settled_funds_only,
            )
            self._cash_manager = CashManager(cm_cfg)
            if config.use_settled_funds_only:
                logger.info(f"💵 Cash Manager initialized: {config.cash_buckets} buckets, T+{config.t_plus_settlement_days} settlement")
            else:
                logger.info(f"💵 Cash Manager initialized: 1 bucket (settled funds tracking disabled)")
        except Exception as e:
            logger.warning(f"⚠️ Cash Manager initialization failed: {e}")
        
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
        
        # AI Capital Advisor (NEW - dynamic position sizing)
        self._capital_advisor = None
        if getattr(config, 'use_ai_capital_advisor', True):
            try:
                from services.ai_capital_advisor import AICapitalAdvisor
                self._capital_advisor = AICapitalAdvisor(llm_analyzer=signal_generator if hasattr(signal_generator, 'analyze_with_llm') else None)
                logger.info("🧠 AI Capital Advisor ENABLED: Dynamic position sizing based on trade quality & capital")
            except Exception as e:
                logger.warning(f"⚠️ AI Capital Advisor not available: {e}")
        
        # Fractional Share Manager (NEW - fractional share support for IBKR)
        self._fractional_manager = None
        if getattr(config, 'use_fractional_shares', False):
            try:
                from services.fractional_share_manager import get_fractional_share_manager, FractionalShareConfig
                
                frac_config = FractionalShareConfig(
                    enabled=True,
                    min_price_threshold=getattr(config, 'fractional_price_threshold', 100.0),
                    min_dollar_amount=getattr(config, 'fractional_min_amount', 50.0),
                    max_dollar_amount=getattr(config, 'fractional_max_amount', 1000.0)
                )
                self._fractional_manager = get_fractional_share_manager(frac_config)
                pass  # logger.info(f"📊 Fractional Share Manager ENABLED: Auto-use for stocks >${} {frac_config.min_price_threshold:.2f}")
            except Exception as e:
                logger.warning(f"⚠️ Fractional Share Manager initialization failed: {e}")
                config.use_fractional_shares = False
        
        # Short position tracking (for paper trading)
        # Format: {symbol: {'quantity': int, 'entry_price': float, 'entry_time': datetime}}
        self._short_positions: Dict[str, Dict] = {}
        
        # Multi-agent orchestrator (for SLOW_SCALPER/MICRO_SWING modes)
        self._orchestrator = None
        self._agent_loop = None
        self._agent_thread = None
        
        # AI-Powered Hybrid System (1-2 KNOCKOUT COMBO)
        self._ml_scanner = None
        self._ai_validator = None
        
        # Position Exit Monitor (CRITICAL - monitors and closes positions)
        self._position_monitor = None
        self._position_monitor_thread = None
        
        if config.enable_position_monitoring:
            try:
                # Get long-term holdings list (default to empty list if None)
                long_term_holdings = config.long_term_holdings if config.long_term_holdings else []
                
                self._position_monitor = PositionExitMonitor(
                    broker_client=self.broker_client,
                    state_manager=self.state_manager,
                    capital_manager=self._capital_manager,
                    check_interval_seconds=config.position_check_interval_seconds,
                    enable_trailing_stops=config.enable_trailing_stops,
                    enable_time_limits=config.enable_time_limits,
                    enable_break_even_stops=config.enable_break_even_stops,
                    long_term_holdings=long_term_holdings
                )
                
                if long_term_holdings:
                    pass  # logger.info(f"🔒 Long-Term Holdings Protected: {', '.join(long_term_holdings} - WILL NEVER BE AUTO-SOLD"))
                
                logger.info("🛡️ Position Exit Monitor ENABLED: Active position management")
            except Exception as e:
                logger.error(f"⚠️ Failed to initialize Position Exit Monitor: {e}")
                config.enable_position_monitoring = False
        else:
            logger.warning("⚠️ Position Exit Monitor DISABLED: Positions rely on bracket orders only!")
        
        # Stock AI Entry Assistant (NEW - intelligent entry timing)
        self._stock_entry_assistant = None
        if getattr(config, 'use_ai_entry_timing', False):
            try:
                from services.ai_stock_entry_assistant import get_ai_stock_entry_assistant
                from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                
                llm_analyzer = LLMStrategyAnalyzer()
                self._stock_entry_assistant = get_ai_stock_entry_assistant(
                    broker_client=self.broker_client,
                    llm_analyzer=llm_analyzer,
                    check_interval_seconds=60,
                    enable_auto_entry=getattr(config, 'enable_auto_entry', False)
                )
                logger.info("🎯 AI Stock Entry Assistant ENABLED: Intelligent entry timing analysis")
            except Exception as e:
                logger.warning(f"⚠️ Stock Entry Assistant initialization failed: {e}")
                config.use_ai_entry_timing = False
        
        if config.use_ml_enhanced_scanner:
            try:
                from services.ml_enhanced_scanner import MLEnhancedScanner
                self._ml_scanner = MLEnhancedScanner(use_ml=True, use_llm=True)
                logger.info("🧠 ML-Enhanced Scanner ENABLED: Triple validation (40% ML + 35% LLM + 25% Quant)")
            except Exception as e:
                logger.warning(f"⚠️ ML-Enhanced Scanner initialization failed: {e}, falling back to standard mode")
                config.use_ml_enhanced_scanner = False
        
        if config.use_ai_validation:
            try:
                from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                from utils.config_loader import get_api_key
                
                api_key = get_api_key('OPENROUTER_API_KEY', 'openrouter')
                
                if api_key:
                    self._ai_validator = LLMStrategyAnalyzer(provider="openrouter", api_key=api_key)
                    logger.info("🛡️ AI Pre-Trade Validation ENABLED: Secondary knockout check before execution")
                else:
                    logger.warning("⚠️ OPENROUTER_API_KEY not found, AI validation disabled")
                    config.use_ai_validation = False
            except Exception as e:
                logger.warning(f"⚠️ AI Validator initialization failed: {e}, proceeding without validation")
                config.use_ai_validation = False
        
        # Log hybrid mode status
        if config.use_ml_enhanced_scanner and config.use_ai_validation:
            logger.info("🥊 HYBRID MODE ACTIVE: ML-Enhanced Scanner + AI Validation (1-2 KNOCKOUT COMBO)")
        elif config.use_ml_enhanced_scanner:
            logger.info("🥊 ML-Enhanced Scanner mode active (triple validation)")
        elif config.use_ai_validation:
            logger.info("🥊 AI Validation mode active (standard signals + AI check)")
        
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
            
            # Log what we found and add to position monitor
            for symbol, trade in self.state_manager.get_all_open_positions().items():
                logger.info(f"  ✅ Tracking {symbol}: {trade.side} {trade.quantity} @ ${trade.entry_price:.2f}")
                
                # Add to position monitor if enabled
                if self._position_monitor:
                    # Calculate default stops if not present
                    entry_price = trade.entry_price
                    stop_loss = entry_price * 0.99  # 1% default stop
                    take_profit = entry_price * 1.02  # 2% default target
                    
                    self._position_monitor.add_position(
                        symbol=symbol,
                        side=trade.side,
                        quantity=trade.quantity,
                        entry_price=entry_price,
                        entry_time=datetime.fromisoformat(trade.entry_time) if isinstance(trade.entry_time, str) else trade.entry_time,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        max_hold_minutes=self.config.max_position_hold_minutes,
                        bracket_order_ids=trade.bracket_order_ids
                    )
                    logger.info(f"  🛡️ Added {symbol} to Position Exit Monitor")
                    
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
        
        # Start Position Exit Monitor if enabled
        if self._position_monitor:
            self._position_monitor_thread = threading.Thread(
                target=self._position_monitor.start_monitoring_loop,
                daemon=True,
                name="PositionExitMonitor"
            )
            self._position_monitor_thread.start()
            logger.info("🛡️ Position Exit Monitor thread started")
        
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
        
        # Stop position monitor if running
        if self._position_monitor:
            logger.info("⏸️ Stopping Position Exit Monitor...")
            self._position_monitor.stop()
            if self._position_monitor_thread:
                self._position_monitor_thread.join(timeout=10)
            logger.info("✅ Position Exit Monitor stopped")
        
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
                    
                    # BEST-PICK-ONLY MODE: If capital is limited, focus on highest confidence trade
                    if self._capital_manager and getattr(self.config, 'best_pick_only_mode', True):
                        available = self._capital_manager.get_available_capital()
                        utilization = self._capital_manager.get_utilization_pct()
                        
                        # Trigger best-pick mode if capital is very limited
                        if available < 200 or utilization > 85:
                            # Sort signals by confidence (highest first)
                            buy_signals = [s for s in signals if s.signal == 'BUY']
                            if buy_signals:
                                buy_signals.sort(key=lambda x: x.confidence, reverse=True)
                                best_signal = buy_signals[0]
                                
                                pass  # logger.info(f"💎 BEST-PICK-ONLY MODE: Limited capital (${}, {utilization:.1f}% utilized) {available:.2f}")
                                logger.info(f"   Focusing on highest confidence trade: {best_signal.symbol} ({best_signal.confidence:.1f}%)")
                                logger.info("   Skipping {} other opportunities to maximize best setup", str(len(buy_signals)-1))
                                
                                # Only process the best signal
                                signals = [best_signal] + [s for s in signals if s.signal == 'SELL']
                    
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
                logger.error("Error in auto-trader loop: {}", str(e), exc_info=True)
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
        import pytz
        
        # Get current time in proper ET timezone with automatic DST handling
        try:
            eastern = pytz.timezone('US/Eastern')
            now_et = datetime.now(eastern)
            
            # Log timezone info for debugging
            tz_name = now_et.strftime('%Z')  # EST or EDT
            pass  # logger.debug(f"🕐 Current time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z'} (Timezone: {tz_name})"))
        except Exception as e:
            # Fallback to manual calculation if pytz fails
            logger.warning(f"⚠️ pytz timezone conversion failed: {e}, using fallback")
            now_utc = datetime.now(timezone.utc)
            
            # Better DST detection: DST is second Sunday in March to first Sunday in November
            # For 2025: DST starts March 9, ends November 2
            # Simple approximation: UTC-4 from mid-March to early November
            month = now_utc.month
            day = now_utc.day
            
            # Approximate DST period (this is a simplification)
            if month < 3 or (month == 3 and day < 9) or (month == 11 and day >= 2) or month == 12:
                et_offset = timedelta(hours=-5)  # EST
            else:
                et_offset = timedelta(hours=-4)  # EDT
            
            now_et = now_utc + et_offset
            now_et = now_et.replace(tzinfo=None)  # Remove timezone info for comparison
        
        # Check if weekend
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Get time component (handle both timezone-aware and naive datetime)
        current_time_et = now_et.time() if hasattr(now_et, 'time') else now_et
        
        # Regular trading hours
        start_time = dt_time(self.config.trading_start_hour, self.config.trading_start_minute)
        end_time = dt_time(self.config.trading_end_hour, self.config.trading_end_minute)
        
        in_regular_hours = start_time <= current_time_et <= end_time
        
        # Pre-market hours (if enabled)
        in_premarket = False
        if hasattr(self.config, 'enable_premarket') and self.config.enable_premarket:
            premarket_start = dt_time(
                getattr(self.config, 'premarket_start_hour', 7),
                getattr(self.config, 'premarket_start_minute', 0)
            )
            premarket_end = start_time  # Pre-market ends when regular hours begin
            in_premarket = premarket_start <= current_time_et < premarket_end
        
        # After-hours (if enabled)
        in_afterhours = False
        if hasattr(self.config, 'enable_afterhours') and self.config.enable_afterhours:
            afterhours_start = end_time  # After-hours starts when regular hours end
            afterhours_end = dt_time(
                getattr(self.config, 'afterhours_end_hour', 20),
                getattr(self.config, 'afterhours_end_minute', 0)
            )
            in_afterhours = afterhours_start < current_time_et <= afterhours_end
        
        in_hours = in_regular_hours or in_premarket or in_afterhours
        
        if not in_hours:
            logger.info("❌ Outside trading hours: ET time is {}, market hours are {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}", str(current_time_et.strftime('%H:%M')))
        elif in_premarket:
            pass  # logger.info(f"🌅 Pre-market hours: {current_time_et.strftime('%H:%M'} ET"))
        elif in_afterhours:
            pass  # logger.info(f"🌙 After-hours: {current_time_et.strftime('%H:%M'} ET"))
        
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
                    logger.info("✅ Smart Scanner found {} optimal tickers: {', '.join(smart_tickers[:5])}...", str(len(smart_tickers)))
                    return smart_tickers
                else:
                    logger.warning("⚠️ Smart Scanner found no tickers, falling back to watchlist")
                    return self.watchlist
                
        except Exception as e:
            logger.error("Error in smart discovery: {}", str(e), exc_info=True)
            return self.watchlist
    
    def _scan_with_ml_enhanced(self) -> List:
        """
        Scan using ML-Enhanced Scanner (Triple Validation)
        Combines: 40% ML + 35% LLM + 25% Quantitative Analysis
        
        This is the 1st PUNCH of the 1-2 KNOCKOUT COMBO
        """
        try:
            logger.info("🧠 Starting ML-Enhanced Scanner (Triple Validation)...")
            
            if not self._ml_scanner:
                logger.error("ML-Enhanced Scanner not initialized, falling back to standard scan")
                return []
            
            # Determine scan type based on trading mode
            if self.config.trading_mode in ["SCALPING", "WARRIOR_SCALPING", "STOCKS", "ALL"]:
                # Scan for stock/options opportunities
                logger.info(f"📊 Scanning for {self.config.trading_mode} opportunities with ML-Enhanced Scanner...")
                
                # Get price filters from config if available
                min_price = getattr(self.config, 'min_stock_price', None)
                max_price = getattr(self.config, 'max_stock_price', None)
                
                if min_price or max_price:
                    logger.info(f"💰 Price Filter Active: ${min_price or 0:.2f} - ${max_price or 999999:.2f}")
                
                ml_trades = self._ml_scanner.scan_top_options_with_ml(
                    top_n=20,
                    min_ensemble_score=self.config.min_ensemble_score,
                    min_price=min_price,
                    max_price=max_price
                )
                
                logger.info(f"✅ ML-Enhanced Scanner found {len(ml_trades)} high-confidence opportunities")
                
                # Log top opportunities with detailed scores
                if ml_trades:
                    logger.info(f"📈 TOP OPPORTUNITIES (Ensemble ≥ {self.config.min_ensemble_score}):")
                    for i, trade in enumerate(ml_trades[:5], 1):
                        # logger.info(f"   #{i}. {trade.ticker}: Ensemble={trade.combined_score:.1f} (ML:{trade.ml_prediction_score:.1f} AI:{trade.ai_rating*10:.1f} Q:{trade.score:.1f}) Price=${trade.price:.2f}")
                        pass
                
                # Convert ML trades to TradingSignals
                from services.ai_trading_signals import TradingSignal
                signals = []
                
                for trade in ml_trades:
                    # Map ML trade to signal format
                    signal = TradingSignal(
                        symbol=trade.ticker,
                        signal='BUY',  # ML scanner focuses on long opportunities
                        confidence=trade.combined_score,  # Use ensemble score as confidence
                        entry_price=trade.price,
                        target_price=trade.price * 1.05,  # 5% target (adjust based on mode)
                        stop_loss=trade.price * 0.98,  # 2% stop
                        position_size=0,  # Will be calculated later
                        reasoning=f"ML-Enhanced: {trade.ai_reasoning} | ML Score: {trade.ml_prediction_score:.1f}, LLM: {trade.ai_rating*10:.1f}, Quant: {trade.score:.1f}",
                        risk_level=trade.risk_level,
                        time_horizon='DAY_TRADE' if self.config.trading_mode in ['SCALPING', 'WARRIOR_SCALPING'] else 'SWING',
                        technical_score=trade.score,
                        sentiment_score=0.0,
                        news_score=0.0,
                        social_score=0.0,
                        discord_score=0.0
                    )
                    signals.append(signal)
                    
                    logger.info(f"  🎯 {trade.ticker}: Combined Score {trade.combined_score:.1f}% "
                              f"(ML:{trade.ml_prediction_score:.1f} + LLM:{trade.ai_rating*10:.1f} + Quant:{trade.score:.1f})")
                
                return signals
            else:
                logger.warning(f"ML-Enhanced Scanner not configured for {self.config.trading_mode} mode")
                return []
                
        except Exception as e:
            logger.error("Error in ML-Enhanced Scanner: {}", str(e), exc_info=True)
            return []
    
    def _validate_trade_with_ai(self, signal) -> tuple[bool, str, float]:
        """
        AI Pre-Trade Validation (2nd PUNCH of 1-2 KNOCKOUT COMBO)
        
        Uses LLM to perform final risk assessment before execution.
        
        Args:
            signal: TradingSignal to validate
            
        Returns:
            (should_execute, reasoning, confidence_score)
        """
        try:
            if not self._ai_validator:
                # No validator, approve by default
                return True, "AI validation disabled", 1.0
            
            logger.info(f"🛡️ Running AI Pre-Trade Validation for {signal.symbol}...")
            
            # Get current portfolio status
            open_positions = self.state_manager.get_all_open_positions()
            available_capital = self._capital_manager.get_available_capital() if self._capital_manager else 0
            utilization = self._capital_manager.get_utilization_pct() if self._capital_manager else 0
            
            # Build validation prompt - use ACTUAL position size, not inflated default
            actual_position_size = signal.position_size if signal.position_size and signal.position_size > 0 else 1
            position_value = signal.entry_price * actual_position_size if signal.entry_price else 0
            risk_reward = ((signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss)) if (signal.entry_price and signal.stop_loss and signal.target_price) else 0
            
            prompt = f"""You are an expert risk manager. Perform a final validation check on this trade before execution.

TRADE DETAILS:
Symbol: {signal.symbol}
Signal: {signal.signal}
Entry Price: ${signal.entry_price:.2f}
Target Price: ${signal.target_price:.2f} ({((signal.target_price/signal.entry_price-1)*100):.1f}% gain)
Stop Loss: ${signal.stop_loss:.2f} ({((signal.stop_loss/signal.entry_price-1)*100):.1f}% loss)
Position Value: ${position_value:,.2f}
Risk/Reward Ratio: {risk_reward:.2f}:1
Confidence: {signal.confidence:.1f}%
Reasoning: {signal.reasoning}

PORTFOLIO STATUS:
Available Capital: ${available_capital:,.2f}
Utilization: {utilization:.1f}%
Open Positions: {len(open_positions)}
Current Holdings: {list(open_positions.keys())}

RISK ASSESSMENT CRITERIA:
1. Is the risk/reward ratio acceptable (minimum 1.5:1)?
2. Is position sizing appropriate for available capital?
3. Does this trade diversify or concentrate portfolio risk?
4. Are there any red flags in the reasoning or setup?
5. Is the confidence level justified?

RESPOND IN THIS EXACT FORMAT:
DECISION: APPROVE or REJECT
CONFIDENCE: [0.0-1.0]
REASONING: [Your brief 1-2 sentence risk assessment]

Be conservative. Only APPROVE trades with solid risk/reward and proper portfolio fit."""

            # Get AI response
            response = self._ai_validator._call_openrouter(prompt)
            
            if not response:
                logger.warning("⚠️ AI validation failed to return response, approving by default")
                return True, "AI validation timeout, proceeding with caution", 0.7
            
            # Parse response
            response_upper = response.upper()
            
            # Extract decision
            if "DECISION:" in response_upper and "APPROVE" in response_upper.split("DECISION:")[1].split("\n")[0]:
                decision = True
            elif "DECISION:" in response_upper and "REJECT" in response_upper.split("DECISION:")[1].split("\n")[0]:
                decision = False
            else:
                # Fallback parsing
                decision = "APPROVE" in response_upper and "REJECT" not in response_upper
            
            # Extract confidence
            import re
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response.split('\n')[-1]
            
            # Log validation result
            decision_icon = "✅" if decision else "🚫"
            logger.info(f"{decision_icon} AI Validation: {'APPROVED' if decision else 'REJECTED'} "
                       f"(confidence: {confidence:.2f}) - {reasoning[:100]}")
            
            return decision, reasoning, confidence
            
        except Exception as e:
            logger.error(f"Error in AI validation: {e}", exc_info=True)
            # On error, be cautious but don't block trade completely
            return True, f"AI validation error: {str(e)[:100]}", 0.6
    
    def _scan_for_signals(self) -> List:
        """Scan watchlist for trading signals"""
        try:
            # WARRIOR_SCALPING mode: Multi-stage pipeline
            if self.config.trading_mode == "WARRIOR_SCALPING":
                return self._scan_warrior_scalping_pipeline()
            
            # PUNCH 1: Use ML-Enhanced Scanner if enabled (Triple Validation)
            if self.config.use_ml_enhanced_scanner and self._ml_scanner:
                logger.info("🥊 PUNCH 1: ML-Enhanced Scanner (Triple Validation)")
                return self._scan_with_ml_enhanced()
            
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
                    num_buckets=self.config.cash_buckets if self.config.use_settled_funds_only else 1,  # Single bucket if not using settlement
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
                    logger.info("✅ Smart Scanner found {} optimal tickers: {', '.join(tickers_to_scan[:10])}{'...' if len(tickers_to_scan) > 10 else ''}", str(len(tickers_to_scan)))
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
                logger.info("📋 Using watchlist: {} tickers: {', '.join(self.watchlist[:10])}{'...' if len(self.watchlist) > 10 else ''}", str(len(self.watchlist)))
            
            # Filter tickers by price range first (for WARRIOR_SCALPING)
            min_price = getattr(self.config, 'MIN_PRICE', 2.0)
            max_price = getattr(self.config, 'MAX_PRICE', 20.0)
            
            # Filter watchlist by price range before scanning
            if min_price or max_price:
                logger.info(f"⚔️ Filtering watchlist by price range: ${min_price}-${max_price}")
                logger.info(f"   📋 Checking prices for {len(tickers_to_scan)} tickers...")
                try:
                    import yfinance as yf
                    
                    # Get current prices for watchlist tickers and filter
                    filtered_tickers = []
                    checked_count = 0
                    error_count = 0
                    
                    # Use yfinance for fast bulk price checks (much faster than IBKR)
                    logger.info(f"   🚀 Using yfinance for fast bulk price checks...")
                    
                    for ticker in tickers_to_scan:
                        try:
                            checked_count += 1
                            if checked_count % 5 == 0:  # Log progress every 5 tickers
                                logger.info(f"   🔍 Progress: {checked_count}/{len(tickers_to_scan)} tickers checked...")
                            
                            # Use yfinance for fast price check
                            try:
                                yf_ticker = yf.Ticker(ticker)
                                info = yf_ticker.fast_info
                                price = info.get('lastPrice', 0)
                                if not price:
                                    # Fallback to regular info
                                    info = yf_ticker.info
                                    price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                            except:
                                # Fallback to broker if yfinance fails
                                quote = self.tradier_client.get_quote(ticker)
                                if quote:
                                    price = float(quote.get('last', 0) or quote.get('bid', 0) or 0)
                                else:
                                    price = 0
                            
                            if price == 0:
                                # No valid price, skip this ticker
                                logger.debug(f"  ⚠️ {ticker}: No valid price available, skipping")
                                error_count += 1
                                continue
                                
                            if min_price <= price <= max_price:
                                filtered_tickers.append(ticker)
                                logger.info(f"  ✅ {ticker}: ${price:.2f} (IN RANGE)")
                            else:
                                logger.debug(f"  ❌ {ticker}: ${price:.2f} (outside ${min_price}-${max_price} range)")
                                
                        except Exception as e:
                            logger.debug(f"  ⚠️ Error checking price for {ticker}: {e}")
                            error_count += 1
                            # Skip tickers with errors (don't include)
                            continue
                    
                    logger.info(f"   ✅ Price check complete: {checked_count} checked, {len(filtered_tickers)} passed, {error_count} errors")
                    
                    if filtered_tickers:
                        tickers_to_scan = filtered_tickers
                        logger.info(f"✅ Filtered to {len(filtered_tickers)} tickers in price range ${min_price}-${max_price}: {', '.join(filtered_tickers)}")
                    else:
                        logger.warning(f"⚠️ No tickers in price range ${min_price}-${max_price}, will scan original watchlist")
                        # Don't replace tickers_to_scan if nothing passed filter
                except Exception as e:
                    logger.error("❌ Error filtering watchlist by price: {}, using original watchlist", str(e), exc_info=True)
            
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
            logger.error("Error scanning for Warrior Trading signals: {}", str(e), exc_info=True)
            return []
    
    def _scan_warrior_scalping_pipeline(self) -> List:
        """
        WARRIOR_SCALPING Multi-Stage Pipeline:
        1. Warrior Scalping Screener (Gap & Go detection)
        2. ML Enhanced Scanner (if enabled) - Triple validation
        3. AI Validation happens automatically in _execute_signal
        
        This implements the full 1-2 KNOCKOUT COMBO for Warrior Scalping:
        - PUNCH 1: Warrior Screener finds gap & go setups
        - PUNCH 2: ML Enhanced Scanner adds ML + LLM + Quant analysis
        - PUNCH 3: AI Pre-Trade Validation (final check before execution)
        """
        try:
            logger.info("⚔️ WARRIOR_SCALPING PIPELINE: Starting multi-stage analysis...")
            
            # STAGE 1: Warrior Scalping Screener
            logger.info("=" * 60)
            logger.info("⚔️ STAGE 1: Warrior Scalping Screener (Gap & Go Detection)")
            logger.info("=" * 60)
            warrior_signals = self._scan_warrior_trading_signals()
            
            if not warrior_signals:
                logger.info("⚠️ Warrior Screener found no setups, pipeline complete")
                return []
            
            logger.info(f"✅ Warrior Screener found {len(warrior_signals)} initial setups")
            for sig in warrior_signals:
                logger.info(f"  📊 {sig.symbol}: {sig.confidence:.1f}% confidence - {sig.reasoning[:100]}")
            
            # STAGE 2: ML Enhanced Scanner (if enabled)
            if self.config.use_ml_enhanced_scanner and self._ml_scanner:
                logger.info("=" * 60)
                logger.info("🧠 STAGE 2: ML-Enhanced Scanner (Triple Validation)")
                logger.info("=" * 60)
                logger.info("Applying ML + LLM + Quantitative analysis to Warrior setups...")
                
                enhanced_signals = []
                for warrior_signal in warrior_signals:
                    try:
                        # Get ticker symbol
                        ticker = warrior_signal.symbol
                        
                        # Enhance this specific ticker with ML+LLM+Quant analysis
                        # We need to analyze the ticker directly rather than scanning the whole universe
                        ml_enhanced_signal = self._enhance_warrior_signal_with_ml(warrior_signal, ticker)
                        
                        if ml_enhanced_signal:
                            enhanced_signals.append(ml_enhanced_signal)
                            logger.info(f"  ✅ {ticker}: Passed ML-Enhanced filter (Combined: {ml_enhanced_signal.confidence:.1f}%)")
                        else:
                            # ML enhancement failed or below threshold
                            logger.info(f"  ⚠️ {ticker}: Failed ML-Enhanced filter, excluding from results")
                            continue
                            
                    except Exception as e:
                        logger.warning(f"  ⚠️ Error enhancing {warrior_signal.symbol} with ML: {e}")
                        # If ML enhancement fails, exclude the signal (fail-safe mode)
                        continue
                
                # Filter by ML ensemble score threshold
                before_count = len(enhanced_signals)
                enhanced_signals = [
                    sig for sig in enhanced_signals 
                    if sig.confidence >= self.config.min_ensemble_score
                ]
                
                if len(enhanced_signals) < before_count:
                    logger.info(f"  📉 Filtered {before_count - len(enhanced_signals)} signals below ML ensemble threshold ({self.config.min_ensemble_score}%)")
                
                logger.info(f"✅ ML-Enhanced Scanner: {len(enhanced_signals)} signals passed triple validation")
                
                # STAGE 3 info (AI Validation happens in _execute_signal)
                logger.info("=" * 60)
                logger.info("🛡️ STAGE 3: AI Pre-Trade Validation will run before execution")
                logger.info("=" * 60)
                logger.info(f"🥊 WARRIOR SCALPING PIPELINE COMPLETE: {len(enhanced_signals)} high-quality setups ready")
                
                return enhanced_signals
            else:
                # ML Enhanced Scanner not enabled, return original Warrior signals
                logger.info("⚠️ ML-Enhanced Scanner not enabled, using Warrior Screener results only")
                logger.info("🛡️ AI Pre-Trade Validation will still run before execution")
                return warrior_signals
            
        except Exception as e:
            logger.error("Error in Warrior Scalping pipeline: {}", str(e), exc_info=True)
            # Fallback to basic warrior scan
            return self._scan_warrior_trading_signals()
    
    def _enhance_warrior_signal_with_ml(self, warrior_signal, ticker: str):
        """
        Enhance a Warrior Trading signal with ML+LLM+Quant analysis
        
        Args:
            warrior_signal: TradingSignal from Warrior screener
            ticker: Ticker symbol
            
        Returns:
            Enhanced TradingSignal with ML scores, or None if below threshold
        """
        try:
            from services.top_trades_scanner import TopTrade
            from services.ai_confidence_scanner import AIConfidenceTrade
            import yfinance as yf
            
            # Get current market data for the ticker
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current_price = warrior_signal.entry_price or (info.get('regularMarketPrice') or info.get('currentPrice') or 0)
            if not current_price:
                logger.warning(f"  ⚠️ {ticker}: Could not get price data")
                return None
            
            # Get historical data for volume calculation
            hist = stock.history(period="5d", interval="1d")
            if not hist.empty:
                current_volume = info.get('regularMarketVolume') or info.get('volume') or hist['Volume'].iloc[-1] if len(hist) > 0 else 0
                avg_volume = info.get('averageVolume') or info.get('averageVolume10days') or hist['Volume'].mean() if len(hist) > 0 else 0
                volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0
                
                # Calculate price change
                if len(hist) > 1:
                    prev_close = hist['Close'].iloc[-2]
                    change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                else:
                    change_pct = 0
            else:
                volume_ratio = 1.0
                change_pct = 0
            
            # Create a TopTrade-like object for ML scanner
            from dataclasses import dataclass
            
            @dataclass
            class SimpleTopTrade:
                ticker: str
                score: float
                price: float
                change_pct: float
                volume: int
                volume_ratio: float
                reason: str
                trade_type: str
                confidence: str
                risk_level: str
            
            top_trade = SimpleTopTrade(
                ticker=ticker,
                score=warrior_signal.confidence,  # Use warrior confidence as base
                price=current_price,
                change_pct=change_pct,
                volume=int(info.get('regularMarketVolume', 0) or info.get('volume', 0)),
                volume_ratio=volume_ratio,
                reason=warrior_signal.reasoning or "Warrior Scalping setup",
                trade_type='options',
                confidence='HIGH' if warrior_signal.confidence >= 70 else 'MEDIUM',
                risk_level='M'
            )
            
            # Get AI confidence analysis (LLM + Quant)
            ai_trade = self._ml_scanner.ai_scanner._generate_ai_confidence(top_trade, 'options')
            if not ai_trade:
                logger.warning(f"  ⚠️ {ticker}: AI analysis failed")
                return None
            
            # Enhance with ML (alpha factors)
            ml_enhanced = self._ml_scanner._enhance_with_ml(
                AIConfidenceTrade(
                    ticker=ticker,
                    score=top_trade.score,
                    price=current_price,
                    change_pct=change_pct,
                    volume=top_trade.volume,
                    volume_ratio=volume_ratio,
                    reason=top_trade.reason,
                    trade_type='options',
                    confidence=top_trade.confidence,
                    risk_level=top_trade.risk_level,
                    ai_confidence=ai_trade.get('ai_confidence', 'MEDIUM'),
                    ai_reasoning=ai_trade.get('ai_reasoning', ''),
                    ai_risks=ai_trade.get('ai_risks', ''),
                    ai_rating=ai_trade.get('ai_rating', 5.0)
                ),
                'options'
            )
            
            # Check if combined score meets threshold
            if ml_enhanced.combined_score < self.config.min_ensemble_score:
                logger.debug(f"  ✗ {ticker}: Combined score {ml_enhanced.combined_score:.1f} < {self.config.min_ensemble_score}")
                return None
            
            # Log the enhancement
            logger.info(f"  🎯 {ticker}: ML Enhanced - "
                      f"Combined: {ml_enhanced.combined_score:.1f}% "
                      f"(ML:{ml_enhanced.ml_prediction_score:.1f} + "
                      f"LLM:{ml_enhanced.ai_rating*10:.1f} + "
                      f"Quant:{ml_enhanced.score:.1f})")
            
            # Update warrior signal with ML scores
            warrior_signal.confidence = ml_enhanced.combined_score
            
            # Update reasoning with ML analysis
            original_reasoning = warrior_signal.reasoning
            ml_reasoning = ml_enhanced.ai_reasoning[:200] if ml_enhanced.ai_reasoning else ""
            warrior_signal.reasoning = (
                f"Warrior + ML-Enhanced: {original_reasoning} | "
                f"ML Analysis: {ml_reasoning} | "
                f"ML Score: {ml_enhanced.ml_prediction_score:.1f}, "
                f"LLM: {ml_enhanced.ai_rating*10:.1f}, "
                f"Quant: {ml_enhanced.score:.1f}"
            )
            
            return warrior_signal
            
        except Exception as e:
            logger.error("Error enhancing {ticker} with ML: {}", str(e), exc_info=True)
            return None
    
    def _calculate_position_size(self, signal, available_capital=None):
        """
        Calculate position size for a signal.
        Returns: (shares, success) tuple
        """
        try:
            # Check prerequisites
            if not hasattr(signal, 'entry_price') or not signal.entry_price:
                logger.warning(f"⚠️ Signal for {signal.symbol} missing entry_price")
                return 0, False
            if not hasattr(signal, 'stop_loss') or not signal.stop_loss:
                logger.warning(f"⚠️ Signal for {signal.symbol} missing stop_loss")
                return 0, False
            if not self._cash_manager:
                logger.warning(f"⚠️ Cash manager not initialized")
                return 0, False
            
            bal_success, bal_data = self.tradier_client.get_account_balance()
            if not bal_success:
                raise APIError("Failed to retrieve account balance")
            if not isinstance(bal_data, dict):
                raise DataProcessingError("Invalid account balance data")
            
            settled_cash = None
            total_equity = 10000.0
            b = bal_data.get('balances', {})
            raw_cash = float(b.get('cash_available', b.get('total_cash', 10000.0)))
            total_equity = float(b.get('total_equity', raw_cash or 10000.0))
            logger.info(f"💵 Account balance: raw_cash=${raw_cash:.2f}, total_equity=${total_equity:.2f}")
            
            # Use actual broker cash for settlement tracking, but respect config.total_capital limit
            settled_cash = self._cash_manager.get_settled_cash(raw_cash)
            logger.info(f"💵 Settled cash after T+{self.config.t_plus_settlement_days}: ${settled_cash:.2f}")
            
            # Limit to configured total capital (e.g., only use $800 of $10k account)
            if settled_cash > self.config.total_capital:
                logger.info(f"💵 Limiting to configured capital: ${self.config.total_capital:.2f} (broker has ${settled_cash:.2f})")
                settled_cash = self.config.total_capital

            bucket_idx = self._cash_manager.select_active_bucket()
            bucket_cash = self._cash_manager.bucket_target_cash(settled_cash, bucket_idx)
            logger.info(f"💵 Bucket #{bucket_idx} cash: ${bucket_cash:.2f}")

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
            logger.info(f"💵 Calling clamp_to_settled_cash: shares={shares_by_risk}, entry=${signal.entry_price:.2f}, bucket_cash=${bucket_cash:.2f}, reserve={self.config.reserve_cash_pct}%")
            affordable = self._cash_manager.clamp_to_settled_cash(
                shares=shares_by_risk,
                entry_price=signal.entry_price,
                settled_cash=bucket_cash,
                reserve_pct=self.config.reserve_cash_pct / 100.0,  # Convert percentage to decimal
            )
            logger.info(f"💵 clamp_to_settled_cash returned: {affordable} shares")
            
            # Use capital manager max position size
            max_position_dollars = self._capital_manager.get_max_position_size() if self._capital_manager else (total_equity * (self.config.max_position_size_pct / 100.0))
            max_by_pct = int(max_position_dollars // signal.entry_price)
            
            # Also check available capital
            if self._capital_manager:
                if available_capital is None:
                    available_capital = self._capital_manager.get_available_capital()
                max_by_available = int(available_capital // signal.entry_price)
                max_by_pct = min(max_by_pct, max_by_available)
                logger.info(f"📊 Position sizing: risk-based={shares_by_risk}, affordable={affordable}, max_pct={max_by_pct}, avail_cap={max_by_available}")
            
            final_shares = max(0, min(affordable, max_by_pct))
            logger.info(f"💰 Final position sizing: {final_shares} shares (${final_shares * signal.entry_price:.2f})")
            
            # AI Capital Advisor Override
            if self._capital_advisor and final_shares > 0:
                try:
                    # Get current portfolio state
                    open_positions = len(self.state_manager.get_all_open_positions())
                    max_positions = getattr(self.config, 'max_concurrent_positions', 10)
                    
                    # Get AI recommendation
                    recommendation = self._capital_advisor.recommend_position_size(
                        ticker=signal.symbol,
                        price=signal.entry_price,
                        signal_confidence=signal.confidence,
                        risk_level=signal.risk_level if hasattr(signal, 'risk_level') else 'M',
                        available_capital=available_capital if available_capital else bucket_cash,
                        total_capital=self.config.total_capital,
                        current_positions=open_positions,
                        max_positions=max_positions,
                        ensemble_score=getattr(signal, 'technical_score', signal.confidence),
                        ai_reasoning=signal.reasoning if hasattr(signal, 'reasoning') else None,
                        use_ai_reasoning=True
                    )
                    
                    if recommendation.is_approved and recommendation.recommended_shares > 0:
                        logger.info(f"🧠 AI Capital Advisor: {recommendation.recommended_shares} shares (${recommendation.recommended_position_value:.2f}, {recommendation.recommended_position_size_pct:.1f}%)")
                        logger.info(f"   Reasoning: {recommendation.reasoning}")
                        
                        # Use AI recommendation if it's more conservative
                        final_shares = min(final_shares, recommendation.recommended_shares)
                    else:
                        logger.warning(f"⚠️ AI Capital Advisor: Position not approved")
                        for warning in recommendation.warnings:
                            logger.warning(f"   - {warning}")
                        if not recommendation.is_approved:
                            logger.info("AI Capital Advisor rejected this position size; skipping trade")
                            return 0, False
                except Exception as e:
                    logger.error(f"Error getting AI capital recommendation: {e}")
                    # Continue with traditional sizing on error
            
            # Fractional Share Support (Override whole share sizing if fractional is appropriate)
            if self._fractional_manager and self._fractional_manager.should_use_fractional(signal.symbol, signal.entry_price):
                try:
                    # Calculate fractional quantity based on available capital
                    fractional_qty, actual_cost = self._fractional_manager.calculate_fractional_quantity(
                        symbol=signal.symbol,
                        price=signal.entry_price,
                        available_capital=available_capital if available_capital else bucket_cash,
                        target_dollar_amount=None  # Uses custom amount or default
                    )
                    
                    if fractional_qty > 0:
                        logger.info(f"📊 Using fractional shares: {fractional_qty} shares @ ${signal.entry_price:.2f} = ${actual_cost:.2f}")
                        logger.info(f"   (Traditional would have been {final_shares} shares = ${final_shares * signal.entry_price:.2f})")
                        return fractional_qty, True
                    else:
                        logger.warning(f"⚠️ Fractional sizing returned 0 shares, falling back to whole shares")
                except Exception as e:
                    logger.error(f"Error calculating fractional quantity: {e}, falling back to whole shares")
                    # Fall through to return whole shares
            
            return final_shares, True
            
        except APIError as e:
            logger.error(f"API Error sizing position: {e}")
            return 0, False
        except DataProcessingError as e:
            logger.error(f"Error processing account balance data: {e}")
            return 0, False
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0, False
    
    def _check_ai_entry_timing(self, signal):
        """Check with AI if now is good time to enter this stock"""
        try:
            # Lazy initialize assistant if not already done
            if not self._stock_entry_assistant:
                logger.warning("Stock Entry Assistant not initialized")
                # Return default ENTER_NOW to not block trades
                from services.ai_stock_entry_assistant import EntryAnalysis
                return EntryAnalysis(
                    symbol=signal.symbol,
                    action="ENTER_NOW",
                    confidence=60.0,
                    reasoning="Entry assistant not initialized",
                    urgency="LOW",
                    current_price=signal.entry_price
                )
            
            # Calculate position size in USD
            position_size_usd = signal.position_size * signal.entry_price
            
            # Calculate risk/reward percentages
            risk_pct = abs((signal.entry_price - signal.stop_loss) / signal.entry_price) * 100 if signal.stop_loss else 2.0
            tp_pct = abs((signal.target_price - signal.entry_price) / signal.entry_price) * 100 if signal.target_price else 5.0
            
            # Get AI analysis
            entry_analysis = self._stock_entry_assistant.analyze_entry(
                symbol=signal.symbol,
                side=signal.signal,
                position_size=position_size_usd,
                risk_pct=risk_pct,
                take_profit_pct=tp_pct
            )
            
            return entry_analysis
            
        except Exception as e:
            logger.error("Error checking AI entry timing: {}", str(e), exc_info=True)
            # On error, default to ENTER_NOW (don't block trades)
            from services.ai_stock_entry_assistant import EntryAnalysis
            return EntryAnalysis(
                symbol=signal.symbol,
                action="ENTER_NOW",
                confidence=60.0,
                reasoning=f"AI check failed: {e}",
                urgency="LOW",
                current_price=signal.entry_price
            )
    
    def _execute_signal(self, signal):
        """Execute a trading signal"""
        try:
            logger.info(f"🎯 Executing signal: {signal.symbol} {signal.signal} (confidence: {signal.confidence}%)")
            
            # ==========================================
            # 🆕 AI ENTRY TIMING CHECK
            # ==========================================
            if self.config.use_ai_entry_timing:
                entry_analysis = self._check_ai_entry_timing(signal)
                
                # DO NOT ENTER - Skip this trade
                if entry_analysis.action == "DO_NOT_ENTER":
                    logger.warning(f"❌ AI Entry: DO NOT ENTER {signal.symbol} (Confidence: {entry_analysis.confidence:.1f}%)")
                    logger.warning(f"   Reasoning: {entry_analysis.reasoning}")
                    return  # Skip trade
                
                # WAIT - Add to monitoring
                elif entry_analysis.action in ["WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT"]:
                    logger.info(f"⏳ AI Entry: WAIT for {signal.symbol} (Confidence: {entry_analysis.confidence:.1f}%)")
                    logger.info(f"   Reasoning: {entry_analysis.reasoning}")
                    
                    # Add to entry monitoring
                    if self._stock_entry_assistant:
                        # Calculate risk/reward
                        risk_pct = abs((signal.entry_price - signal.stop_loss) / signal.entry_price) * 100 if signal.stop_loss else 2.0
                        tp_pct = abs((signal.target_price - signal.entry_price) / signal.entry_price) * 100 if signal.target_price else 5.0
                        position_size_usd = signal.position_size * signal.entry_price
                        
                        self._stock_entry_assistant.monitor_entry_opportunity(
                            symbol=signal.symbol,
                            side=signal.signal,
                            position_size=position_size_usd,
                            risk_pct=risk_pct,
                            take_profit_pct=tp_pct,
                            analysis=entry_analysis,
                            auto_execute=self.config.enable_auto_entry
                        )
                        logger.info(f"🔔 Added {signal.symbol} to entry monitoring")
                    return  # Don't execute now, wait
                
                # ENTER NOW - Continue with execution
                elif entry_analysis.action == "ENTER_NOW":
                    logger.info(f"✅ AI Entry: ENTER NOW {signal.symbol} (Confidence: {entry_analysis.confidence:.1f}%)")
                    logger.info(f"   Reasoning: {entry_analysis.reasoning}")
                    
                    # Optional: Update signal with AI suggestions
                    if entry_analysis.suggested_stop:
                        signal.stop_loss = entry_analysis.suggested_stop
                        logger.debug(f"   Updated stop loss to AI suggestion: ${signal.stop_loss:.2f}")
                    if entry_analysis.suggested_target:
                        signal.target_price = entry_analysis.suggested_target
                        logger.debug(f"   Updated target to AI suggestion: ${signal.target_price:.2f}")
            
            # ==========================================
            # EXISTING EXECUTION CODE CONTINUES BELOW
            # ==========================================
            
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
                    
                    # CHECK CAPITAL AVAILABILITY
                    available_capital = None
                    if self._capital_manager:
                        available_capital = self._capital_manager.get_available_capital()
                        utilization = self._capital_manager.get_utilization_pct()
                        
                        if utilization >= self.config.max_capital_utilization_pct:
                            logger.warning(f"⚠️ Capital utilization too high ({utilization:.1f}% >= {self.config.max_capital_utilization_pct}%), skipping {signal.symbol}")
                            return
                        
                        if available_capital <= 0:
                            logger.warning(f"⚠️ No capital available (${available_capital:.2f}), skipping {signal.symbol}")
                            return
                        
                        logger.info(f"💰 Capital check: ${available_capital:,.2f} available ({utilization:.1f}% utilization)")
                    
                    # CALCULATE POSITION SIZE BEFORE AI VALIDATION
                    # This ensures AI validator sees the correct, affordable position size
                    calculated_shares, sizing_success = self._calculate_position_size(signal, available_capital)
                    if not sizing_success or calculated_shares <= 0:
                        logger.warning(f"⚠️ Position sizing failed or returned 0 shares, skipping {signal.symbol}")
                        return
                    
                    # Update signal with calculated position size
                    signal.position_size = calculated_shares
                    logger.info(f"✅ Position size calculated: {calculated_shares} shares (${calculated_shares * signal.entry_price:.2f})")
            
            # PUNCH 2: AI Pre-Trade Validation (2nd knockout check)
            if self.config.use_ai_validation and self._ai_validator:
                logger.info("🥊 PUNCH 2: AI Pre-Trade Validation (Final Risk Check)")
                
                should_execute, validation_reasoning, validation_confidence = self._validate_trade_with_ai(signal)
                
                if not should_execute:
                    logger.warning(f"🚫 AI VALIDATION REJECTED trade for {signal.symbol}")
                    logger.warning(f"   Reason: {validation_reasoning}")
                    return  # Block the trade
                elif validation_confidence < self.config.min_ai_validation_confidence:
                    logger.warning(f"⚠️ AI validation confidence too low ({validation_confidence:.2f} < {self.config.min_ai_validation_confidence}), skipping {signal.symbol}")
                    logger.warning(f"   Reason: {validation_reasoning}")
                    return  # Block the trade
                else:
                    logger.info(f"✅ AI VALIDATION APPROVED trade for {signal.symbol} (confidence: {validation_confidence:.2f})")
                    logger.info(f"   Reason: {validation_reasoning}")
            
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
            
            # Position sizing for closing orders (use existing position quantity)
            if signal.signal == 'SELL' and has_position:
                # Closing a LONG position - use existing position quantity
                signal.position_size = position_quantity
                logger.info(f"📊 Closing {position_quantity} shares of {signal.symbol}")
            elif signal.signal == 'BUY' and has_short_position:
                # Covering a SHORT position - use existing short quantity
                signal.position_size = short_quantity
                logger.info(f"📊 Covering {short_quantity} shares SHORT of {signal.symbol}")
            # For new BUY positions, position size was already calculated before AI validation
            # No need to recalculate here

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
                    stop_loss_price=signal.stop_loss,
                    take_profit_price=signal.target_price,
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
                        
                        # LOG EXIT TO UNIFIED JOURNAL
                        try:
                            from services.unified_trade_journal import get_unified_journal
                            journal = get_unified_journal()
                            
                            # Find the trade_id from recent trades
                            recent_trades = journal.get_trades(
                                symbol=signal.symbol,
                                status="OPEN",
                                limit=10
                            )
                            
                            if recent_trades:
                                # Update the most recent open trade for this symbol
                                trade = recent_trades[0]
                                journal.update_trade_exit(
                                    trade_id=trade.trade_id,
                                    exit_price=signal.entry_price,
                                    exit_time=datetime.now(),
                                    exit_reason=f"Auto-trader signal (confidence: {signal.confidence}%)"
                                )
                                logger.info(f"📝 Updated journal with trade exit: {signal.symbol}")
                        except Exception as journal_err:
                            logger.debug(f"Could not update journal with exit: {journal_err}")
                        
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
                            pass  # logger.info("📊 Capital status: ${} available ({self._capital_manager.get_utilization_pct():.1f}% utilization)", str(self._capital_manager.get_available_capital():,.2f))
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
                        
                        # LOG TO UNIFIED JOURNAL
                        try:
                            from services.unified_trade_journal import get_unified_journal, UnifiedTradeEntry, TradeType
                            journal = get_unified_journal()
                            
                            # Calculate risk/reward percentages
                            if signal.signal == 'BUY':
                                risk_pct = ((signal.entry_price - signal.stop_loss) / signal.entry_price) * 100 if signal.stop_loss else 2.0
                                reward_pct = ((signal.target_price - signal.entry_price) / signal.entry_price) * 100 if signal.target_price else 5.0
                            else:
                                risk_pct = ((signal.stop_loss - signal.entry_price) / signal.entry_price) * 100 if signal.stop_loss else 2.0
                                reward_pct = ((signal.entry_price - signal.target_price) / signal.entry_price) * 100 if signal.target_price else 5.0
                            
                            rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                            
                            trade_entry = UnifiedTradeEntry(
                                trade_id=f"{signal.symbol}_{order_id}_{int(datetime.now().timestamp())}",
                                trade_type=TradeType.STOCK.value,
                                symbol=signal.symbol,
                                side=signal.signal,
                                entry_time=datetime.now(),
                                entry_price=signal.entry_price,
                                quantity=signal.position_size,
                                position_size_usd=signal.entry_price * signal.position_size,
                                stop_loss=signal.stop_loss if signal.stop_loss else 0,
                                take_profit=signal.target_price if signal.target_price else 0,
                                risk_pct=risk_pct,
                                reward_pct=reward_pct,
                                risk_reward_ratio=rr_ratio,
                                strategy=self.config.trading_mode,
                                setup_type=getattr(signal, 'setup_type', None),
                                ai_managed=False,  # Auto-trader but not AI position manager
                                broker="TRADIER" if hasattr(self.tradier_client, '__class__') and "Tradier" in self.tradier_client.__class__.__name__ else "IBKR",
                                order_id=str(order_id) if order_id else None,
                                status="OPEN"
                            )
                            
                            journal.log_trade_entry(trade_entry)
                            logger.info(f"📝 Logged stock trade to unified journal: {signal.symbol}")
                        except Exception as journal_err:
                            logger.debug(f"Could not log to unified journal: {journal_err}")
                        
                        # ADD TO POSITION MONITOR (CRITICAL)
                        if self._position_monitor and signal.entry_price and signal.stop_loss and signal.target_price:
                            self._position_monitor.add_position(
                                symbol=signal.symbol,
                                side=signal.signal,
                                quantity=signal.position_size,
                                entry_price=signal.entry_price,
                                entry_time=datetime.now(),
                                stop_loss=signal.stop_loss,
                                take_profit=signal.target_price,
                                trailing_stop_pct=2.0 if self.config.enable_trailing_stops else None,  # 2% trailing
                                max_hold_minutes=self.config.max_position_hold_minutes,
                                bracket_order_ids=bracket_ids
                            )
                            logger.info(f"🛡️ Added {signal.symbol} to Position Exit Monitor")
                        
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
                                pass  # logger.info("📊 Capital status: ${} available ({self._capital_manager.get_utilization_pct():.1f}% utilization)", str(self._capital_manager.get_available_capital():,.2f))
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
            logger.error("Error executing signal for {signal.symbol}: {}", str(e), exc_info=True)
    
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
        status = {
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
                'test_mode': self.config.test_mode,
                'position_monitoring_enabled': self.config.enable_position_monitoring
            }
        }
        
        # Add position monitor status if enabled
        if self._position_monitor:
            status['position_monitor'] = self._position_monitor.get_status()
        
        return status
    
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
                    logger.error("Agent system error: {}", str(e), exc_info=True)
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
            logger.error("Failed to start agent system: {}", str(e), exc_info=True)
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
    broker_client,
    signal_generator,
    watchlist: List[str],
    config: Optional[AutoTraderConfig] = None,
    use_smart_scanner: bool = False
) -> AutoTrader:
    """
    Create and configure auto-trader
    
    Args:
        broker_client: Broker client instance (TradierClient, IBKRClient, or BrokerAdapter)
        signal_generator: AITradingSignalGenerator instance
        watchlist: List of tickers to monitor
        config: Optional AutoTraderConfig (uses defaults if None)
        use_smart_scanner: Use Advanced Scanner to find optimal tickers
        
    Returns:
        AutoTrader instance
    """
    if config is None:
        config = AutoTraderConfig()
    
    return AutoTrader(config, broker_client, signal_generator, watchlist, use_smart_scanner)
