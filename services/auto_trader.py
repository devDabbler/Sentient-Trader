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

logger = logging.getLogger(__name__)


@dataclass
class AutoTraderConfig:
    """Configuration for auto-trader"""
    enabled: bool = False
    scan_interval_minutes: int = 15  # How often to scan for signals
    min_confidence: float = 75.0  # Minimum confidence to auto-execute
    max_daily_orders: int = 10
    max_position_size_pct: float = 20.0  # Max % of account per trade
    trading_start_hour: int = 9  # 9:30 AM ET
    trading_start_minute: int = 30
    trading_end_hour: int = 15  # 3:30 PM ET (close before market close)
    trading_end_minute: int = 30
    use_bracket_orders: bool = True  # Use stop-loss/take-profit
    risk_tolerance: str = "MEDIUM"
    paper_trading: bool = True  # Safety: start with paper trading
    trading_mode: str = "STOCKS"  # STOCKS, OPTIONS, SCALPING, SLOW_SCALPER, MICRO_SWING, ALL
    scalping_take_profit_pct: float = 2.0  # For scalping mode
    scalping_stop_loss_pct: float = 1.0  # For scalping mode
    # PDT-safe cash/risk controls
    use_settled_funds_only: bool = True
    cash_buckets: int = 3
    t_plus_settlement_days: int = 2
    risk_per_trade_pct: float = 0.02
    max_daily_loss_pct: float = 0.04
    max_consecutive_losses: int = 2
    reserve_cash_pct: float = 0.05
    # Multi-agent mode (new architecture for SLOW_SCALPER/MICRO_SWING)
    use_agent_system: bool = False  # Enable multi-agent architecture
    # Short selling support (paper trading only)
    # WARNING: Requires margin account in real trading. Only enable if you understand the risks.
    allow_short_selling: bool = False  # Enable short selling in paper trading mode (DISABLED by default)


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
        
        # Short position tracking (for paper trading)
        # Format: {symbol: {'quantity': int, 'entry_price': float, 'entry_time': datetime}}
        self._short_positions: Dict[str, Dict] = {}
        
        # Multi-agent orchestrator (for SLOW_SCALPER/MICRO_SWING modes)
        self._orchestrator = None
        self._agent_loop = None
        self._agent_thread = None
        
        logger.info(f"AutoTrader initialized with {len(watchlist)} tickers, smart_scanner={use_smart_scanner}, agent_system={config.use_agent_system}")
    
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
                signals = self._scan_for_signals()
                
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
        """Use Advanced Scanner to find optimal tickers for current trading mode"""
        try:
            from services.advanced_opportunity_scanner import AdvancedOpportunityScanner, ScanType
            
            # Define focused ticker lists for each strategy (faster scanning)
            strategy_universes = {
                "SCALPING": [
                    # High volume, liquid stocks perfect for scalping
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
                    'PLTR', 'SOFI', 'RIVN', 'PLUG', 'NOK', 'AMC', 'GME', 'MARA',
                    'RIOT', 'COIN', 'HOOD', 'SNAP', 'UBER', 'LYFT', 'NIO', 'LCID'
                ],
                "STOCKS": [
                    # Solid stocks for swing trading
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
                    'NFLX', 'DIS', 'PLTR', 'SOFI', 'COIN', 'RBLX', 'ABNB', 'DASH',
                    'SHOP', 'SNOW', 'CRWD', 'ZS', 'DDOG', 'NIO', 'RIVN', 'PLUG',
                    'MRNA', 'BNTX', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'XOM', 'CVX'
                ],
                "OPTIONS": [
                    # High IV stocks for options
                    'TSLA', 'AMD', 'NVDA', 'PLTR', 'SOFI', 'RIVN', 'LCID', 'NIO',
                    'PLUG', 'GME', 'AMC', 'COIN', 'HOOD', 'SNAP', 'RBLX', 'DASH',
                    'MRNA', 'BNTX', 'MARA', 'RIOT', 'TLRY', 'SNDL', 'ACB', 'CGC'
                ],
                "ALL": [
                    # Balanced mix
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
                    'PLTR', 'SOFI', 'COIN', 'RIVN', 'PLUG', 'NIO', 'LCID', 'GME',
                    'AMC', 'MARA', 'RIOT', 'MRNA', 'BNTX', 'SNAP', 'RBLX', 'DASH'
                ]
            }
            
            # Get focused universe for this strategy
            custom_universe = strategy_universes.get(self.config.trading_mode, strategy_universes["ALL"])
            
            # Map trading mode to scan type and trading style
            scan_config = {
                "SCALPING": {"scan_type": ScanType.MOMENTUM, "trading_style": "SCALP", "top_n": 10},
                "STOCKS": {"scan_type": ScanType.ALL, "trading_style": "SWING_TRADE", "top_n": 15},
                "OPTIONS": {"scan_type": ScanType.OPTIONS, "trading_style": "OPTIONS", "top_n": 15},
                "ALL": {"scan_type": ScanType.ALL, "trading_style": "SWING_TRADE", "top_n": 20}
            }
            
            config = scan_config.get(self.config.trading_mode, scan_config["ALL"])
            
            logger.info(f"🔍 Smart Scanner: Scanning {len(custom_universe)} curated tickers for {self.config.trading_mode} strategy...")
            
            scanner = AdvancedOpportunityScanner(use_ai=False)  # Quick scan without AI for speed
            opportunities = scanner.scan_opportunities(
                scan_type=config["scan_type"],
                trading_style=config["trading_style"],
                top_n=config["top_n"],
                custom_tickers=custom_universe,  # Use curated list instead of extended universe
                use_extended_universe=False
            )
            
            # Extract ticker symbols
            smart_tickers = [opp.ticker for opp in opportunities if opp.score >= 60]
            
            if smart_tickers:
                logger.info(f"✅ Smart Scanner found {len(smart_tickers)} optimal tickers: {', '.join(smart_tickers[:5])}...")
                return smart_tickers
            else:
                logger.warning("⚠️ Smart Scanner found no tickers, falling back to watchlist")
                return self.watchlist
                
        except Exception as e:
            logger.error(f"Error in smart scanner: {e}")
            return self.watchlist
    
    def _scan_for_signals(self) -> List:
        """Scan watchlist for trading signals"""
        try:
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
                if success and isinstance(bal_data, dict):
                    b = bal_data.get('balances', {})
                    settled_cash = float(b.get('cash_available', b.get('total_cash', 10000.0)))
                    account_balance = float(b.get('total_equity', settled_cash or 10000.0))
            except Exception as e:
                logger.error(f"Error getting account balance: {e}")
            
            # Get current positions to inform AI
            try:
                success, positions = self.tradier_client.get_positions()
                if success and positions:
                    current_positions = [pos.get('symbol') for pos in positions if pos.get('symbol')]
                    logger.info(f"📊 Current positions: {current_positions}")
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
    
    def _execute_signal(self, signal):
        """Execute a trading signal"""
        try:
            logger.info(f"🎯 Executing signal: {signal.symbol} {signal.signal} (confidence: {signal.confidence}%)")
            
            # Check current positions with retry handling
            success, positions = self.tradier_client.get_positions()
            has_position = False
            position_quantity = 0
            
            if not success:
                logger.error(f"⚠️ Failed to retrieve positions for {signal.symbol} - API may be experiencing issues")
                logger.warning(f"⚠️ Skipping {signal.signal} order for {signal.symbol} due to position check failure (safety measure)")
                return
            
            # Parse positions if successful
            if success and positions:
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
                elif has_position:
                    # Already long - skip adding to position
                    logger.info(f"Already have LONG position in {signal.symbol} ({position_quantity} shares), skipping additional BUY")
                    return
                else:
                    # Opening a long position
                    logger.info(f"✅ BUY signal validated - opening LONG position in {signal.symbol}")
            
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
                            # Closing a LONG position - target is higher, stop is lower
                            signal.target_price = signal.entry_price * (1 + self.config.scalping_take_profit_pct / 100)
                            signal.stop_loss = signal.entry_price * (1 - self.config.scalping_stop_loss_pct / 100)
                        else:
                            # Opening a SHORT position - target is LOWER (profit when price drops), stop is HIGHER (stop when price rises)
                            signal.target_price = signal.entry_price * (1 - self.config.scalping_take_profit_pct / 100)
                            signal.stop_loss = signal.entry_price * (1 + self.config.scalping_stop_loss_pct / 100)
                    
                    position_type = "LONG" if signal.signal == 'BUY' else ("CLOSING LONG" if has_position else "SHORT")
                    logger.info(f"📊 Scalping mode ({signal.signal}/{position_type}): Entry=${signal.entry_price:.2f}, Target=${signal.target_price:.2f}, Stop=${signal.stop_loss:.2f}")
            
            # PDT-safe position sizing using settled funds
            if signal.entry_price and signal.stop_loss and self._cash_manager:
                try:
                    bal_success, bal_data = self.tradier_client.get_account_balance()
                    settled_cash = None
                    total_equity = 10000.0
                    if bal_success and isinstance(bal_data, dict):
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
                    max_by_pct = int((total_equity * (self.config.max_position_size_pct / 100.0)) // signal.entry_price)
                    final_shares = max(0, min(affordable, max_by_pct))
                    if final_shares <= 0:
                        logger.info("No settled cash available for this entry; skipping")
                        return
                    signal.position_size = final_shares
                except Exception as e:
                    logger.error(f"Error sizing position with settled funds: {e}")
                    return

            # Place bracket order if enabled
            if self.config.use_bracket_orders and signal.entry_price and signal.target_price and signal.stop_loss:
                # Determine duration based on mode
                duration = 'day' if self.config.trading_mode == "SCALPING" else 'gtc'
                
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
                logger.info(f"✅ Order placed successfully for {signal.symbol}")
            else:
                logger.error(f"❌ Failed to place order for {signal.symbol}: {result}")
                
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}", exc_info=True)
    
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
                'allow_short_selling': self.config.allow_short_selling
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
            if success and isinstance(bal_data, dict):
                b = bal_data.get('balances', {})
                settled_cash = float(b.get('cash_available', b.get('total_cash', 10000.0)))
                total_equity = float(b.get('total_equity', settled_cash or 10000.0))
            else:
                settled_cash = 10000.0
                total_equity = 10000.0
            
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
