"""
Position Exit Monitor - Actively monitors and manages open positions
Ensures positions are closed based on stop-loss, take-profit, time limits, and other exit conditions

Features:
- Real-time position monitoring
- Multiple exit strategies (stop-loss, take-profit, time-based, trailing stops)
- Safety guards and error handling
- State persistence
- Discord notifications
- Automatic bracket order management
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Exit reason enumeration"""
    STOP_LOSS = "Stop Loss Hit"
    TAKE_PROFIT = "Take Profit Hit"
    TIME_LIMIT = "Time Limit Reached"
    TRAILING_STOP = "Trailing Stop Hit"
    MARKET_CLOSE = "Market Closing"
    MANUAL = "Manual Close"
    EMERGENCY = "Emergency Exit"
    BRACKET_ORDER = "Bracket Order Filled"


@dataclass
class MonitoredPosition:
    """Tracked position with exit rules"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    
    # Optional advanced exit rules
    trailing_stop_pct: Optional[float] = None
    max_hold_minutes: Optional[int] = None
    break_even_pct: Optional[float] = None  # Move stop to breakeven after this profit
    
    # Tracking
    highest_price: float = 0.0
    lowest_price: float = 0.0
    current_price: float = 0.0
    last_check_time: datetime = None
    exit_attempts: int = 0
    
    # Bracket order tracking
    bracket_order_ids: Optional[List[str]] = None
    has_active_bracket: bool = False


class PositionExitMonitor:
    """
    Monitors open positions and executes exits based on multiple strategies
    Runs in background and ensures positions don't sit unprotected
    """
    
    def __init__(self, broker_client, state_manager, capital_manager=None,
                 check_interval_seconds: int = 30,
                 max_exit_retries: int = 3,
                 enable_trailing_stops: bool = True,
                 enable_time_limits: bool = True,
                 enable_break_even_stops: bool = True,
                 long_term_holdings: Optional[List[str]] = None):
        """
        Initialize Position Exit Monitor
        
        Args:
            broker_client: Broker client (TradierClient, IBKRClient, or BrokerAdapter) for orders and quotes
            state_manager: TradeStateManager for state persistence
            capital_manager: CapitalManager for capital tracking
            check_interval_seconds: How often to check positions (default: 30s)
            max_exit_retries: Max attempts to close a position
            enable_trailing_stops: Enable trailing stop feature
            enable_time_limits: Enable time-based exits
            enable_break_even_stops: Enable break-even stop moves
            long_term_holdings: List of tickers to NEVER auto-sell (e.g., ['BXP', 'AAPL'])
        """
        self.broker_client = broker_client
        # Keep tradier_client as alias for backward compatibility
        self.tradier_client = broker_client
        self.state_manager = state_manager
        self.capital_manager = capital_manager
        
        self.check_interval_seconds = check_interval_seconds
        self.max_exit_retries = max_exit_retries
        self.enable_trailing_stops = enable_trailing_stops
        self.enable_time_limits = enable_time_limits
        self.enable_break_even_stops = enable_break_even_stops
        
        # Long-Term Holdings Protection (CRITICAL SAFETY FEATURE)
        self.long_term_holdings: Set[str] = set(long_term_holdings) if long_term_holdings else set()
        
        # Monitored positions
        self.monitored_positions: Dict[str, MonitoredPosition] = {}
        
        # Track pending exit orders (symbol -> order_id)
        self.pending_exit_orders: Dict[str, str] = {}
        
        # Blacklist for positions with rejected orders (prevent re-adding)
        self.rejected_positions: Set[str] = set()
        
        # Tradier error tracking (for circuit breaker pattern)
        self.consecutive_tradier_errors = 0
        self.last_tradier_error_time = None
        self.tradier_circuit_breaker_active = False
        
        # Safety flags
        self.is_running = False
        self.emergency_mode = False
        self.last_health_check = datetime.now()
        
        # Statistics
        self.total_exits_executed = 0
        self.stop_loss_exits = 0
        self.take_profit_exits = 0
        self.time_limit_exits = 0
        self.trailing_stop_exits = 0
        self.failed_exit_attempts = 0
        
        logger.info("=" * 80)
        logger.info("🛡️ POSITION EXIT MONITOR INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"   Check Interval: {check_interval_seconds}s")
        logger.info(f"   Trailing Stops: {'✅ ENABLED' if enable_trailing_stops else '❌ DISABLED'}")
        logger.info(f"   Time Limits: {'✅ ENABLED' if enable_time_limits else '❌ DISABLED'}")
        logger.info(f"   Break-Even Stops: {'✅ ENABLED' if enable_break_even_stops else '❌ DISABLED'}")
        logger.info(f"   Max Exit Retries: {max_exit_retries}")
        logger.info("=" * 80)
    
    def add_position(self, symbol: str, side: str, quantity: int, entry_price: float,
                    entry_time: datetime, stop_loss: float, take_profit: float,
                    trailing_stop_pct: Optional[float] = None,
                    max_hold_minutes: Optional[int] = None,
                    bracket_order_ids: Optional[List[str]] = None) -> bool:
        """
        Add a position to monitor
        
        Args:
            symbol: Stock ticker
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            entry_price: Entry price
            entry_time: Entry datetime
            stop_loss: Stop loss price
            take_profit: Take profit price
            trailing_stop_pct: Optional trailing stop percentage
            max_hold_minutes: Optional max hold time in minutes
            bracket_order_ids: Optional bracket order IDs if already placed
            
        Returns:
            True if added successfully
        """
        try:
            # CRITICAL: Skip long-term holdings
            if symbol.upper() in self.long_term_holdings:
                logger.info(f"🔒 SKIPPING {symbol}: Protected long-term holding - will NOT be monitored or auto-sold")
                return False
            
            # Validate inputs
            if not symbol or quantity <= 0:
                logger.error(f"❌ Invalid position parameters: {symbol}, qty={quantity}")
                return False
            
            # Create monitored position
            position = MonitoredPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=entry_time,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_pct=trailing_stop_pct,
                max_hold_minutes=max_hold_minutes,
                highest_price=entry_price,
                lowest_price=entry_price,
                current_price=entry_price,
                last_check_time=datetime.now(),
                bracket_order_ids=bracket_order_ids,
                has_active_bracket=bracket_order_ids is not None and len(bracket_order_ids) > 0
            )
            
            self.monitored_positions[symbol] = position
            
            # Calculate risk/reward
            if side == 'BUY':
                risk_pct = ((entry_price - stop_loss) / entry_price) * 100
                reward_pct = ((take_profit - entry_price) / entry_price) * 100
            else:  # SELL (short)
                risk_pct = ((stop_loss - entry_price) / entry_price) * 100
                reward_pct = ((entry_price - take_profit) / entry_price) * 100
            
            rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
            
            logger.info("=" * 80)
            logger.info(f"🎯 MONITORING STARTED: {symbol}")
            logger.info("=" * 80)
            logger.info(f"   Position: {side} {quantity} shares @ ${entry_price:.2f}")
            logger.info(f"   Stop Loss: ${stop_loss:.2f} ({-risk_pct:.2f}%)")
            logger.info(f"   Take Profit: ${take_profit:.2f} (+{reward_pct:.2f}%)")
            logger.info(f"   Risk/Reward: {rr_ratio:.2f}:1")
            if trailing_stop_pct:
                logger.info(f"   Trailing Stop: {trailing_stop_pct:.1f}%")
            if max_hold_minutes:
                logger.info(f"   Max Hold Time: {max_hold_minutes} minutes")
            if bracket_order_ids:
                logger.info(f"   Bracket Orders: {bracket_order_ids}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding position {symbol}: {e}", exc_info=True)
            return False
    
    def remove_position(self, symbol: str) -> bool:
        """Remove a position from monitoring"""
        if symbol in self.monitored_positions:
            del self.monitored_positions[symbol]
            logger.info(f"🗑️ Removed {symbol} from monitoring")
            return True
        return False
    
    def sync_with_broker(self) -> Tuple[int, int]:
        """
        Sync monitored positions with actual broker positions
        Removes positions that were closed, adds new ones
        
        Returns:
            (positions_added, positions_removed)
        """
        try:
            success, positions = self.tradier_client.get_positions()
            if not success or not positions:
                logger.debug("No broker positions to sync")
                return 0, 0
            
            broker_symbols = {pos.get('symbol') for pos in positions if pos.get('symbol')}
            monitored_symbols = set(self.monitored_positions.keys())
            
            # Remove positions no longer at broker (were closed)
            removed = 0
            for symbol in list(monitored_symbols - broker_symbols):
                logger.info(f"📤 Position {symbol} closed at broker, removing from monitoring")
                self.remove_position(symbol)
                
                # Clear pending exit order if it exists
                if symbol in self.pending_exit_orders:
                    order_id = self.pending_exit_orders.pop(symbol)
                    logger.info(f"✅ Pending exit order {order_id} for {symbol} confirmed filled/closed")
                
                removed += 1
            
            # Add positions at broker that aren't monitored
            added = 0
            for pos in positions:
                symbol = pos.get('symbol')
                
                # CRITICAL: Skip long-term holdings
                if symbol and symbol.upper() in self.long_term_holdings:
                    logger.debug(f"🔒 Skipping {symbol} - protected long-term holding")
                    continue
                
                # Skip positions on the rejected blacklist
                if symbol and symbol in self.rejected_positions:
                    logger.debug(f"⏭️ Skipping {symbol} - on rejected positions blacklist")
                    continue
                
                if symbol and symbol not in self.monitored_positions:
                    # Get position details
                    quantity = int(pos.get('quantity', 0))
                    cost_basis = float(pos.get('cost_basis', 0))
                    entry_price = cost_basis / quantity if quantity > 0 else 0
                    
                    # Use default stops if not specified (2% profit, 1% loss for scalping)
                    stop_loss = entry_price * 0.99  # 1% stop
                    take_profit = entry_price * 1.02  # 2% target
                    
                    self.add_position(
                        symbol=symbol,
                        side='BUY',  # Assume long position
                        quantity=quantity,
                        entry_price=entry_price,
                        entry_time=datetime.now(),  # Unknown, use current time
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        max_hold_minutes=480  # 8 hours default
                    )
                    added += 1
            
            if added > 0 or removed > 0:
                logger.info(f"🔄 Sync complete: +{added} added, -{removed} removed")
            
            return added, removed
            
        except Exception as e:
            logger.error(f"❌ Error syncing with broker: {e}", exc_info=True)
            return 0, 0
    
    def check_positions(self) -> List[Dict]:
        """
        Check all monitored positions for exit conditions
        
        Returns:
            List of positions that need to exit
        """
        positions_to_exit = []
        
        if not self.monitored_positions:
            logger.debug("No positions to monitor")
            return positions_to_exit
        
        # Circuit breaker check - if Tradier is having issues, log warning
        if self.tradier_circuit_breaker_active:
            if self.last_tradier_error_time:
                time_since_error = (datetime.now() - self.last_tradier_error_time).total_seconds() / 60
                if time_since_error > 5:  # Try to reset after 5 minutes
                    logger.info("🔄 Circuit breaker: Attempting recovery after 5 min cooldown...")
                    self.tradier_circuit_breaker_active = False
                    self.consecutive_tradier_errors = 0
                else:
                    logger.debug(f"⚠️ Circuit breaker active: Tradier backend issues ({time_since_error:.1f} min ago)")
        
        logger.debug(f"🔍 Checking {len(self.monitored_positions)} positions...")
        
        for symbol, position in list(self.monitored_positions.items()):
            try:
                # Get current price
                quote = self.tradier_client.get_quote(symbol)
                if not quote:
                    logger.warning(f"⚠️ Failed to get quote for {symbol}, skipping check")
                    continue
                
                current_price = float(quote.get('last', 0) or quote.get('bid', 0) or 0)
                if current_price <= 0:
                    logger.warning(f"⚠️ Invalid price for {symbol}: {current_price}")
                    continue
                
                # Update position tracking
                position.current_price = current_price
                position.last_check_time = datetime.now()
                position.highest_price = max(position.highest_price, current_price)
                position.lowest_price = min(position.lowest_price, current_price) if position.lowest_price > 0 else current_price
                
                # Calculate current P&L
                if position.side == 'BUY':
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                else:  # SELL (short)
                    pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                
                # Check exit conditions
                exit_reason = self._check_exit_conditions(position, current_price, pnl_pct)
                
                if exit_reason:
                    positions_to_exit.append({
                        'position': position,
                        'current_price': current_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })
                    logger.info(f"🚨 EXIT SIGNAL: {symbol} - {exit_reason.value} (P&L: {pnl_pct:+.2f}%)")
                else:
                    # Log status every 10 checks (reduce spam)
                    if not hasattr(position, '_check_count'):
                        position._check_count = 0
                    position._check_count += 1
                    
                    if position._check_count % 10 == 0:
                        logger.debug(f"✓ {symbol}: ${current_price:.2f} (P&L: {pnl_pct:+.2f}%), Stop: ${position.stop_loss:.2f}, Target: ${position.take_profit:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Error checking {symbol}: {e}", exc_info=True)
                continue
        
        return positions_to_exit
    
    def _check_exit_conditions(self, position: MonitoredPosition, current_price: float, pnl_pct: float) -> Optional[ExitReason]:
        """
        Check if position meets any exit conditions
        
        Args:
            position: Position to check
            current_price: Current market price
            pnl_pct: Current P&L percentage
            
        Returns:
            ExitReason if exit condition met, None otherwise
        """
        # 1. STOP LOSS CHECK (highest priority)
        if position.side == 'BUY':
            if current_price <= position.stop_loss:
                return ExitReason.STOP_LOSS
        else:  # SELL (short)
            if current_price >= position.stop_loss:
                return ExitReason.STOP_LOSS
        
        # 2. TAKE PROFIT CHECK
        if position.side == 'BUY':
            if current_price >= position.take_profit:
                return ExitReason.TAKE_PROFIT
        else:  # SELL (short)
            if current_price <= position.take_profit:
                return ExitReason.TAKE_PROFIT
        
        # 3. TRAILING STOP CHECK (if enabled and in profit)
        if self.enable_trailing_stops and position.trailing_stop_pct and pnl_pct > 0:
            trailing_stop_price = position.highest_price * (1 - position.trailing_stop_pct / 100)
            if position.side == 'BUY' and current_price <= trailing_stop_price:
                return ExitReason.TRAILING_STOP
        
        # 4. BREAK-EVEN STOP CHECK (if enabled and configured)
        if self.enable_break_even_stops and position.break_even_pct:
            if pnl_pct >= position.break_even_pct:
                # Move stop to break-even if not already
                if position.stop_loss < position.entry_price:
                    logger.info(f"📈 {position.symbol}: Moving stop to break-even (P&L: +{pnl_pct:.2f}%)")
                    position.stop_loss = position.entry_price
        
        # 5. TIME LIMIT CHECK (if enabled)
        if self.enable_time_limits and position.max_hold_minutes:
            hold_time = (datetime.now() - position.entry_time).total_seconds() / 60
            if hold_time >= position.max_hold_minutes:
                return ExitReason.TIME_LIMIT
        
        # 6. EMERGENCY MODE CHECK
        if self.emergency_mode:
            return ExitReason.EMERGENCY
        
        return None
    
    def execute_exit(self, position: MonitoredPosition, current_price: float, exit_reason: ExitReason) -> bool:
        """
        Execute exit for a position
        
        Args:
            position: Position to exit
            current_price: Current market price
            exit_reason: Reason for exit
            
        Returns:
            True if exit successful
        """
        symbol = position.symbol
        
        # Check if there's already a pending exit order for this symbol
        if symbol in self.pending_exit_orders:
            order_id = self.pending_exit_orders[symbol]
            logger.info(f"⏳ EXIT ALREADY PENDING: {symbol} (Order ID: {order_id}) - Waiting for fill...")
            return False  # Don't place another order
        
        try:
            logger.info("=" * 80)
            logger.info(f"🚪 EXECUTING EXIT: {symbol}")
            logger.info("=" * 80)
            logger.info(f"   Reason: {exit_reason.value}")
            logger.info(f"   Quantity: {position.quantity}")
            logger.info(f"   Entry: ${position.entry_price:.2f}")
            logger.info(f"   Current: ${current_price:.2f}")
            
            # Calculate P&L
            if position.side == 'BUY':
                pnl = (current_price - position.entry_price) * position.quantity
                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            else:  # SELL (short)
                pnl = (position.entry_price - current_price) * position.quantity
                pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
            
            logger.info(f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
            logger.info("=" * 80)
            
            # Place market order to close
            order_side = 'sell' if position.side == 'BUY' else 'buy'  # Opposite of entry
            
            # Remove underscores from tag (Tradier doesn't accept them)
            reason_tag = exit_reason.name.replace('_', '')
            
            order_data = {
                'class': 'equity',
                'symbol': symbol,
                'side': order_side,
                'quantity': str(position.quantity),
                'type': 'market',
                'duration': 'day',
                'tag': f"EXIT{reason_tag}{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
            
            success, result = self.tradier_client.place_order(order_data)
            
            if success:
                position.exit_attempts += 1
                
                # Reset Tradier error counter on success
                self.consecutive_tradier_errors = 0
                self.tradier_circuit_breaker_active = False
                
                # Track pending order (extract order ID from result)
                order_id = result.get('order', {}).get('id', 'unknown') if isinstance(result, dict) else 'unknown'
                self.pending_exit_orders[symbol] = str(order_id)
                logger.info(f"📋 Tracking pending exit order for {symbol}: Order ID {order_id}")
                
                # Record in state manager
                self.state_manager.record_trade_closed(
                    symbol=symbol,
                    exit_price=current_price,
                    reason=exit_reason.value
                )
                
                # Release capital
                if self.capital_manager:
                    self.capital_manager.release_capital(symbol, pnl=pnl)
                    logger.info(f"💰 Capital released: ${self.capital_manager.get_available_capital():,.2f} available")
                
                # Update statistics
                self.total_exits_executed += 1
                if exit_reason == ExitReason.STOP_LOSS:
                    self.stop_loss_exits += 1
                elif exit_reason == ExitReason.TAKE_PROFIT:
                    self.take_profit_exits += 1
                elif exit_reason == ExitReason.TIME_LIMIT:
                    self.time_limit_exits += 1
                elif exit_reason == ExitReason.TRAILING_STOP:
                    self.trailing_stop_exits += 1
                
                # CRITICAL: Do NOT remove from monitoring yet!
                # The order is PENDING and may not fill. We wait for sync_with_broker() to confirm
                # that the position is actually gone from the broker before removing it.
                # This prevents re-submitting duplicate exit orders for the same position.
                logger.info(f"⏳ Position {symbol} exit order submitted (Order ID: {order_id}). Waiting for fill...")
                
                # Check order status after brief delay to detect immediate rejections
                import time
                time.sleep(1)  # Give Tradier a moment to process
                
                try:
                    success_check, order_status = self.tradier_client.get_order_status(str(order_id))
                    if success_check:
                        status = order_status.get('order', {}).get('status', '').lower()
                        
                        if status == 'rejected':
                            logger.error("=" * 80)
                            logger.error(f"🚫 ORDER REJECTED BY BROKER: {symbol}")
                            logger.error(f"   Order ID: {order_id}")
                            logger.error(f"   This is a broker issue (Tradier paper trading limitation)")
                            logger.error(f"   STOPPING retry attempts to prevent order spam")
                            logger.error(f"   Manual intervention required via Tradier web interface")
                            logger.error("=" * 80)
                            
                            # Add to blacklist to prevent re-adding
                            self.rejected_positions.add(symbol)
                            logger.info(f"🚫 Added {symbol} to rejected positions blacklist")
                            
                            # Remove from monitoring to prevent retry spam
                            if symbol in self.monitored_positions:
                                del self.monitored_positions[symbol]
                                logger.info(f"🗑️ Removed {symbol} from monitoring (rejected order)")
                            
                            # Remove pending order tracking
                            if symbol in self.pending_exit_orders:
                                del self.pending_exit_orders[symbol]
                            
                            return False  # Order was rejected, not successful
                        elif status == 'open' or status == 'pending':
                            logger.info(f"✅ Order {order_id} status: {status.upper()} - monitoring for fill")
                        elif status == 'filled':
                            logger.info(f"🎉 Order {order_id} FILLED immediately!")
                except Exception as e:
                    logger.debug(f"Could not check order status (non-critical): {e}")
                
                # Send Discord notification
                try:
                    self._send_exit_notification(position, current_price, pnl, pnl_pct, exit_reason)
                except Exception as e:
                    logger.error(f"Failed to send Discord notification: {e}")
                
                logger.info(f"✅ EXIT SUCCESSFUL: {symbol} - Order placed (waiting for fill)")
                return True
            else:
                position.exit_attempts += 1
                self.failed_exit_attempts += 1
                
                # Check if this is a Tradier backend error (500)
                is_tradier_500_error = False
                if isinstance(result, dict):
                    error_msg = str(result.get('error', ''))
                    response_text = str(result.get('response_text', ''))
                    if '500' in error_msg or '500' in response_text or 'backend' in response_text.lower():
                        is_tradier_500_error = True
                        self.consecutive_tradier_errors += 1
                        self.last_tradier_error_time = datetime.now()
                        
                        logger.error(f"❌ EXIT FAILED: {symbol} - Tradier 500 Server Error (consecutive: {self.consecutive_tradier_errors})")
                        
                        # Activate circuit breaker after 5 consecutive errors
                        if self.consecutive_tradier_errors >= 5 and not self.tradier_circuit_breaker_active:
                            self.tradier_circuit_breaker_active = True
                            logger.error("=" * 80)
                            logger.error("⚠️ CIRCUIT BREAKER ACTIVATED: Tradier backend issues detected")
                            logger.error("   Will retry exits less frequently to avoid spam")
                            logger.error("   Position monitoring continues - will retry when Tradier recovers")
                            logger.error("=" * 80)
                    else:
                        # Other error, not 500
                        logger.error(f"❌ EXIT FAILED: {symbol} - {result}")
                        self.consecutive_tradier_errors = 0  # Reset counter for non-500 errors
                else:
                    logger.error(f"❌ EXIT FAILED: {symbol} - {result}")
                
                # Retry if under max attempts
                if position.exit_attempts < self.max_exit_retries:
                    retry_msg = f"🔄 Will retry exit on next check (attempt {position.exit_attempts}/{self.max_exit_retries})"
                    if is_tradier_500_error and self.consecutive_tradier_errors >= 3:
                        retry_msg += " - Waiting for Tradier backend to recover"
                    logger.info(retry_msg)
                else:
                    if is_tradier_500_error:
                        logger.error(f"⚠️ MAX RETRIES REACHED for {symbol} due to Tradier backend issues")
                        logger.error(f"   Resetting retry counter - will continue trying when Tradier recovers")
                        # Reset attempts to keep trying (Tradier issue, not our fault)
                        position.exit_attempts = 0
                    else:
                        logger.error(f"⚠️ MAX RETRIES REACHED for {symbol}, manual intervention required!")
                
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing exit for {symbol}: {e}", exc_info=True)
            position.exit_attempts += 1
            self.failed_exit_attempts += 1
            return False
    
    def _send_exit_notification(self, position: MonitoredPosition, exit_price: float, 
                               pnl: float, pnl_pct: float, exit_reason: ExitReason):
        """Send Discord notification for position exit"""
        import os
        import requests
        
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if not webhook_url:
            return
        
        # Color based on P&L
        color = 65280 if pnl > 0 else 16711680  # Green for profit, Red for loss
        
        # Emoji based on exit reason
        emoji_map = {
            ExitReason.STOP_LOSS: "🛑",
            ExitReason.TAKE_PROFIT: "🎯",
            ExitReason.TIME_LIMIT: "⏱️",
            ExitReason.TRAILING_STOP: "📈",
            ExitReason.MARKET_CLOSE: "🔔",
            ExitReason.MANUAL: "👤",
            ExitReason.EMERGENCY: "🚨"
        }
        emoji = emoji_map.get(exit_reason, "📤")
        
        title = f"{emoji} POSITION CLOSED: {position.symbol}"
        if pnl > 0:
            title += " 💰 PROFIT"
        else:
            title += " 💸 LOSS"
        
        embed = {
            'title': title,
            'description': f"**Exit Reason:** {exit_reason.value}",
            'color': color,
            'timestamp': datetime.now().isoformat(),
            'fields': [
                {
                    'name': '📊 Position Details',
                    'value': f"{position.side} {position.quantity} shares",
                    'inline': False
                },
                {
                    'name': '💵 Entry Price',
                    'value': f"${position.entry_price:.2f}",
                    'inline': True
                },
                {
                    'name': '💵 Exit Price',
                    'value': f"${exit_price:.2f}",
                    'inline': True
                },
                {
                    'name': '⏱️ Hold Time',
                    'value': f"{(datetime.now() - position.entry_time).total_seconds() / 60:.0f} min",
                    'inline': True
                },
                {
                    'name': '💰 Profit/Loss',
                    'value': f"${pnl:+,.2f}",
                    'inline': True
                },
                {
                    'name': '📈 P/L %',
                    'value': f"{pnl_pct:+.2f}%",
                    'inline': True
                },
                {
                    'name': '📊 Total Value',
                    'value': f"${exit_price * position.quantity:,.2f}",
                    'inline': True
                }
            ],
            'footer': {
                'text': f"Position Exit Monitor | Exit #{self.total_exits_executed}"
            }
        }
        
        payload = {'embeds': [embed]}
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.debug(f"✅ Discord exit notification sent for {position.symbol}")
        except Exception as e:
            logger.debug(f"Failed to send Discord notification: {e}")
    
    def close_all_positions(self, reason: str = "Manual close all") -> int:
        """
        Emergency function to close all monitored positions
        
        Args:
            reason: Reason for closing all positions
            
        Returns:
            Number of positions closed
        """
        logger.warning("=" * 80)
        logger.warning(f"⚠️ CLOSING ALL POSITIONS: {reason}")
        logger.warning("=" * 80)
        
        closed_count = 0
        for symbol, position in list(self.monitored_positions.items()):
            try:
                # Get current price
                quote = self.tradier_client.get_quote(symbol)
                current_price = float(quote.get('last', position.entry_price)) if quote else position.entry_price
                
                # Execute exit
                if self.execute_exit(position, current_price, ExitReason.MANUAL):
                    closed_count += 1
                    
            except Exception as e:
                logger.error(f"Error closing {symbol}: {e}")
        
        logger.warning(f"⚠️ Closed {closed_count}/{len(self.monitored_positions)} positions")
        return closed_count
    
    def get_status(self) -> Dict:
        """Get current monitor status"""
        return {
            'is_running': self.is_running,
            'monitored_positions': len(self.monitored_positions),
            'pending_exit_orders': len(self.pending_exit_orders),
            'pending_orders': dict(self.pending_exit_orders),  # Include order IDs
            'positions': {
                symbol: {
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'quantity': pos.quantity,
                    'side': pos.side,
                    'pnl_pct': ((pos.current_price - pos.entry_price) / pos.entry_price) * 100 if pos.side == 'BUY' else ((pos.entry_price - pos.current_price) / pos.entry_price) * 100,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'hold_time_minutes': (datetime.now() - pos.entry_time).total_seconds() / 60,
                    'has_pending_exit': symbol in self.pending_exit_orders
                }
                for symbol, pos in self.monitored_positions.items()
            },
            'statistics': {
                'total_exits': self.total_exits_executed,
                'stop_loss_exits': self.stop_loss_exits,
                'take_profit_exits': self.take_profit_exits,
                'time_limit_exits': self.time_limit_exits,
                'trailing_stop_exits': self.trailing_stop_exits,
                'failed_attempts': self.failed_exit_attempts
            },
            'emergency_mode': self.emergency_mode,
            'last_health_check': self.last_health_check.isoformat()
        }
    
    def start_monitoring_loop(self):
        """
        Main monitoring loop - checks positions regularly
        Should be called in a separate thread
        """
        self.is_running = True
        logger.info("=" * 80)
        logger.info("🚀 POSITION EXIT MONITOR STARTED")
        logger.info("=" * 80)
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                # Health check
                self.last_health_check = datetime.now()
                
                # Sync with broker first
                self.sync_with_broker()
                
                # Log pending exit orders if any
                if self.pending_exit_orders:
                    logger.info(f"⏳ {len(self.pending_exit_orders)} pending exit orders awaiting fill: {list(self.pending_exit_orders.keys())}")
                    
                    # Check status of pending orders to detect rejections
                    rejected_symbols = []
                    for symbol, order_id in list(self.pending_exit_orders.items()):
                        try:
                            success, order_status = self.tradier_client.get_order_status(str(order_id))
                            if success:
                                status = order_status.get('order', {}).get('status', '').lower()
                                
                                if status == 'rejected':
                                    logger.error("=" * 80)
                                    logger.error(f"🚫 DETECTED REJECTED ORDER: {symbol}")
                                    logger.error(f"   Order ID: {order_id}")
                                    logger.error(f"   Broker rejected this exit order")
                                    logger.error(f"   Removing from monitoring to prevent retry spam")
                                    logger.error("=" * 80)
                                    rejected_symbols.append(symbol)
                                    # Add to blacklist
                                    self.rejected_positions.add(symbol)
                                    logger.info(f"🚫 Added {symbol} to rejected positions blacklist")
                        except Exception as e:
                            logger.debug(f"Could not check order status for {symbol}: {e}")
                    
                    # Remove rejected orders from tracking and monitoring
                    for symbol in rejected_symbols:
                        if symbol in self.pending_exit_orders:
                            del self.pending_exit_orders[symbol]
                        if symbol in self.monitored_positions:
                            del self.monitored_positions[symbol]
                            logger.info(f"🗑️ Removed {symbol} from monitoring due to rejected order")
                
                # Check all positions
                positions_to_exit = self.check_positions()
                
                # Execute exits
                for exit_info in positions_to_exit:
                    self.execute_exit(
                        position=exit_info['position'],
                        current_price=exit_info['current_price'],
                        exit_reason=exit_info['exit_reason']
                    )
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Sleep until next check
                time.sleep(self.check_interval_seconds)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ Error in monitoring loop: {e}", exc_info=True)
                
                # If too many consecutive errors, enter emergency mode
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"⚠️ TOO MANY ERRORS ({consecutive_errors}), entering emergency mode!")
                    self.emergency_mode = True
                
                # Short sleep before retry
                time.sleep(10)
        
        logger.info("🛑 Position Exit Monitor stopped")
    
    def stop(self):
        """Stop the monitoring loop"""
        logger.info("⏸️ Stopping Position Exit Monitor...")
        self.is_running = False

