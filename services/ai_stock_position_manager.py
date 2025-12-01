"""
AI Stock Position Manager
Manages stock positions with Discord approval workflow and broker execution

Features:
- Trade execution via Tradier or IBKR broker adapters
- Discord approval integration for trade confirmation
- Position monitoring with stop loss and take profit
- Paper trading support for testing
- Integration with AI Stock Entry Assistant for entry timing
"""

import json
import os
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class StockPositionStatus(Enum):
    """Position lifecycle status"""
    PENDING_APPROVAL = "PENDING_APPROVAL"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class StockPositionAction(Enum):
    """Trade actions"""
    HOLD = "HOLD"
    TIGHTEN_STOP = "TIGHTEN_STOP"
    TAKE_PARTIAL = "TAKE_PARTIAL"
    CLOSE_NOW = "CLOSE_NOW"
    MOVE_TO_BREAKEVEN = "MOVE_TO_BREAKEVEN"


@dataclass
class StockPosition:
    """Tracked stock position"""
    trade_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL' (short)
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    strategy: str
    
    # Dynamic tracking
    current_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    last_check_time: Optional[datetime] = None
    
    # Management settings
    trailing_stop_pct: float = 2.0
    breakeven_trigger_pct: float = 3.0
    moved_to_breakeven: bool = False
    partial_exit_taken: bool = False
    
    # Status
    status: str = StockPositionStatus.PENDING_APPROVAL.value
    order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    
    # Reasoning
    reasoning: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        data['last_check_time'] = self.last_check_time.isoformat() if self.last_check_time else None
        return data


@dataclass
class StockTradeDecision:
    """Trade decision for approval"""
    symbol: str
    side: str
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float
    reasoning: str
    urgency: str = "MEDIUM"
    
    # Action for position updates
    action: str = "BUY"


def _load_service_interval(service_name: str = "sentient-stock-ai-trader") -> int:
    """Load the check interval from service config file"""
    config_file = Path(__file__).parent.parent / "data" / "service_intervals.json"
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                service_config = config.get(service_name, {})
                return service_config.get("check_interval_seconds", 60)
    except Exception as e:
        logger.debug(f"Could not load service interval config: {e}")
    return 60  # Default


class AIStockPositionManager:
    """
    AI-Enhanced Position Manager for Stock Trading
    Integrates with Tradier/IBKR via BrokerAdapter for execution
    """
    
    def __init__(
        self,
        broker_adapter=None,
        llm_analyzer=None,
        check_interval_seconds: Optional[int] = None,
        enable_trailing_stops: bool = True,
        enable_breakeven_moves: bool = True,
        min_confidence: float = 70.0,
        default_position_size: float = 500.0,
        risk_per_trade_pct: float = 2.0,
        state_file: str = "data/ai_stock_positions.json",
        require_manual_approval: bool = True,
        paper_mode: bool = True,
        max_positions: int = 10
    ):
        """
        Initialize AI Stock Position Manager
        
        Args:
            broker_adapter: BrokerAdapter instance (TradierAdapter or IBKRAdapter)
            llm_analyzer: LLM analyzer for AI decisions
            check_interval_seconds: How often to check positions
            enable_trailing_stops: Enable trailing stop feature
            enable_breakeven_moves: Enable break-even stop moves
            min_confidence: Minimum confidence to execute
            default_position_size: Default position size in USD
            risk_per_trade_pct: Risk percentage per trade
            state_file: File to persist position state
            require_manual_approval: Require Discord/manual approval before trades
            paper_mode: Use paper trading (safety first!)
            max_positions: Maximum concurrent positions
        """
        self.broker_adapter = broker_adapter
        self.llm_analyzer = llm_analyzer
        
        # Load interval from config if not provided
        if check_interval_seconds is None:
            check_interval_seconds = _load_service_interval()
        self.check_interval_seconds = check_interval_seconds
        
        self.enable_trailing_stops = enable_trailing_stops
        self.enable_breakeven_moves = enable_breakeven_moves
        self.min_confidence = min_confidence
        self.default_position_size = default_position_size
        self.risk_per_trade_pct = risk_per_trade_pct
        self.require_manual_approval = require_manual_approval
        self.paper_mode = paper_mode
        self.max_positions = max_positions
        
        # State file path
        if state_file == "data/ai_stock_positions.json":
            base_dir = Path(__file__).parent.parent
            self.state_file = str(base_dir / "data" / "ai_stock_positions.json")
        else:
            self.state_file = state_file
        
        # Positions
        self.positions: Dict[str, StockPosition] = {}
        
        # Pending approvals (for Discord integration)
        self.pending_approvals: Dict[str, Dict] = {}
        
        # Discord approval integration
        self.discord_approval_manager = None
        self._init_discord_approval()
        
        # State management
        self.is_running = False
        self.thread = None
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Load persisted state
        self._load_state()
        
        logger.info("=" * 80)
        logger.info("📊 AI STOCK POSITION MANAGER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"   Broker: {'Configured' if broker_adapter else 'NOT CONFIGURED'}")
        logger.info(f"   Check Interval: {check_interval_seconds}s")
        logger.info(f"   Paper Mode: {'✅ ENABLED' if paper_mode else '⚠️ LIVE TRADING'}")
        logger.info(f"   Min Confidence: {min_confidence}%")
        logger.info(f"   Default Position Size: ${default_position_size:.2f}")
        logger.info(f"   Manual Approval: {'✅ REQUIRED' if require_manual_approval else '⚠️ AUTO'}")
        logger.info(f"   Discord: {'✅ ENABLED' if self.discord_approval_manager else '❌ NOT CONFIGURED'}")
        logger.info("=" * 80)
    
    def _init_discord_approval(self):
        """Initialize Discord approval manager"""
        try:
            from services.discord_trade_approval import get_discord_approval_manager
            
            def on_approval(approval_id: str, approved: bool):
                self._handle_discord_approval(approval_id, approved)
            
            self.discord_approval_manager = get_discord_approval_manager(approval_callback=on_approval)
            
            if self.discord_approval_manager and self.discord_approval_manager.enabled:
                logger.info("✅ Discord approval integration enabled for stocks")
            else:
                logger.warning("⚠️ Discord approval not configured for stocks")
                self.discord_approval_manager = None
                
        except Exception as e:
            logger.warning(f"Could not initialize Discord approval: {e}")
            self.discord_approval_manager = None
    
    def _handle_discord_approval(self, approval_id: str, approved: bool):
        """Handle approval/rejection from Discord"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📨 STOCK TRADE APPROVAL RECEIVED")
        logger.info("=" * 60)
        logger.info(f"   Approval ID: {approval_id}")
        logger.info(f"   Decision: {'✅ APPROVED' if approved else '❌ REJECTED'}")
        
        # Find the pending approval
        if approval_id not in self.pending_approvals:
            logger.warning(f"   ⚠️ Approval ID not found!")
            return
        
        pending = self.pending_approvals[approval_id]
        trade_id = pending.get('trade_id')
        decision = pending.get('decision')
        symbol = pending.get('symbol', trade_id)
        
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Side: {decision.side if decision else 'N/A'}")
        logger.info(f"   Confidence: {decision.confidence:.0f}%" if decision else "N/A")
        
        if approved and decision:
            logger.info("")
            logger.info(f"🚀 EXECUTING APPROVED STOCK TRADE: {symbol}")
            result = self.execute_trade(decision, skip_approval=True)
            if result:
                logger.info("   ✅ Trade executed successfully!")
            else:
                logger.error("   ❌ Trade execution failed!")
        else:
            logger.info("   Trade cancelled")
        
        # Remove from pending
        del self.pending_approvals[approval_id]
        logger.info("=" * 60)
    
    def queue_trade_for_approval(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str,
        confidence: float,
        reasoning: str
    ) -> Optional[str]:
        """
        Queue a trade for Discord approval
        
        Returns:
            Approval ID if queued successfully, None otherwise
        """
        try:
            decision = StockTradeDecision(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=strategy,
                confidence=confidence,
                reasoning=reasoning
            )
            
            approval_id = f"stock_{symbol}_{int(datetime.now().timestamp())}"
            
            # Store pending approval
            self.pending_approvals[approval_id] = {
                'trade_id': approval_id,
                'symbol': symbol,
                'decision': decision,
                'timestamp': datetime.now()
            }
            
            # Send to Discord if available
            if self.discord_approval_manager and self.discord_approval_manager.enabled:
                # Calculate risk metrics
                risk_pct = abs((entry_price - stop_loss) / entry_price) * 100 if entry_price > 0 else 0
                reward_pct = abs((take_profit - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                
                position_size = quantity * entry_price
                
                self.discord_approval_manager.send_approval_request(
                    approval_id=approval_id,
                    pair=symbol,
                    side=side,
                    entry_price=entry_price,
                    position_size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy=strategy,
                    confidence=confidence,
                    reasoning=reasoning,
                    additional_info=f"📊 **Stock Trade** | Paper: {self.paper_mode} | R:R {rr_ratio:.2f}"
                )
                logger.info(f"📨 Stock trade approval sent to Discord: {symbol} {side}")
            else:
                logger.warning(f"⚠️ Discord not available, trade queued locally: {approval_id}")
            
            return approval_id
            
        except Exception as e:
            logger.error(f"Error queuing trade for approval: {e}")
            return None
    
    def execute_trade(self, decision: StockTradeDecision, skip_approval: bool = False) -> bool:
        """
        Execute a stock trade via broker adapter
        
        Args:
            decision: Trade decision to execute
            skip_approval: If True, skip manual approval (already approved)
            
        Returns:
            True if executed successfully
        """
        try:
            if not self.broker_adapter:
                logger.error("❌ No broker adapter configured!")
                return False
            
            # Check approval requirement
            if self.require_manual_approval and not skip_approval:
                logger.warning("⚠️ Manual approval required but not granted")
                return False
            
            symbol = decision.symbol
            side = decision.side
            quantity = decision.quantity
            
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"🚀 EXECUTING STOCK TRADE")
            logger.info("=" * 60)
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Side: {side}")
            logger.info(f"   Quantity: {quantity}")
            logger.info(f"   Entry Price: ${decision.entry_price:.2f}")
            logger.info(f"   Stop Loss: ${decision.stop_loss:.2f}")
            logger.info(f"   Take Profit: ${decision.take_profit:.2f}")
            logger.info(f"   Paper Mode: {self.paper_mode}")
            
            if self.paper_mode:
                # Paper trade - simulate execution
                order_id = f"PAPER_{symbol}_{int(time.time())}"
                logger.info(f"   📝 PAPER ORDER: {order_id}")
                success = True
            else:
                # Real trade via broker adapter
                success, result = self.broker_adapter.place_equity_order(
                    symbol=symbol,
                    side=side.lower(),  # 'buy' or 'sell'
                    quantity=quantity,
                    order_type="market",
                    duration="day",
                    tag=f"AI_STOCK_{decision.strategy}"
                )
                
                if success:
                    order_id = result.get('order_id', f"ORDER_{int(time.time())}")
                    logger.info(f"   ✅ ORDER PLACED: {order_id}")
                else:
                    logger.error(f"   ❌ ORDER FAILED: {result}")
                    return False
            
            # Create position record
            position = StockPosition(
                trade_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=decision.entry_price,
                entry_time=datetime.now(),
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                strategy=decision.strategy,
                current_price=decision.entry_price,
                highest_price=decision.entry_price,
                lowest_price=decision.entry_price,
                status=StockPositionStatus.ACTIVE.value,
                order_id=order_id,
                reasoning=decision.reasoning,
                confidence=decision.confidence
            )
            
            self.positions[order_id] = position
            self.total_trades += 1
            
            # Save state
            self._save_state()
            
            logger.info(f"   📊 Position added to tracking")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return False
    
    def close_position(self, trade_id: str, reason: str = "Manual close") -> bool:
        """Close an existing position"""
        try:
            if trade_id not in self.positions:
                logger.warning(f"Position {trade_id} not found")
                return False
            
            position = self.positions[trade_id]
            
            if position.status != StockPositionStatus.ACTIVE.value:
                logger.warning(f"Position {trade_id} is not active")
                return False
            
            logger.info(f"📊 Closing position: {position.symbol}")
            
            # Determine exit side (opposite of entry)
            exit_side = "sell" if position.side.upper() == "BUY" else "buy"
            
            if self.paper_mode:
                # Paper close
                exit_order_id = f"PAPER_EXIT_{position.symbol}_{int(time.time())}"
                success = True
            else:
                # Real close via broker
                if self.broker_adapter:
                    success, result = self.broker_adapter.place_equity_order(
                        symbol=position.symbol,
                        side=exit_side,
                        quantity=position.quantity,
                        order_type="market",
                        duration="day",
                        tag="AI_STOCK_EXIT"
                    )
                    exit_order_id = result.get('order_id') if success else None
                else:
                    success = False
                    exit_order_id = None
            
            if success:
                position.status = StockPositionStatus.CLOSED.value
                position.exit_order_id = exit_order_id
                
                # Calculate P&L
                if position.side.upper() == "BUY":
                    pnl_pct = ((position.current_price - position.entry_price) / position.entry_price) * 100
                else:
                    pnl_pct = ((position.entry_price - position.current_price) / position.entry_price) * 100
                
                if pnl_pct > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                logger.info(f"   ✅ Position closed | P&L: {pnl_pct:+.2f}% | Reason: {reason}")
                self._save_state()
                return True
            else:
                logger.error(f"   ❌ Failed to close position")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_positions(self, status: Optional[str] = None) -> List[StockPosition]:
        """Get all positions, optionally filtered by status"""
        if status:
            return [p for p in self.positions.values() if p.status == status]
        return list(self.positions.values())
    
    def get_active_positions(self) -> List[StockPosition]:
        """Get all active positions"""
        return self.get_positions(StockPositionStatus.ACTIVE.value)
    
    def get_pending_approvals(self) -> List[Dict]:
        """Get all pending trade approvals"""
        return [
            {
                'approval_id': k,
                'symbol': v.get('symbol'),
                'decision': v.get('decision'),
                'timestamp': v.get('timestamp')
            }
            for k, v in self.pending_approvals.items()
        ]
    
    def _save_state(self):
        """Persist positions to file"""
        try:
            state = {
                'positions': {k: v.to_dict() for k, v in self.positions.items()},
                'stats': {
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'losing_trades': self.losing_trades
                },
                'last_updated': datetime.now().isoformat()
            }
            
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.debug(f"💾 State saved to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self):
        """Load persisted positions from file"""
        try:
            if not Path(self.state_file).exists():
                logger.debug("No saved stock positions found")
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore stats
            stats = state.get('stats', {})
            self.total_trades = stats.get('total_trades', 0)
            self.winning_trades = stats.get('winning_trades', 0)
            self.losing_trades = stats.get('losing_trades', 0)
            
            # Restore positions
            positions_data = state.get('positions', {})
            for trade_id, pos_data in positions_data.items():
                try:
                    # Only restore active positions
                    if pos_data.get('status') == StockPositionStatus.ACTIVE.value:
                        position = StockPosition(
                            trade_id=pos_data['trade_id'],
                            symbol=pos_data['symbol'],
                            side=pos_data['side'],
                            quantity=pos_data['quantity'],
                            entry_price=pos_data['entry_price'],
                            entry_time=datetime.fromisoformat(pos_data['entry_time']),
                            stop_loss=pos_data['stop_loss'],
                            take_profit=pos_data['take_profit'],
                            strategy=pos_data.get('strategy', ''),
                            current_price=pos_data.get('current_price', pos_data['entry_price']),
                            highest_price=pos_data.get('highest_price', pos_data['entry_price']),
                            lowest_price=pos_data.get('lowest_price', pos_data['entry_price']),
                            status=pos_data.get('status', StockPositionStatus.ACTIVE.value),
                            order_id=pos_data.get('order_id'),
                            reasoning=pos_data.get('reasoning', ''),
                            confidence=pos_data.get('confidence', 0)
                        )
                        self.positions[trade_id] = position
                        logger.info(f"📂 Restored stock position: {position.symbol}")
                except Exception as e:
                    logger.warning(f"Failed to restore position {trade_id}: {e}")
            
            if self.positions:
                logger.info(f"✅ Loaded {len(self.positions)} stock positions")
                
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def start_monitoring(self):
        """Start background position monitoring"""
        if self.is_running:
            logger.warning("Stock position monitoring already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info("🎯 Stock position monitoring started")
    
    def stop_monitoring(self):
        """Stop background position monitoring"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        self._save_state()
        logger.info("🛑 Stock position monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                active_positions = self.get_active_positions()
                
                if active_positions:
                    self._check_positions(active_positions)
                
                time.sleep(self.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval_seconds)
    
    def _check_positions(self, positions: List[StockPosition]):
        """Check positions for stop loss / take profit triggers"""
        for position in positions:
            try:
                # Get current price
                current_price = self._get_current_price(position.symbol)
                if not current_price:
                    continue
                
                position.current_price = current_price
                position.last_check_time = datetime.now()
                
                # Update high/low tracking
                position.highest_price = max(position.highest_price, current_price)
                position.lowest_price = min(position.lowest_price, current_price)
                
                # Check stop loss
                if position.side.upper() == "BUY":
                    if current_price <= position.stop_loss:
                        logger.warning(f"🛑 STOP LOSS HIT: {position.symbol} @ ${current_price:.2f}")
                        self.close_position(position.trade_id, "Stop loss triggered")
                        continue
                    
                    if current_price >= position.take_profit:
                        logger.info(f"🎯 TAKE PROFIT HIT: {position.symbol} @ ${current_price:.2f}")
                        self.close_position(position.trade_id, "Take profit triggered")
                        continue
                else:
                    # Short position
                    if current_price >= position.stop_loss:
                        logger.warning(f"🛑 STOP LOSS HIT (SHORT): {position.symbol} @ ${current_price:.2f}")
                        self.close_position(position.trade_id, "Stop loss triggered")
                        continue
                    
                    if current_price <= position.take_profit:
                        logger.info(f"🎯 TAKE PROFIT HIT (SHORT): {position.symbol} @ ${current_price:.2f}")
                        self.close_position(position.trade_id, "Take profit triggered")
                        continue
                
                # Check breakeven move
                if self.enable_breakeven_moves and not position.moved_to_breakeven:
                    pnl_pct = self._calculate_pnl_pct(position)
                    if pnl_pct >= position.breakeven_trigger_pct:
                        position.stop_loss = position.entry_price
                        position.moved_to_breakeven = True
                        logger.info(f"✅ Moved {position.symbol} stop to breakeven")
                
            except Exception as e:
                logger.error(f"Error checking position {position.symbol}: {e}")
        
        # Save state after checking
        self._save_state()
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from broker or fallback"""
        try:
            if self.broker_adapter and hasattr(self.broker_adapter, 'get_quote'):
                quote = self.broker_adapter.get_quote(symbol)
                if quote:
                    return float(quote.get('last', quote.get('last_price', 0)))
            
            # Fallback to yfinance
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def _calculate_pnl_pct(self, position: StockPosition) -> float:
        """Calculate current P&L percentage"""
        if position.side.upper() == "BUY":
            return ((position.current_price - position.entry_price) / position.entry_price) * 100
        else:
            return ((position.entry_price - position.current_price) / position.entry_price) * 100


# Singleton instance
_stock_position_manager: Optional[AIStockPositionManager] = None


def get_ai_stock_position_manager(
    broker_adapter=None,
    **kwargs
) -> Optional[AIStockPositionManager]:
    """
    Get or create singleton AI Stock Position Manager instance
    
    Args:
        broker_adapter: Optional broker adapter (Tradier or IBKR)
        **kwargs: Additional configuration options
        
    Returns:
        AIStockPositionManager instance or None if broker not configured
    """
    global _stock_position_manager
    
    # Try to create broker adapter if not provided
    if broker_adapter is None:
        broker_adapter = _create_broker_adapter()
    
    if _stock_position_manager is None:
        # Determine paper mode from environment/config
        paper_mode = os.getenv('STOCK_PAPER_MODE', 'true').lower() == 'true'
        
        _stock_position_manager = AIStockPositionManager(
            broker_adapter=broker_adapter,
            paper_mode=paper_mode,
            **kwargs
        )
    
    return _stock_position_manager


def _create_broker_adapter():
    """Create broker adapter based on configuration"""
    try:
        broker_type = os.getenv('BROKER_TYPE', 'TRADIER').upper()
        
        if broker_type == 'IBKR':
            from src.integrations.ibkr_client import IBKRClient
            from src.integrations.broker_adapter import IBKRAdapter
            
            port = int(os.getenv('IBKR_PAPER_PORT', '7497'))
            client_id = int(os.getenv('IBKR_PAPER_CLIENT_ID', '1'))
            
            client = IBKRClient(port=port, client_id=client_id)
            if client.connect():
                return IBKRAdapter(client)
            else:
                logger.warning("Failed to connect to IBKR")
                return None
                
        elif broker_type == 'TRADIER':
            from src.integrations.tradier_client import TradierClient
            from src.integrations.broker_adapter import TradierAdapter
            
            api_key = os.getenv('TRADIER_API_KEY')
            account_id = os.getenv('TRADIER_ACCOUNT_ID')
            
            if api_key and account_id:
                # Check if paper trading
                paper_mode = os.getenv('TRADIER_PAPER', 'true').lower() == 'true'
                
                client = TradierClient(
                    api_key=api_key,
                    account_id=account_id,
                    sandbox=paper_mode
                )
                return TradierAdapter(client)
            else:
                logger.warning("Tradier API credentials not configured")
                return None
        
        return None
        
    except Exception as e:
        logger.warning(f"Could not create broker adapter: {e}")
        return None

