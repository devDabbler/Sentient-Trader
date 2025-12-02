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
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float = 75.0
    ) -> Dict:
        """
        Calculate optimal position size based on risk profile
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Signal confidence (0-100)
            
        Returns:
            Dict with shares, value, and risk metrics
        """
        try:
            from services.risk_profile_config import get_risk_profile_manager
            manager = get_risk_profile_manager()
            
            sizing = manager.calculate_position_size(
                price=entry_price,
                stop_loss=stop_loss,
                confidence=confidence
            )
            sizing['symbol'] = symbol
            return sizing
            
        except Exception as e:
            logger.warning(f"Could not calculate position size from risk profile: {e}")
            # Fallback to default sizing
            shares = int(self.default_position_size / entry_price) if entry_price > 0 else 0
            return {
                'symbol': symbol,
                'recommended_shares': shares,
                'recommended_value': shares * entry_price,
                'position_pct': (shares * entry_price / 10000) * 100,  # Assume 10k capital
                'risk_pct': 2.0,
                'risk_amount': shares * abs(entry_price - stop_loss),
                'confidence_adjustment': 1.0,
                'profile_tolerance': 'Moderate'
            }
    
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
        reasoning: str,
        use_risk_profile_sizing: bool = True
    ) -> Optional[str]:
        """
        Queue a trade for Discord approval
        
        Args:
            use_risk_profile_sizing: If True, override quantity with risk profile calculation
        
        Returns:
            Approval ID if queued successfully, None otherwise
        """
        try:
            # Calculate position size from risk profile if enabled
            if use_risk_profile_sizing:
                sizing = self.calculate_position_size(
                    symbol=symbol,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    confidence=confidence
                )
                quantity = sizing['recommended_shares']
                position_value = sizing['recommended_value']
                risk_info = f"AI-Sized: {quantity} shares @ ${position_value:,.2f} ({sizing['position_pct']:.1f}% of portfolio, {sizing['risk_pct']:.1f}% risk)"
            else:
                position_value = quantity * entry_price
                risk_info = f"Manual: {quantity} shares @ ${position_value:,.2f}"
            
            decision = StockTradeDecision(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=strategy,
                confidence=confidence,
                reasoning=f"{risk_info} | {reasoning}"
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
                
                self.discord_approval_manager.send_approval_request(
                    approval_id=approval_id,
                    pair=symbol,
                    side=side,
                    entry_price=entry_price,
                    position_size=position_value,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy=strategy,
                    confidence=confidence,
                    reasoning=decision.reasoning,
                    additional_info=f"📊 **Stock Trade** | Paper: {self.paper_mode} | R:R {rr_ratio:.2f}"
                )
                logger.info(f"📨 Stock trade approval sent to Discord: {symbol} {side} ({quantity} shares)")
            else:
                logger.warning(f"⚠️ Discord not available, trade queued locally: {approval_id}")
            
            return approval_id
            
        except Exception as e:
            logger.error(f"Error queuing trade for approval: {e}")
            return None
    
    def execute_trade(self, decision: StockTradeDecision, skip_approval: bool = False, skip_ai_validation: bool = False) -> bool:
        """
        Execute a stock trade via broker adapter
        
        Args:
            decision: Trade decision to execute
            skip_approval: If True, skip manual approval (already approved)
            skip_ai_validation: If True, skip final AI validation check
            
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
            
            # 🥊 AI PRE-TRADE VALIDATION (Final Risk Check)
            if not skip_ai_validation and self.llm_analyzer:
                logger.info("🥊 AI PRE-TRADE VALIDATION - Final Risk Check")
                approved, confidence, reasoning = self._ai_validate_trade(decision)
                
                if not approved:
                    logger.warning(f"🚫 AI VALIDATION REJECTED trade for {symbol}")
                    logger.warning(f"   Reason: {reasoning}")
                    return False
                elif confidence < self.min_confidence:
                    logger.warning(f"⚠️ AI confidence too low ({confidence:.0f}% < {self.min_confidence}%)")
                    logger.warning(f"   Reason: {reasoning}")
                    return False
                else:
                    logger.info(f"✅ AI VALIDATION APPROVED: {symbol} (confidence: {confidence:.0f}%)")
            
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
            
            # Log to unified trade journal (matching crypto position manager)
            self._journal_trade_entry(position, decision)
            
            logger.info(f"   📊 Position added to tracking")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return False
    
    def _ai_validate_trade(self, decision: StockTradeDecision) -> Tuple[bool, float, str]:
        """Final AI validation before trade execution (PUNCH 2)"""
        try:
            if not self.llm_analyzer:
                return True, 100.0, "AI validation skipped - no analyzer"
            
            risk_reward = abs(decision.take_profit - decision.entry_price) / max(abs(decision.entry_price - decision.stop_loss), 0.01)
            stop_pct = abs(decision.entry_price - decision.stop_loss) / decision.entry_price * 100
            
            prompt = f"""🥊 FINAL PRE-TRADE VALIDATION
Symbol: {decision.symbol} | Side: {decision.side} | Qty: {decision.quantity}
Entry: ${decision.entry_price:.2f} | Stop: ${decision.stop_loss:.2f} ({stop_pct:.1f}%) | TP: ${decision.take_profit:.2f}
R:R = 1:{risk_reward:.2f} | Confidence: {decision.confidence:.0f}% | Strategy: {decision.strategy}

VALIDATE: Risk:Reward (min 1:1.5), Stop distance (2-8%), Position size.
RESPOND: APPROVED: YES/NO | CONFIDENCE: 0-100 | REASONING: brief"""

            response = self.llm_analyzer.analyze_with_llm(prompt) or ""
            approved = "APPROVED: YES" in response.upper() or ("YES" in response.upper() and "NO" not in response.upper())
            
            import re
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response, re.I)
            confidence = float(conf_match.group(1)) if conf_match else 75.0
            reasoning = response[:200] if response else "AI validation completed"
            
            return approved, confidence, reasoning
        except Exception as e:
            logger.error(f"AI validation error: {e}")
            return True, 70.0, f"Validation error - approved: {str(e)[:50]}"
    
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
                
                # Log exit to unified trade journal (matching crypto position manager)
                self._journal_trade_exit(position, reason)
                
                return True
            else:
                logger.error(f"   ❌ Failed to close position")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def _journal_trade_entry(self, position: StockPosition, decision: StockTradeDecision):
        """
        Log trade entry to unified trade journal (matching crypto position manager pattern)
        Ensures stock trades are properly journaled for reference and historical analysis
        """
        try:
            from services.unified_trade_journal import get_unified_journal, UnifiedTradeEntry, TradeType
            journal = get_unified_journal()
            
            # Calculate risk/reward metrics
            if position.side.upper() == "BUY":
                risk_pct = ((position.entry_price - position.stop_loss) / position.entry_price) * 100
                reward_pct = ((position.take_profit - position.entry_price) / position.entry_price) * 100
            else:
                risk_pct = ((position.stop_loss - position.entry_price) / position.entry_price) * 100
                reward_pct = ((position.entry_price - position.take_profit) / position.entry_price) * 100
            
            rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
            position_size_usd = position.quantity * position.entry_price
            
            # Determine broker
            broker = "UNKNOWN"
            if self.broker_adapter:
                broker_type = type(self.broker_adapter).__name__
                if "Tradier" in broker_type:
                    broker = "TRADIER"
                elif "IBKR" in broker_type:
                    broker = "IBKR"
            if self.paper_mode:
                broker = f"{broker}_PAPER"
            
            entry = UnifiedTradeEntry(
                trade_id=position.trade_id,
                trade_type=TradeType.STOCK.value,
                symbol=position.symbol,
                side=position.side,
                entry_time=position.entry_time,
                entry_price=position.entry_price,
                quantity=float(position.quantity),
                position_size_usd=position_size_usd,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                risk_pct=risk_pct,
                reward_pct=reward_pct,
                risk_reward_ratio=rr_ratio,
                strategy=position.strategy,
                setup_type=decision.strategy if hasattr(decision, 'strategy') else None,
                ai_managed=True,
                broker=broker,
                order_id=position.order_id,
                notes=position.reasoning[:500] if position.reasoning else None,
                status="OPEN"
            )
            
            journal.log_trade_entry(entry)
            logger.info(f"   📝 Logged STOCK trade to unified journal: {position.symbol}")
            
        except Exception as e:
            logger.debug(f"Could not log stock trade to unified journal: {e}")
    
    def _journal_trade_exit(self, position: StockPosition, exit_reason: str):
        """
        Update unified trade journal with exit information (matching crypto position manager pattern)
        Records the exit price, P&L, and reason for trade closure
        """
        try:
            from services.unified_trade_journal import get_unified_journal
            journal = get_unified_journal()
            
            # Get market conditions for exit (optional enhancement)
            market_conditions = None
            try:
                # Try to get current technical indicators
                import yfinance as yf
                ticker = yf.Ticker(position.symbol)
                hist = ticker.history(period='5d')
                if not hist.empty and len(hist) >= 2:
                    # Calculate simple RSI approximation
                    deltas = hist['Close'].diff()
                    gains = deltas.where(deltas > 0, 0).mean()
                    losses = (-deltas.where(deltas < 0, 0)).mean()
                    rs = gains / losses if losses != 0 else 1
                    rsi = 100 - (100 / (1 + rs))
                    
                    market_conditions = {
                        'rsi': rsi,
                        'trend': 'UPTREND' if hist['Close'].iloc[-1] > hist['Close'].iloc[-5] else 'DOWNTREND'
                    }
            except Exception as e:
                logger.debug(f"Could not get market conditions for exit: {e}")
            
            journal.update_trade_exit(
                trade_id=position.trade_id,
                exit_price=position.current_price,
                exit_time=datetime.now(),
                exit_reason=exit_reason,
                market_conditions=market_conditions
            )
            logger.info(f"   📝 Updated STOCK trade exit in unified journal: {position.symbol}")
            
        except Exception as e:
            logger.debug(f"Could not update stock trade exit in journal: {e}")
    
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
    
    def sync_with_broker(self) -> Dict[str, int]:
        """
        Sync AI monitor positions with current broker positions (Tradier/IBKR)
        Removes positions not in broker, adds missing positions
        
        Returns:
            Dict with 'removed', 'added', 'kept' counts
        """
        try:
            logger.info("🔄 Syncing with broker portfolio...")
            
            if not self.broker_adapter:
                logger.warning("⚠️ No broker adapter configured - cannot sync positions")
                return {'removed': 0, 'added': 0, 'kept': len(self.positions)}
            
            # Fetch current broker positions
            success, broker_positions = self.broker_adapter.get_positions()
            
            if not success:
                logger.error("❌ Failed to fetch broker positions")
                return {'removed': 0, 'added': 0, 'kept': len(self.positions)}
            
            logger.info(f"   Found {len(broker_positions)} position(s) in broker")
            
            # Create a map of broker positions by symbol
            broker_symbols = {pos['symbol'].upper(): pos for pos in broker_positions}
            
            removed_count = 0
            kept_count = 0
            
            # Remove AI positions that no longer exist in broker
            positions_to_remove = []
            for trade_id, ai_pos in self.positions.items():
                symbol = ai_pos.symbol.upper()
                if symbol not in broker_symbols:
                    positions_to_remove.append(trade_id)
                    logger.info(f"   📤 Removing {ai_pos.symbol} - No longer in broker portfolio")
                else:
                    kept_count += 1
            
            # Actually remove the positions
            for trade_id in positions_to_remove:
                if trade_id in self.positions:
                    del self.positions[trade_id]
                    removed_count += 1
            
            # Add new positions from broker that aren't being monitored
            added_count = 0
            existing_symbols = {p.symbol.upper() for p in self.positions.values()}
            
            for symbol, broker_pos in broker_symbols.items():
                if symbol not in existing_symbols:
                    try:
                        quantity = abs(int(broker_pos.get('quantity', 0)))
                        if quantity == 0:
                            continue
                        
                        # Determine side based on quantity sign
                        side = "BUY" if broker_pos.get('quantity', 0) > 0 else "SELL"
                        
                        # Get cost basis and current price
                        cost_basis = float(broker_pos.get('cost_basis', 0))
                        current_price = float(broker_pos.get('current_price', 0))
                        
                        # Calculate entry price from cost basis
                        entry_price = cost_basis / quantity if quantity > 0 else current_price
                        if entry_price <= 0:
                            entry_price = current_price
                        
                        # Calculate default stops (2% stop, 4% target)
                        if side == "BUY":
                            stop_loss = entry_price * 0.98
                            take_profit = entry_price * 1.04
                        else:
                            stop_loss = entry_price * 1.02
                            take_profit = entry_price * 0.96
                        
                        trade_id = f"BROKER_{symbol}_{int(time.time())}"
                        
                        position = StockPosition(
                            trade_id=trade_id,
                            symbol=symbol,
                            side=side,
                            quantity=quantity,
                            entry_price=entry_price,
                            entry_time=datetime.now(),
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            strategy="SYNCED_FROM_BROKER",
                            current_price=current_price if current_price > 0 else entry_price,
                            highest_price=max(entry_price, current_price) if current_price > 0 else entry_price,
                            lowest_price=min(entry_price, current_price) if current_price > 0 else entry_price,
                            status=StockPositionStatus.ACTIVE.value
                        )
                        
                        self.positions[trade_id] = position
                        added_count += 1
                        
                        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                        pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                        logger.info(f"   📥 Added {symbol}: {quantity} shares @ ${entry_price:.2f} {pnl_emoji} ({pnl_pct:+.1f}%)")
                        
                    except Exception as e:
                        logger.warning(f"   Failed to add broker position {symbol}: {e}")
            
            # Save state after sync
            self._save_state()
            
            logger.info(f"✅ Sync complete: {added_count} added, {removed_count} removed, {kept_count} kept")
            return {'removed': removed_count, 'added': added_count, 'kept': kept_count}
            
        except Exception as e:
            logger.error(f"Error syncing with broker: {e}")
            return {'removed': 0, 'added': 0, 'kept': len(self.positions)}
    
    def get_broker_positions(self) -> List[Dict]:
        """
        Get current positions directly from broker
        
        Returns:
            List of position dicts from broker
        """
        if not self.broker_adapter:
            logger.warning("No broker adapter configured")
            return []
        
        success, positions = self.broker_adapter.get_positions()
        if success:
            return positions
        return []
    
    def start_monitoring(self):
        """Start background position monitoring (alias for start_monitoring_loop)"""
        self.start_monitoring_loop()
    
    def start_monitoring_loop(self):
        """
        Start the monitoring loop in a background thread
        Checks positions every check_interval_seconds
        """
        if self.is_running:
            logger.warning("Stock position monitoring already running")
            return
        
        def monitoring_loop():
            self.is_running = True
            cycle_count = 0
            logger.info("=" * 60)
            logger.info("🚀 AI STOCK POSITION MANAGER MONITORING STARTED")
            logger.info("=" * 60)
            
            while self.is_running:
                try:
                    cycle_count += 1
                    logger.info("")
                    logger.info(f"━━━ CHECK CYCLE #{cycle_count} ━━━")
                    
                    # Get active positions
                    active_positions = self.get_active_positions()
                    
                    if active_positions:
                        logger.info(f"📊 Checking {len(active_positions)} active position(s)...")
                        self._check_positions(active_positions)
                    else:
                        logger.info("📭 No active positions to monitor")
                    
                    # Periodic broker sync (every 10 cycles)
                    if cycle_count % 10 == 0:
                        logger.info("🔄 Periodic broker sync...")
                        self.sync_with_broker()
                    
                    logger.info(f"💤 Next check in {self.check_interval_seconds}s...")
                    time.sleep(self.check_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                    time.sleep(self.check_interval_seconds)
        
        self.thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.thread.start()
        logger.info("🎯 Stock position monitoring thread started")
    
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
    
    def get_status(self) -> Dict:
        """
        Get current status of the position manager
        
        Returns:
            Dict with status information
        """
        active_positions = self.get_active_positions()
        
        total_value = 0
        total_pnl = 0
        
        for pos in active_positions:
            value = pos.quantity * pos.current_price if pos.current_price > 0 else 0
            total_value += value
            
            if pos.entry_price > 0 and pos.current_price > 0:
                pnl_pct = self._calculate_pnl_pct(pos)
                pnl_amount = (pnl_pct / 100) * (pos.quantity * pos.entry_price)
                total_pnl += pnl_amount
        
        return {
            'is_running': self.is_running,
            'paper_mode': self.paper_mode,
            'total_positions': len(self.positions),
            'active_positions': len(active_positions),
            'pending_approvals': len(self.pending_approvals),
            'check_interval_seconds': self.check_interval_seconds,
            'broker_connected': self.broker_adapter is not None,
            'discord_enabled': self.discord_approval_manager is not None,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'stats': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            }
        }


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
            
            # Check for paper or production mode (matching trading_config.py)
            paper_mode = os.getenv('STOCK_PAPER_MODE', 'true').lower() == 'true'
            
            if paper_mode:
                # Paper trading settings (TWS paper port)
                port = int(os.getenv('IBKR_PAPER_PORT', '7497'))
                client_id = int(os.getenv('IBKR_PAPER_CLIENT_ID', '1'))
            else:
                # Live trading settings (TWS live port)
                port = int(os.getenv('IBKR_LIVE_PORT', '7496'))
                client_id = int(os.getenv('IBKR_LIVE_CLIENT_ID', '2'))
            
            logger.info(f"🔌 Connecting to IBKR ({'PAPER' if paper_mode else 'LIVE'} mode) on port {port}...")
            
            client = IBKRClient(port=port, client_id=client_id)
            if client.connect():
                logger.info(f"✅ IBKR client connected ({'PAPER' if paper_mode else 'LIVE'} mode)")
                return IBKRAdapter(client)
            else:
                logger.warning(f"Failed to connect to IBKR on port {port}")
                return None
                
        elif broker_type == 'TRADIER':
            from src.integrations.tradier_client import TradierClient
            from src.integrations.broker_adapter import TradierAdapter
            
            # Check for paper or production credentials (matching trading_config.py)
            paper_mode = os.getenv('STOCK_PAPER_MODE', 'true').lower() == 'true'
            
            if paper_mode:
                # Paper trading credentials
                account_id = os.getenv('TRADIER_PAPER_ACCOUNT_ID') or os.getenv('TRADIER_ACCOUNT_ID')
                access_token = os.getenv('TRADIER_PAPER_ACCESS_TOKEN') or os.getenv('TRADIER_ACCESS_TOKEN')
            else:
                # Production credentials
                account_id = os.getenv('TRADIER_PROD_ACCOUNT_ID')
                access_token = os.getenv('TRADIER_PROD_ACCESS_TOKEN')
            
            if access_token and account_id:
                client = TradierClient(
                    account_id=account_id,
                    access_token=access_token
                )
                logger.info(f"✅ Tradier client created ({'PAPER' if paper_mode else 'LIVE'} mode)")
                return TradierAdapter(client)
            else:
                logger.warning(f"Tradier credentials not configured (account_id: {bool(account_id)}, token: {bool(access_token)})")
                return None
        
        return None
        
    except Exception as e:
        logger.warning(f"Could not create broker adapter: {e}")
        return None

