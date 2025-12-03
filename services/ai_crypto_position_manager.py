"""
AI-Enhanced Crypto Position Manager
Actively monitors crypto positions and uses AI to make intelligent exit decisions

Features:
- Real-time position monitoring (every 60 seconds)
- AI-powered exit decision making
- Dynamic stop loss and take profit adjustments
- Multi-factor trend analysis
- Trailing stops and break-even protection
- Partial profit taking strategies
- 24/7 monitoring for crypto markets
"""

from loguru import logger
import time
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

from services.trading_agents import TradingAgentOrchestrator, CoordinatorDecisionContext


def log_and_print(message: str, level: str = "INFO"):
    """Log message AND print to stdout to ensure visibility in service logs"""
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    else:
        logger.debug(message)
    
    # Also print to stdout for immediate visibility in tail -f
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} | {level:8} | {message}", flush=True)


class PositionAction(Enum):
    """AI-recommended actions for position management"""
    HOLD = "HOLD"
    TIGHTEN_STOP = "TIGHTEN_STOP"
    EXTEND_TARGET = "EXTEND_TARGET"
    TAKE_PARTIAL = "TAKE_PARTIAL"
    CLOSE_NOW = "CLOSE_NOW"
    MOVE_TO_BREAKEVEN = "MOVE_TO_BREAKEVEN"


class PositionStatus(Enum):
    """Position lifecycle status"""
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    ERROR = "ERROR"
    EXCLUDED = "EXCLUDED"


class PositionIntent(Enum):
    """
    User's intended holding strategy for a position.
    This affects how aggressively the AI suggests exits.
    """
    HODL = "HODL"          # Long-term hold - very tolerant of losses, minimal sell alerts
    SWING = "SWING"        # Medium-term trade - standard exit rules
    SCALP = "SCALP"        # Quick trade - tight stops, aggressive exits
    
    @classmethod
    def get_description(cls, intent: str) -> str:
        descriptions = {
            "HODL": "Long-term hold - Will ride through volatility. Only alert on catastrophic moves.",
            "SWING": "Swing trade - Patient mid-term approach with balanced alerts. Respects intended hold period.",
            "SCALP": "Quick trade - Tight stops, aggressive profit-taking."
        }
        return descriptions.get(intent, "Unknown intent")


@dataclass
class TradingStyleConfig:
    """
    Configurable thresholds per trading style.
    Controls how aggressive the AI is with exit/close recommendations.
    """
    # Minimum hold time before AI can suggest CLOSE_NOW (in hours)
    min_hold_hours_before_close: float
    # Minimum hold time before AI can suggest TIGHTEN_STOP (in hours)
    min_hold_hours_before_tighten: float
    # Alert cooldown - minimum time between repeated alerts for same action (in minutes)
    alert_cooldown_minutes: int
    # Loss threshold before exit suggestion (percentage)
    loss_threshold_pct: float
    # Profit threshold before take profit suggestion (percentage)
    profit_suggestion_pct: float
    # Whether to be more forward-looking (analyze trend continuation)
    forward_looking: bool
    # Description
    description: str


# Default configurations per trading style
TRADING_STYLE_CONFIGS = {
    PositionIntent.HODL.value: TradingStyleConfig(
        min_hold_hours_before_close=168.0,  # 1 week - very patient
        min_hold_hours_before_tighten=72.0,  # 3 days
        alert_cooldown_minutes=240,  # 4 hours between alerts
        loss_threshold_pct=30.0,  # Only alert on 30%+ loss
        profit_suggestion_pct=50.0,  # Suggest profit taking at 50%+
        forward_looking=True,
        description="Long-term hold: Very patient, minimal alerts, ride volatility"
    ),
    PositionIntent.SWING.value: TradingStyleConfig(
        min_hold_hours_before_close=4.0,  # 4 hours minimum
        min_hold_hours_before_tighten=2.0,  # 2 hours before tightening
        alert_cooldown_minutes=60,  # 1 hour between alerts
        loss_threshold_pct=12.0,  # Alert at 12%+ loss
        profit_suggestion_pct=15.0,  # Suggest profit taking at 15%+
        forward_looking=True,
        description="Mid-term swing: Patient approach, analyzes trend continuation"
    ),
    PositionIntent.SCALP.value: TradingStyleConfig(
        min_hold_hours_before_close=0.0,  # Immediate - quick trades
        min_hold_hours_before_tighten=0.0,  # Immediate
        alert_cooldown_minutes=15,  # 15 min between alerts
        loss_threshold_pct=3.0,  # Tight 3% loss threshold
        profit_suggestion_pct=5.0,  # Quick 5% profit taking
        forward_looking=False,
        description="Quick scalp: Aggressive exits, tight risk management"
    )
}


@dataclass
class MonitoredCryptoPosition:
    """Tracked crypto position with AI management"""
    trade_id: str
    pair: str
    side: str  # 'BUY' or 'SELL'
    volume: float
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
    
    # AI management
    trailing_stop_pct: float = 2.0  # 2% trailing stop
    breakeven_trigger_pct: float = 3.0  # Move to BE after 3% profit
    moved_to_breakeven: bool = False
    partial_exit_taken: bool = False
    partial_exit_pct: float = 0.0
    
    # Position Intent - Controls how aggressive AI is with sell alerts
    # HODL = very tolerant, SWING = balanced, SCALP = aggressive exits
    position_intent: str = PositionIntent.SWING.value
    
    # Status
    status: str = PositionStatus.ACTIVE.value
    exit_attempts: int = 0
    last_ai_action: str = PositionAction.HOLD.value
    last_ai_reasoning: str = ""
    last_ai_confidence: float = 0.0
    
    # Performance tracking
    max_favorable_pct: float = 0.0  # Max profit seen
    max_adverse_pct: float = 0.0  # Max drawdown seen
    ai_adjustment_count: int = 0
    
    # Alert cooldown tracking (prevents repeated alerts for same action)
    last_close_alert_time: Optional[datetime] = None
    last_tighten_alert_time: Optional[datetime] = None
    last_partial_alert_time: Optional[datetime] = None
    alert_suppressed_count: int = 0  # Track how many alerts were suppressed
    
    # Order IDs for Kraken
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None


@dataclass
class AITradeDecision:
    """AI recommendation for position management"""
    action: str  # PositionAction value
    confidence: float  # 0-100
    reasoning: str
    urgency: str  # LOW, MEDIUM, HIGH
    new_stop: Optional[float] = None
    new_target: Optional[float] = None
    partial_pct: Optional[float] = None
    technical_score: float = 0.0
    trend_score: float = 0.0
    risk_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


def _load_service_interval(service_name: str = "sentient-crypto-ai-trader") -> int:
    """Load the check interval from service config file"""
    import json
    from pathlib import Path
    
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


class AICryptoPositionManager:
    """
    AI-Enhanced Position Manager for Crypto Trading
    Monitors positions 24/7 and makes intelligent exit decisions
    """
    
    def __init__(
        self,
        kraken_client,
        llm_analyzer=None,
        check_interval_seconds: Optional[int] = None,  # Now optional, loads from config
        enable_ai_decisions: bool = True,
        enable_trailing_stops: bool = True,
        enable_breakeven_moves: bool = True,
        enable_partial_exits: bool = True,
        min_ai_confidence: float = 65.0,
        state_file: str = "data/ai_crypto_positions.json",
        require_manual_approval: bool = True,  # SAFETY: Require manual approval for trades
        max_positions: int = 15  # Maximum concurrent monitored positions
    ):
        """
        Initialize AI Crypto Position Manager
        
        Args:
            kraken_client: KrakenClient instance
            llm_analyzer: LLM analyzer for AI decisions (LLMStrategyAnalyzer)
            check_interval_seconds: How often to check positions (default: loads from config or 60s)
            enable_ai_decisions: Enable AI-powered exit decisions
            enable_trailing_stops: Enable trailing stop feature
            enable_breakeven_moves: Enable break-even stop moves
            enable_partial_exits: Enable partial profit taking
            min_ai_confidence: Minimum AI confidence to act (0-100)
            state_file: File to persist position state
            require_manual_approval: SAFETY - Require user approval before executing trades (default: True)
            max_positions: Maximum concurrent positions to monitor (default: 15)
        """
        self.kraken_client = kraken_client
        self.llm_analyzer = llm_analyzer
        
        # Load interval from config file if not provided
        if check_interval_seconds is None:
            check_interval_seconds = _load_service_interval()
            logger.info(f"📂 Loaded check interval from config: {check_interval_seconds}s")
        self.check_interval_seconds = check_interval_seconds
        
        self.enable_ai_decisions = enable_ai_decisions
        self.enable_trailing_stops = enable_trailing_stops
        self.enable_breakeven_moves = enable_breakeven_moves
        self.enable_partial_exits = enable_partial_exits
        self.min_ai_confidence = min_ai_confidence
        self.state_file = state_file
        self.require_manual_approval = require_manual_approval
        
        # Use absolute path for state file to ensure persistence
        if state_file == "data/ai_crypto_positions.json":
            from pathlib import Path
            # Calculate absolute path: services/.. -> data/ai_crypto_positions.json
            # This works regardless of where the script is run from
            base_dir = Path(__file__).parent.parent
            self.state_file = str(base_dir / "data" / "ai_crypto_positions.json")
            logger.info(f"📂 Using absolute state file path: {self.state_file}")
        else:
            self.state_file = state_file
            
        # Monitored positions
        self.positions: Dict[str, MonitoredCryptoPosition] = {}
        
        # Excluded pairs (user opted out of monitoring)
        self.excluded_pairs: set = set()
        
        # SAFETY: Pending approvals queue
        self.pending_approvals: Dict[str, Dict] = {}  # {approval_id: {trade_id, decision, timestamp}}
        
        # Discord approval integration
        self.discord_approval_manager = None
        self._init_discord_approval()
        
        # State management
        self.is_running = False
        self.thread = None
        
        # Statistics
        self.total_ai_adjustments = 0
        self.trailing_stop_activations = 0
        self.breakeven_moves = 0
        self.partial_exits_taken = 0
        self.ai_exit_signals = 0
        
        # Safety
        self.max_positions = max_positions
        self.emergency_mode = False

        # Multi-agent coordinator (lazy instantiation)
        self.agent_orchestrator: Optional[TradingAgentOrchestrator] = None
        
        # Load persisted state
        self._load_state()
        
        logger.info("=" * 80)
        logger.info(" AI CRYPTO POSITION MANAGER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"   Check Interval: {check_interval_seconds}s")
        logger.info(f"   AI Decisions: {' ENABLED' if enable_ai_decisions else ' DISABLED'}")
        logger.info(f"   Trailing Stops: {' ENABLED' if enable_trailing_stops else ' DISABLED'}")
        logger.info(f"   Breakeven Moves: {' ENABLED' if enable_breakeven_moves else ' DISABLED'}")
        logger.info(f"   Partial Exits: {' ENABLED' if enable_partial_exits else ' DISABLED'}")
        logger.info(f"   Min AI Confidence: {min_ai_confidence}%")
        logger.info(f"   Max Positions: {max_positions}")
        logger.info("   SAFETY - MANUAL APPROVAL: {}", str(' REQUIRED' if require_manual_approval else ' AUTO-EXECUTE (DANGEROUS!)'))
        logger.info(f"   Discord Approval: {'✅ ENABLED' if self.discord_approval_manager else '❌ NOT CONFIGURED'}")
        logger.info("=" * 80)
    
    def _init_discord_approval(self):
        """Initialize Discord approval manager for trade approvals"""
        try:
            from services.discord_trade_approval import get_discord_approval_manager
            
            # Create callback for when trades are approved/rejected via Discord
            def on_discord_approval(approval_id: str, approved: bool):
                self._handle_discord_approval(approval_id, approved)
            
            self.discord_approval_manager = get_discord_approval_manager(approval_callback=on_discord_approval)
            
            if self.discord_approval_manager and self.discord_approval_manager.enabled:
                logger.info("✅ Discord approval integration enabled")
            else:
                logger.warning("⚠️ Discord approval not configured - will use local approval queue")
                self.discord_approval_manager = None
                
        except Exception as e:
            logger.warning(f"Could not initialize Discord approval: {e}")
            self.discord_approval_manager = None
    
    def _handle_discord_approval(self, approval_id: str, approved: bool):
        """Handle approval/rejection from Discord"""
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"📨 DISCORD APPROVAL RECEIVED")
        logger.info("=" * 60)
        logger.info(f"   Approval ID: {approval_id}")
        logger.info(f"   Decision: {'✅ APPROVED' if approved else '❌ REJECTED'}")
        
        # Find the pending approval
        if approval_id not in self.pending_approvals:
            logger.warning(f"   ⚠️ Approval ID not found in pending queue!")
            logger.warning(f"   Current pending: {list(self.pending_approvals.keys())}")
            return
        
        pending = self.pending_approvals[approval_id]
        trade_id = pending['trade_id']
        decision = pending['decision']
        pair = pending.get('pair', trade_id)
        
        logger.info(f"   Pair: {pair}")
        logger.info(f"   Action: {decision.action}")
        logger.info(f"   Confidence: {decision.confidence:.0f}%")
        
        if approved:
            logger.info("")
            logger.info(f"🚀 EXECUTING APPROVED TRADE: {pair}")
            # Execute with skip_approval=True since it's been approved via Discord
            result = self.execute_decision(trade_id, decision, skip_approval=True)
            if result:
                logger.info(f"   ✅ Trade executed successfully!")
            else:
                logger.error(f"   ❌ Trade execution failed!")
        else:
            logger.info(f"   Trade cancelled - no action taken")
        
        # Remove from pending
        del self.pending_approvals[approval_id]
        logger.info(f"   Removed from pending queue (remaining: {len(self.pending_approvals)})")
        logger.info("=" * 60)
    
    def add_position(
        self,
        trade_id: str,
        pair: str,
        side: str,
        volume: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str,
        entry_order_id: Optional[str] = None,
        trailing_stop_pct: float = 2.0,
        breakeven_trigger_pct: float = 3.0
    ) -> bool:
        """
        Add a position to AI monitoring
        
        Args:
            trade_id: Unique trade identifier
            pair: Trading pair (e.g., 'BTC/USD')
            side: 'BUY' or 'SELL'
            volume: Position size
            entry_price: Entry price
            stop_loss: Initial stop loss price
            take_profit: Initial take profit price
            strategy: Strategy name
            entry_order_id: Kraken order ID
            trailing_stop_pct: Trailing stop percentage (default: 2%)
            breakeven_trigger_pct: Move to breakeven after this % profit (default: 3%)
        
        Returns:
            True if added successfully
        """
        try:
            # Validate
            if not trade_id or not pair:
                logger.error("Invalid position parameters")
                return False
            
            # Check if excluded
            if pair in self.excluded_pairs:
                logger.warning(f"Cannot add {pair} - it is in the excluded list")
                return False
            
            if len(self.positions) >= self.max_positions:
                logger.warning(f"Max positions ({self.max_positions}) reached, cannot add {pair}")
                return False
            
            # Create position
            position = MonitoredCryptoPosition(
                trade_id=trade_id,
                pair=pair,
                side=side,
                volume=volume,
                entry_price=entry_price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=strategy,
                current_price=entry_price,
                highest_price=entry_price,
                lowest_price=entry_price,
                last_check_time=datetime.now(),
                trailing_stop_pct=trailing_stop_pct,
                breakeven_trigger_pct=breakeven_trigger_pct,
                entry_order_id=entry_order_id,
                status=PositionStatus.ACTIVE.value
            )
            
            self.positions[trade_id] = position
            
            # Calculate risk/reward
            if side == 'BUY':
                risk_pct = ((entry_price - stop_loss) / entry_price) * 100
                reward_pct = ((take_profit - entry_price) / entry_price) * 100
            else:
                risk_pct = ((stop_loss - entry_price) / entry_price) * 100
                reward_pct = ((entry_price - take_profit) / entry_price) * 100
            
            rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
            
            logger.info("=" * 80)
            logger.info(f" AI MONITORING STARTED: {pair}")
            logger.info("=" * 80)
            logger.info(f"   Trade ID: {trade_id}")
            logger.info(f"   Position: {side} {volume:.6f} @ ${entry_price:,.2f}")
            logger.info(f"   Stop Loss: ${stop_loss:,.2f} ({-risk_pct:.2f}%)")
            logger.info(f"   Take Profit: ${take_profit:,.2f} (+{reward_pct:.2f}%)")
            pass  # logger.info(f"   Risk/Reward: {}:1 {rr_ratio:.2f}")
            logger.info(f"   Strategy: {strategy}")
            logger.info(f"   Trailing Stop: {trailing_stop_pct}%")
            logger.info(f"   Breakeven Trigger: {breakeven_trigger_pct}%")
            logger.info("=" * 80)
            
            # Save state
            self._save_state()
            
            # Send Discord notification
            self._send_notification(
                f" AI Monitoring Started: {pair}",
                f"**{side}** {volume:.6f} @ ${entry_price:,.2f}\n"
                f"**Stop:** ${stop_loss:,.2f} | **Target:** ${take_profit:,.2f}\n"
                f"**R:R:** {rr_ratio:.2f}:1 | **Strategy:** {strategy}",
                color=3447003  # Blue
            )
            
            # Log to unified journal
            try:
                from services.unified_trade_journal import get_unified_journal, UnifiedTradeEntry, TradeType
                journal = get_unified_journal()
                
                entry = UnifiedTradeEntry(
                    trade_id=trade_id,
                    trade_type=TradeType.CRYPTO.value,
                    symbol=pair,
                    side=side,
                    entry_time=datetime.now(),
                    entry_price=entry_price,
                    quantity=volume,
                    position_size_usd=entry_price * volume,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_pct=risk_pct,
                    reward_pct=reward_pct,
                    risk_reward_ratio=rr_ratio,
                    strategy=strategy,
                    ai_managed=True,
                    broker="KRAKEN",
                    order_id=entry_order_id,
                    status="OPEN"
                )
                
                journal.log_trade_entry(entry)
                logger.debug(f" Logged trade to unified journal: {trade_id}")
            except Exception as e:
                logger.debug(f"Could not log to unified journal: {e}")
            
            # Store in Signal Memory for pattern learning (RAG)
            try:
                from services.signal_memory import store_crypto_signal
                
                signal_id = store_crypto_signal(
                    symbol=pair,
                    strategy=strategy,
                    signal_type=side,
                    confidence=75.0,  # Default confidence for manual trades
                    price=entry_price,
                    rsi=50.0,  # Will be updated with real values when available
                    volume_ratio=1.0,
                    change_24h=0.0,
                    trade_id=trade_id
                )
                
                if signal_id:
                    logger.info(f"🧠 Stored crypto signal in memory for pattern learning: {pair}")
                    # Store signal_id in position for outcome tracking
                    position.signal_memory_id = signal_id if hasattr(position, 'signal_memory_id') else None
            except Exception as mem_err:
                logger.debug(f"Could not store signal in memory (non-critical): {mem_err}")
            
            return True
            
        except Exception as e:
            logger.error("Error adding position {trade_id}: {}", str(e), exc_info=True)
            return False
    
    def remove_position(self, trade_id: str, reason: str = "Manual") -> bool:
        """Remove a position from monitoring"""
        if trade_id in self.positions:
            position = self.positions[trade_id]
            position.status = PositionStatus.CLOSED.value
            del self.positions[trade_id]
            
            log_and_print(f"📤 Removed {position.pair} from AI monitoring (Reason: {reason})")
            self._save_state()
            return True
        return False
    
    def monitor_positions(self) -> List[tuple]:
        """
        Check all monitored positions and get AI recommendations
        
        Returns:
            List of (AITradeDecision, trade_id) tuples for positions requiring action
        """
        decisions: List[tuple] = []
        
        if not self.positions:
            return decisions
        
        active_positions = [(tid, p) for tid, p in self.positions.items() if p.status == PositionStatus.ACTIVE.value]
        log_and_print(f"   Analyzing {len(active_positions)} active position(s)...")
        
        for trade_id, position in active_positions:
            try:
                log_and_print(f"")
                log_and_print(f"   ━━━ {position.pair} ━━━")
                
                # Skip futures/staking positions (not supported for spot trading)
                pair_parts = position.pair.split('/')
                if len(pair_parts) > 0:
                    base_currency = pair_parts[0]
                    if any(base_currency.endswith(suffix) for suffix in ['.F', '.S', '.M', '.P']):
                        log_and_print(f"   ⏭️ Skipping {position.pair} - futures/staking not supported", "WARNING")
                        position.status = PositionStatus.CLOSED.value
                        position.exit_time = datetime.now()
                        position.exit_reason = "Futures/staking contract not supported"
                        continue
                
                # Get current price
                log_and_print(f"   📡 Fetching current price...")
                ticker_info = self.kraken_client.get_ticker_info(position.pair)
                if not ticker_info:
                    log_and_print(f"   ⚠️ Failed to get ticker for {position.pair}", "WARNING")
                    continue
                
                current_price = float(ticker_info.get('c', [0])[0])
                if current_price <= 0:
                    log_and_print(f"   ⚠️ Invalid price: {current_price}", "WARNING")
                    continue
                
                # Update position tracking
                old_price = position.current_price
                position.current_price = current_price
                position.last_check_time = datetime.now()
                position.highest_price = max(position.highest_price, current_price)
                position.lowest_price = min(position.lowest_price, current_price) if position.lowest_price > 0 else current_price
                
                # Calculate P&L
                if position.side == 'BUY':
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                else:
                    pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                
                # Track max favorable/adverse
                position.max_favorable_pct = max(position.max_favorable_pct, pnl_pct)
                if pnl_pct < 0:
                    position.max_adverse_pct = min(position.max_adverse_pct, pnl_pct)
                
                # Log position status
                price_change = ((current_price - old_price) / old_price * 100) if old_price > 0 else 0
                pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                log_and_print(f"   💰 Price: ${current_price:,.6f} ({price_change:+.2f}% since last check)")
                log_and_print(f"   {pnl_emoji} P&L: {pnl_pct:+.2f}% (Max: {position.max_favorable_pct:+.2f}%, Min: {position.max_adverse_pct:+.2f}%)")
                log_and_print(f"   🎯 Stop: ${position.stop_loss:,.6f} | Target: ${position.take_profit:,.6f}")
                
                # Calculate distance to stop/target
                dist_to_stop = ((current_price - position.stop_loss) / current_price) * 100
                dist_to_target = ((position.take_profit - current_price) / current_price) * 100
                log_and_print(f"   📏 Distance: Stop {dist_to_stop:+.2f}% | Target {dist_to_target:+.2f}%")
                
                # 1. Check basic exit conditions (stop/target)
                log_and_print(f"   🔍 Checking exit conditions...")
                basic_action = self._check_basic_exit_conditions(position, current_price, pnl_pct)
                if basic_action:
                    decisions.append((basic_action, trade_id))  # Include trade_id with decision
                    log_and_print(f"   🚨 EXIT SIGNAL: {basic_action.action}")
                    log_and_print(f"      Reason: {basic_action.reasoning}")
                    continue
                
                # 2. Check breakeven move
                if self.enable_breakeven_moves:
                    be_action = self._check_breakeven_move(position, pnl_pct)
                    if be_action:
                        decisions.append((be_action, trade_id))  # Include trade_id with decision
                        log_and_print(f"   🛡️ BREAKEVEN MOVE triggered at {pnl_pct:+.2f}%")
                        continue
                    else:
                        if not position.moved_to_breakeven:
                            log_and_print(f"   🛡️ Breakeven: Not triggered (need {position.breakeven_trigger_pct}%+)")
                
                # 3. Check trailing stop
                if self.enable_trailing_stops and pnl_pct > 0:
                    trail_action = self._check_trailing_stop(position, current_price, pnl_pct)
                    if trail_action:
                        decisions.append((trail_action, trade_id))  # Include trade_id with decision
                        log_and_print(f"   📈 TRAILING STOP: Tightening to ${trail_action.new_stop:,.4f}")
                        continue
                    else:
                        log_and_print(f"   📈 Trailing: Active (trail: {position.trailing_stop_pct}%)")
                
                # 4. Get AI recommendation
                if self.enable_ai_decisions and self.llm_analyzer:
                    log_and_print(f"   🤖 Getting AI recommendation...")
                    ai_decision = self._get_ai_recommendation(position, current_price, pnl_pct)
                    
                    if ai_decision:
                        log_and_print(f"   🤖 AI says: {ai_decision.action} (confidence: {ai_decision.confidence:.0f}%)")
                        if ai_decision.confidence >= self.min_ai_confidence:
                            log_and_print(f"   ✅ Confidence meets threshold ({self.min_ai_confidence}%) - executing")
                            decisions.append((ai_decision, trade_id))  # Include trade_id with decision
                            
                            # Execute AI decision (will require approval if enabled)
                            result = self.execute_decision(position.trade_id, ai_decision)
                            if not result and self.require_manual_approval:
                                log_and_print(f"   ⏳ Sent to Discord for approval")
                            
                            # Update position with AI feedback
                            position.last_ai_action = ai_decision.action
                            position.last_ai_reasoning = ai_decision.reasoning
                            position.last_ai_confidence = ai_decision.confidence
                        else:
                            log_and_print(f"   ⏸️ Confidence below threshold - no action")
                    else:
                        log_and_print(f"   🤖 AI: No strong recommendation")
                else:
                    log_and_print(f"   🤖 AI decisions: {'Disabled' if not self.enable_ai_decisions else 'No LLM configured'}")
                
                log_and_print(f"   ✅ {position.pair} check complete")
                
            except Exception as e:
                log_and_print(f"   ❌ Error monitoring {trade_id}: {str(e)}", "ERROR")
                continue
        
        # Save state after check
        self._save_state()
        
        return decisions
    
    def _get_style_config(self, intent: str) -> TradingStyleConfig:
        """Get the trading style configuration for a given intent"""
        return TRADING_STYLE_CONFIGS.get(intent, TRADING_STYLE_CONFIGS[PositionIntent.SWING.value])
    
    def _get_hold_time_hours(self, position: MonitoredCryptoPosition) -> float:
        """Calculate how long a position has been held in hours"""
        if not position.entry_time:
            return 0.0
        return (datetime.now() - position.entry_time).total_seconds() / 3600
    
    def _should_suppress_alert(
        self,
        position: MonitoredCryptoPosition,
        action: str
    ) -> Tuple[bool, str]:
        """
        Check if an alert should be suppressed based on cooldown and minimum hold time.
        
        Returns:
            Tuple of (should_suppress, reason)
        """
        style_config = self._get_style_config(position.position_intent)
        hold_time_hours = self._get_hold_time_hours(position)
        
        # 1. Check minimum hold time for CLOSE_NOW
        if action == PositionAction.CLOSE_NOW.value:
            if hold_time_hours < style_config.min_hold_hours_before_close:
                return (True, f"Position held {hold_time_hours:.1f}h, need {style_config.min_hold_hours_before_close}h minimum for {position.position_intent} style")
            
            # Check cooldown since last close alert
            if position.last_close_alert_time:
                minutes_since_last = (datetime.now() - position.last_close_alert_time).total_seconds() / 60
                if minutes_since_last < style_config.alert_cooldown_minutes:
                    return (True, f"Cooldown: {minutes_since_last:.0f}m since last close alert (need {style_config.alert_cooldown_minutes}m)")
        
        # 2. Check minimum hold time for TIGHTEN_STOP
        elif action == PositionAction.TIGHTEN_STOP.value:
            if hold_time_hours < style_config.min_hold_hours_before_tighten:
                return (True, f"Position held {hold_time_hours:.1f}h, need {style_config.min_hold_hours_before_tighten}h minimum for tighten alerts")
            
            # Check cooldown since last tighten alert
            if position.last_tighten_alert_time:
                minutes_since_last = (datetime.now() - position.last_tighten_alert_time).total_seconds() / 60
                if minutes_since_last < style_config.alert_cooldown_minutes:
                    return (True, f"Cooldown: {minutes_since_last:.0f}m since last tighten alert (need {style_config.alert_cooldown_minutes}m)")
        
        # 3. Check cooldown for TAKE_PARTIAL
        elif action == PositionAction.TAKE_PARTIAL.value:
            if position.last_partial_alert_time:
                minutes_since_last = (datetime.now() - position.last_partial_alert_time).total_seconds() / 60
                if minutes_since_last < style_config.alert_cooldown_minutes:
                    return (True, f"Cooldown: {minutes_since_last:.0f}m since last partial alert (need {style_config.alert_cooldown_minutes}m)")
        
        return (False, "")
    
    def _record_alert_sent(self, position: MonitoredCryptoPosition, action: str):
        """Record that an alert was sent for cooldown tracking"""
        now = datetime.now()
        if action == PositionAction.CLOSE_NOW.value:
            position.last_close_alert_time = now
        elif action == PositionAction.TIGHTEN_STOP.value:
            position.last_tighten_alert_time = now
        elif action == PositionAction.TAKE_PARTIAL.value:
            position.last_partial_alert_time = now
    
    def _check_style_thresholds(
        self,
        position: MonitoredCryptoPosition,
        pnl_pct: float
    ) -> Optional[str]:
        """
        Check if P&L thresholds for the trading style have been breached.
        Returns a warning message if close to threshold, None otherwise.
        """
        style_config = self._get_style_config(position.position_intent)
        
        # Check loss threshold
        if pnl_pct < 0 and abs(pnl_pct) >= style_config.loss_threshold_pct * 0.8:
            if abs(pnl_pct) >= style_config.loss_threshold_pct:
                return f"⚠️ Loss threshold breached: {pnl_pct:.1f}% (threshold: -{style_config.loss_threshold_pct}%)"
            else:
                return f"📉 Approaching loss threshold: {pnl_pct:.1f}% (threshold: -{style_config.loss_threshold_pct}%)"
        
        # Check profit threshold
        if pnl_pct > 0 and pnl_pct >= style_config.profit_suggestion_pct * 0.8:
            if pnl_pct >= style_config.profit_suggestion_pct:
                return f"🎯 Profit target zone: +{pnl_pct:.1f}% (suggestion at +{style_config.profit_suggestion_pct}%)"
            else:
                return f"📈 Approaching profit target: +{pnl_pct:.1f}% (suggestion at +{style_config.profit_suggestion_pct}%)"
        
        return None
    
    def _check_basic_exit_conditions(
        self,
        position: MonitoredCryptoPosition,
        current_price: float,
        pnl_pct: float
    ) -> Optional[AITradeDecision]:
        """Check if stop loss or take profit hit, respecting position intent"""
        
        intent = position.position_intent
        
        # HODL intent: Much more lenient - only exit on catastrophic moves
        # Skip normal stop loss for HODL positions, only emergency exits
        if intent == PositionIntent.HODL.value:
            # For HODL, only trigger stop if it's a TRUE emergency (>30% crash)
            catastrophic_threshold = 0.30  # 30% loss threshold for HODL
            
            if pnl_pct <= -catastrophic_threshold * 100:
                log_and_print(f"   ⚠️ HODL position {position.pair} hit catastrophic loss ({pnl_pct:.1f}%)", "WARNING")
                return AITradeDecision(
                    action=PositionAction.CLOSE_NOW.value,
                    confidence=90.0,
                    reasoning=f"HODL position hit catastrophic loss of {pnl_pct:.1f}% - this exceeds normal volatility",
                    urgency="HIGH"
                )
            
            # For HODL, still check take profit but log that we're respecting intent
            if position.side == 'BUY' and current_price >= position.take_profit:
                return AITradeDecision(
                    action=PositionAction.CLOSE_NOW.value,
                    confidence=100.0,
                    reasoning=f"Take profit target reached at ${current_price:,.2f} - HODL goal achieved!",
                    urgency="MEDIUM"
                )
            
            # HODL positions ignore normal stop loss
            if current_price <= position.stop_loss:
                log_and_print(f"   💎 {position.pair}: Stop loss level reached but HODL intent active - holding through volatility")
            
            return None
        
        # Standard stop loss check for SWING/SCALP
        if position.side == 'BUY':
            if current_price <= position.stop_loss:
                return AITradeDecision(
                    action=PositionAction.CLOSE_NOW.value,
                    confidence=100.0,
                    reasoning=f"Stop loss hit at ${current_price:,.2f}",
                    urgency="HIGH"
                )
        else:  # SELL
            if current_price >= position.stop_loss:
                return AITradeDecision(
                    action=PositionAction.CLOSE_NOW.value,
                    confidence=100.0,
                    reasoning=f"Stop loss hit at ${current_price:,.2f}",
                    urgency="HIGH"
                )
        
        # Take profit check
        if position.side == 'BUY':
            if current_price >= position.take_profit:
                return AITradeDecision(
                    action=PositionAction.CLOSE_NOW.value,
                    confidence=100.0,
                    reasoning=f"Take profit target reached at ${current_price:,.2f}",
                    urgency="MEDIUM"
                )
        else:  # SELL
            if current_price <= position.take_profit:
                return AITradeDecision(
                    action=PositionAction.CLOSE_NOW.value,
                    confidence=100.0,
                    reasoning=f"Take profit target reached at ${current_price:,.2f}",
                    urgency="MEDIUM"
                )
        
        return None
    
    def _check_breakeven_move(
        self,
        position: MonitoredCryptoPosition,
        pnl_pct: float
    ) -> Optional[AITradeDecision]:
        """Check if we should move stop to breakeven"""
        
        if position.moved_to_breakeven:
            return None
        
        # Check if profit threshold reached
        if pnl_pct >= position.breakeven_trigger_pct:
            # Calculate new breakeven stop (entry price + small buffer for fees)
            fee_buffer = position.entry_price * 0.002  # 0.2% buffer for fees
            
            if position.side == 'BUY':
                new_stop = position.entry_price + fee_buffer
                # Only move up, never down
                if new_stop > position.stop_loss:
                    position.moved_to_breakeven = True
                    self.breakeven_moves += 1
                    return AITradeDecision(
                        action=PositionAction.MOVE_TO_BREAKEVEN.value,
                        confidence=100.0,
                        reasoning=f"Profit reached {pnl_pct:.2f}%, protecting capital with breakeven stop",
                        urgency="MEDIUM",
                        new_stop=new_stop
                    )
            else:  # SELL
                new_stop = position.entry_price - fee_buffer
                if new_stop < position.stop_loss:
                    position.moved_to_breakeven = True
                    self.breakeven_moves += 1
                    return AITradeDecision(
                        action=PositionAction.MOVE_TO_BREAKEVEN.value,
                        confidence=100.0,
                        reasoning=f"Profit reached {pnl_pct:.2f}%, protecting capital with breakeven stop",
                        urgency="MEDIUM",
                        new_stop=new_stop
                    )
        
        return None
    
    def _check_trailing_stop(
        self,
        position: MonitoredCryptoPosition,
        current_price: float,
        pnl_pct: float
    ) -> Optional[AITradeDecision]:
        """Check if trailing stop should be activated"""
        
        if position.side == 'BUY':
            # Calculate trailing stop from highest price
            trailing_stop_price = position.highest_price * (1 - position.trailing_stop_pct / 100)
            
            # Only tighten stop, never loosen
            if trailing_stop_price > position.stop_loss:
                self.trailing_stop_activations += 1
                return AITradeDecision(
                    action=PositionAction.TIGHTEN_STOP.value,
                    confidence=95.0,
                    reasoning=f"Trailing stop activated: {position.trailing_stop_pct}% from high ${position.highest_price:,.2f}",
                    urgency="LOW",
                    new_stop=trailing_stop_price
                )
        else:  # SELL
            # For short positions, trail from lowest price
            trailing_stop_price = position.lowest_price * (1 + position.trailing_stop_pct / 100)
            
            if trailing_stop_price < position.stop_loss:
                self.trailing_stop_activations += 1
                return AITradeDecision(
                    action=PositionAction.TIGHTEN_STOP.value,
                    confidence=95.0,
                    reasoning=f"Trailing stop activated: {position.trailing_stop_pct}% from low ${position.lowest_price:,.2f}",
                    urgency="LOW",
                    new_stop=trailing_stop_price
                )
        
        return None
    
    def _get_ai_recommendation(
        self,
        position: MonitoredCryptoPosition,
        current_price: float,
        pnl_pct: float
    ) -> Optional[AITradeDecision]:
        """
        Get AI-powered recommendation for position management (ENHANCED with real-time news)
        Uses LLM to analyze market conditions, news, and sentiment
        """
        try:
            # Get technical data
            technical_data = self._get_technical_indicators(position.pair)
            
            # Calculate hold time
            hold_time = (datetime.now() - position.entry_time).total_seconds() / 60
            
            # ENHANCED: Fetch real-time news and sentiment (last 2 hours)
            recent_news: List[Dict[str, Any]] = []
            sentiment_score: Optional[float] = None
            news_sentiment = None
            
            # Extract base asset from pair (e.g., "BTC/USD" -> "BTC")
            base_asset = position.pair.split('/')[0]
            
            # Prepare event loop (shared for multi-agent orchestration)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Try to get real-time news
            try:
                from services.crypto_news_analyzer import CryptoNewsAnalyzer
                news_analyzer = CryptoNewsAnalyzer(use_finbert=True)
                
                # Get comprehensive sentiment (last 2 hours for more current data)
                news_sentiment = loop.run_until_complete(
                    news_analyzer.analyze_comprehensive_sentiment(
                        base_asset,
                        include_social=False,  # Skip social for speed
                        hours=2  # Last 2 hours only
                    )
                )
                
                # Format news for prompt
                for article in news_sentiment.recent_news[:5]:
                    recent_news.append({
                        'timestamp': article.published_at.split('T')[1][:5] if 'T' in article.published_at else 'recent',  # HH:MM
                        'title': article.title,
                        'sentiment': article.sentiment,
                        'confidence': article.sentiment_confidence,
                        'impact': article.market_impact
                    })
                
                sentiment_score = news_sentiment.overall_sentiment_score
                
                if recent_news:
                    logger.info(
                        f"📰 Fetched {len(recent_news)} recent news articles for {base_asset} "
                        f"(sentiment: {sentiment_score:.1f}/100)"
                    )
                
            except Exception as e:
                logger.debug(f"Could not fetch real-time news for {base_asset}: {e}")
                news_sentiment = None  # Continue without news - not critical
            
            # Attempt multi-agent orchestration first
            multi_agent_decision = None
            try:
                if not hasattr(self, "agent_orchestrator") or self.agent_orchestrator is None:
                    self.agent_orchestrator = TradingAgentOrchestrator(self.llm_analyzer)
                
                if self.agent_orchestrator and self.llm_analyzer:
                    context = CoordinatorDecisionContext(
                        position=position,
                        current_price=current_price,
                        pnl_pct=pnl_pct,
                        hold_time_minutes=hold_time,
                        technical_data=technical_data,
                        recent_news=recent_news,
                        sentiment_score=sentiment_score,
                        news_sentiment=news_sentiment,
                    )
                    
                    multi_agent_decision = loop.run_until_complete(
                        self.agent_orchestrator.make_decision(context)
                    )
                    
                    if multi_agent_decision:
                        logger.info(
                            f"🤝 Multi-agent consensus: {position.pair} -> "
                            f"{multi_agent_decision.action} (confidence {multi_agent_decision.confidence:.1f}%)"
                        )
                        return multi_agent_decision
            except Exception as orchestration_error:
                logger.error(
                    f"Multi-agent orchestrator failed for {position.pair}: {orchestration_error}",
                    exc_info=True
                )
            
            # Fallback: Single-pass prompt (existing behavior)
            prompt = self._build_ai_prompt(
                position=position,
                current_price=current_price,
                pnl_pct=pnl_pct,
                hold_time=hold_time,
                technical_data=technical_data,
                recent_news=recent_news,
                sentiment_score=sentiment_score
            )
            
            # Get AI response
            if self.llm_analyzer is None:
                logger.warning("LLM analyzer not available")
                return None
            response = self.llm_analyzer.analyze_with_llm(prompt)
            
            # Parse JSON response
            decision_data = self._parse_ai_response(response)
            
            if decision_data:
                return AITradeDecision(
                    action=decision_data.get('action', 'HOLD'),
                    confidence=float(decision_data.get('confidence', 0)),
                    reasoning=decision_data.get('reasoning', ''),
                    urgency=decision_data.get('urgency', 'LOW'),
                    new_stop=decision_data.get('new_stop'),
                    new_target=decision_data.get('new_target'),
                    partial_pct=decision_data.get('partial_pct'),
                    technical_score=float(decision_data.get('technical_score', 0)),
                    trend_score=float(decision_data.get('trend_score', 0)),
                    risk_score=float(decision_data.get('risk_score', 0))
                )
            
        except Exception as e:
            logger.error("Error getting AI recommendation for {position.pair}: {}", str(e), exc_info=True)
        
        return None
    
    def _get_technical_indicators(self, pair: str) -> Dict:
        """Get technical indicators for AI analysis"""
        try:
            # Get OHLC data (1 hour candles, last 100)
            ohlc_data = self.kraken_client.get_ohlc_data(pair, interval=60, since=None)
            
            if not ohlc_data or len(ohlc_data) < 20:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data, columns=pd.Index(['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']))
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Calculate indicators
            indicators = {}
            
            # Moving averages
            indicators['ema_20'] = df['close'].ewm(span=20).mean().iloc[-1]
            indicators['ema_50'] = df['close'].ewm(span=50).mean().iloc[-1] if len(df) >= 50 else indicators['ema_20']
            
            # RSI
            delta = df['close'].diff().fillna(0).astype(float)
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_histogram'] = (macd - signal).iloc[-1]
            
            # Volume
            indicators['volume_avg'] = df['volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_current'] = df['volume'].iloc[-1]
            indicators['volume_change_pct'] = ((indicators['volume_current'] / indicators['volume_avg']) - 1) * 100
            
            # Support/Resistance (simplified)
            indicators['support'] = df['low'].tail(20).min()
            indicators['resistance'] = df['high'].tail(20).max()
            
            # Trend
            if indicators['ema_20'] > indicators['ema_50']:
                indicators['trend'] = 'BULLISH'
            elif indicators['ema_20'] < indicators['ema_50']:
                indicators['trend'] = 'BEARISH'
            else:
                indicators['trend'] = 'NEUTRAL'
            
            return indicators
            
        except Exception as e:
            logger.error("Error calculating indicators for {pair}: {}", str(e), exc_info=True)
            return {}
    
    def _build_ai_prompt(
        self,
        position: MonitoredCryptoPosition,
        current_price: float,
        pnl_pct: float,
        hold_time: float,
        technical_data: Dict,
        recent_news: Optional[List[Dict]] = None,
        sentiment_score: Optional[float] = None
    ) -> str:
        """
        Build prompt for AI analysis (ENHANCED with real-time news/sentiment)
        
        Args:
            recent_news: List of recent news articles with sentiment
            sentiment_score: Aggregate sentiment score (0-100)
        """
        
        # Get position intent and build intent-specific context
        intent = position.position_intent
        intent_emoji = {"HODL": "💎", "SWING": "🔄", "SCALP": "⚡"}.get(intent, "📊")
        
        # Intent-specific instructions that dramatically change AI behavior
        if intent == PositionIntent.HODL.value:
            intent_context = f"""
🚨 **CRITICAL: USER INTENT = {intent_emoji} HODL (LONG-TERM HOLD)** 🚨
The user has explicitly marked this as a LONG-TERM HOLD position. They are:
- INTENTIONALLY holding through volatility and temporary losses
- NOT interested in frequent sell/exit alerts
- Willing to wait for recovery rather than realize losses
- Only wants alerts for CATASTROPHIC situations (>30% crash, project failure, etc.)

**YOU MUST:**
- Default to HOLD unless there is a CRITICAL reason to exit
- Do NOT suggest CLOSE_NOW for normal market volatility
- Do NOT suggest CLOSE_NOW just because the position is in the red
- Only suggest exit if: major security breach, rug pull warning, extreme market crash (>25% in hours)
- Be supportive of their hold strategy, not alarming
"""
        elif intent == PositionIntent.SCALP.value:
            intent_context = f"""
**USER INTENT = {intent_emoji} SCALP (QUICK TRADE)**
The user is actively trading this position for quick profits. They want:
- Tight risk management and quick exits
- Aggressive profit-taking suggestions
- Low tolerance for drawdowns
- Fast action recommendations
"""
        else:  # SWING (default)
            # Get trading style config for hold time context
            style_config = self._get_style_config(PositionIntent.SWING.value)
            hold_hours = self._get_hold_time_hours(position)
            min_hold_hours = style_config.min_hold_hours_before_close
            
            intent_context = f"""
**USER INTENT = {intent_emoji} SWING (MID-TERM TRADE)**
The user is swing trading this position - this is a PATIENT, MID-TERM approach.

**SWING TRADING PHILOSOPHY:**
- Swing trades are meant to capture multi-day to multi-week moves
- Short-term volatility is EXPECTED and ACCEPTABLE
- Position has been held for {hold_hours:.1f} hours (minimum recommended: {min_hold_hours}h)
- Do NOT recommend exit just because of normal market fluctuations
- Analyze TREND CONTINUATION potential before suggesting exits
- Be FORWARD-LOOKING: Consider where price is likely to go, not just where it's been

**WHEN TO HOLD (Default for Swing):**
- Position is consolidating but trend is intact
- Temporary pullback within the larger trend
- No fundamental change in the asset
- Technical structure still supports the trade thesis

**WHEN TO CONSIDER EXIT:**
- Clear trend reversal (multiple timeframe confirmation)
- Break of key support/resistance that invalidates the setup
- Fundamental change (negative news with high confidence)
- Already exceeded original target significantly
"""
        
        prompt = f"""
Analyze this active crypto position with REAL-TIME MARKET CONTEXT and recommend the BEST action.

{intent_context}

**Position Details:**
- Asset: {position.pair} ({position.side} LONG position)
- Entry: ${position.entry_price:,.2f} at {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}
- Current: ${current_price:,.2f} (P&L: {pnl_pct:+.2f}%)
- Stop Loss: ${position.stop_loss:,.2f} | Take Profit: ${position.take_profit:,.2f}
- Hold Time: {hold_time:.1f} minutes
- Strategy: {position.strategy}
- Max Profit Seen: {position.max_favorable_pct:+.2f}%
- Max Drawdown: {position.max_adverse_pct:+.2f}%
- Breakeven Moved: {position.moved_to_breakeven}
- Partial Exit Taken: {position.partial_exit_taken}
- **Position Intent: {intent_emoji} {intent}**

**Technical Indicators:**"""
        
        if technical_data:
            prompt += f"""
- RSI: {technical_data.get('rsi', 'N/A'):.2f}
- MACD: {technical_data.get('macd', 0):.4f} (Signal: {technical_data.get('macd_signal', 0):.4f})
- EMA 20: ${technical_data.get('ema_20', 0):,.2f} | EMA 50: ${technical_data.get('ema_50', 0):,.2f}
- Volume Change: {technical_data.get('volume_change_pct', 0):+.1f}%
- Support: ${technical_data.get('support', 0):,.2f} | Resistance: ${technical_data.get('resistance', 0):,.2f}
- Trend: {technical_data.get('trend', 'NEUTRAL')}
"""
        else:
            prompt += "\n- Technical data unavailable (use price action)\n"
        
        # ENHANCED: Add real-time news and sentiment context
        if recent_news and len(recent_news) > 0:
            prompt += f"""
**🔥 BREAKING NEWS & MARKET SENTIMENT (Last 2 hours):**
"""
            for news in recent_news[:5]:  # Top 5 most recent
                timestamp = news.get('timestamp', 'recent')
                title = news.get('title', 'No title')
                sentiment = news.get('sentiment', 'NEUTRAL')
                confidence = news.get('confidence', 0.0)
                impact = news.get('impact', 'UNKNOWN')
                
                # Add emoji indicators
                sentiment_emoji = '🟢' if sentiment == 'BULLISH' else '🔴' if sentiment == 'BEARISH' else '⚪'
                impact_indicator = f"[{impact}]" if impact != "UNKNOWN" else ""
                
                prompt += f"""
- {sentiment_emoji} [{timestamp}] {impact_indicator} {title[:80]}
  Sentiment: {sentiment} (Confidence: {confidence:.0%})
"""
            
            if sentiment_score is not None:
                sentiment_emoji = '🟢 BULLISH momentum' if sentiment_score > 65 else '🔴 BEARISH pressure' if sentiment_score < 35 else '⚪ NEUTRAL'
                prompt += f"""
**Aggregate Market Sentiment Score: {sentiment_score:.1f}/100** ({sentiment_emoji})
"""
        else:
            prompt += """
**News Context:** No significant news in last 2 hours (market quiet)
"""
        
        # Build intent-specific decision rules
        if intent == PositionIntent.HODL.value:
            decision_rules = """
**🛡️ HODL-SPECIFIC DECISION RULES (RESPECT USER'S LONG-TERM INTENT):**
- **DEFAULT ACTION = HOLD** - The user is intentionally holding long-term
- Only recommend CLOSE_NOW if: rug pull detected, exchange hack, project abandoned, or >30% crash in hours
- Do NOT recommend exit just because position is down 5-15% - this is NORMAL volatility
- If in loss: Suggest HOLD with supportive reasoning about potential recovery
- Loss threshold for exit suggestion: Only if >25% loss AND clear fundamental breakdown
- For news: Only react to CATASTROPHIC news (security breaches, regulatory shutdown)
- Normal market FUD should result in HOLD, not exit
- Your job is to SUPPORT their holding strategy, not second-guess it

**When position is in LOSS and intent is HODL:**
- Acknowledge the current loss but emphasize the long-term view
- Do NOT suggest cutting losses unless truly catastrophic
- Confidence for HOLD should be HIGH (75-90%) for normal volatility"""
        elif intent == PositionIntent.SCALP.value:
            decision_rules = """
**⚡ SCALP-SPECIFIC DECISION RULES:**
- Be aggressive with exits - quick trades should have tight risk management
- Any loss >3% should trigger exit consideration
- Any profit >5% should trigger partial profit taking
- High urgency on all recommendations
- Quick action is key - don't wait for confirmation"""
        else:  # SWING
            decision_rules = """
**🔄 SWING TRADE DECISION RULES (PATIENT MID-TERM APPROACH):**

**⚠️ DEFAULT ACTION FOR SWING = HOLD** unless there's a compelling reason to exit.

**Before recommending CLOSE_NOW, ask yourself:**
1. Is this just normal volatility, or a true trend reversal?
2. Has the trade thesis been INVALIDATED, or just tested?
3. Would an experienced swing trader exit here, or hold through the noise?
4. Is there HIGH-CONFIDENCE negative news, or just FUD?

**Analysis Factors (in order of importance):**
1. **Trend Continuation**: Does the LARGER trend support holding? (Most Important)
2. **Support/Resistance**: Is price at a key level that changes the thesis?
3. **News Impact**: Only HIGH-CONFIDENCE, HIGH-IMPACT news should trigger exits
4. **Technical Signals**: Use as confirmation, not primary decision driver
5. **Risk/Reward**: Only reconsider if R:R has significantly deteriorated

**HOLD unless:**
- Clear multi-timeframe trend reversal confirmed
- Key support/resistance level broken with conviction
- HIGH confidence (>80%) negative news directly impacting this asset
- Position has hit take profit or extended significantly beyond target"""

        prompt += f"""
**Decision Framework:**
{decision_rules}

**Available Actions:**
1. **HOLD** - Continue monitoring, no changes needed
2. **TIGHTEN_STOP** - Move stop closer to protect profits (provide new_stop price)
3. **EXTEND_TARGET** - Raise take profit if trend strengthening (provide new_target price)
4. **TAKE_PARTIAL** - Close part of position to lock profits (provide partial_pct: 25/50/75)
5. **CLOSE_NOW** - Exit entire position immediately

**Critical Context Rules:**
- If BEARISH news with >80% confidence in last 30 min → Consider CLOSE_NOW or TIGHTEN_STOP
- If BULLISH news with >80% confidence → Consider EXTEND_TARGET or HOLD
- If sentiment contradicts technicals → Prioritize recent news (sentiment often leads price)
- If HIGH impact news (regulation, hacks, partnerships) → Increase URGENCY
- If no news and technicals favorable → Trust the technical setup
- **RESPECT THE USER'S INTENT: {intent}** - This is their chosen strategy!

**Respond ONLY with valid JSON (no other text):**
{{
    "action": "HOLD|TIGHTEN_STOP|EXTEND_TARGET|TAKE_PARTIAL|CLOSE_NOW",
    "confidence": 0-100,
    "reasoning": "Brief 1-2 sentence explanation citing news/technical factors AND respecting {intent} intent",
    "urgency": "LOW|MEDIUM|HIGH",
    "new_stop": price_value_or_null,
    "new_target": price_value_or_null,
    "partial_pct": percentage_or_null,
    "technical_score": 0-100,
    "trend_score": 0-100,
    "risk_score": 0-100
}}
"""
        
        return prompt
    
    def _parse_ai_response(self, response: str) -> Optional[Dict]:
        """Parse AI JSON response"""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in AI response")
                return None
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response JSON: {e}")
            logger.debug(f"Response: {response}")
            return None
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return None
    
    def execute_decision(self, trade_id: str, decision: AITradeDecision, skip_approval: bool = False) -> bool:
        """
        Execute AI decision for a position
        
        Args:
            trade_id: Position trade ID
            decision: AI decision to execute
            skip_approval: Skip manual approval (only for user-initiated actions)
        
        Returns:
            True if executed successfully, False if awaiting approval
        """
        if trade_id not in self.positions:
            log_and_print(f"Position {trade_id} not found", "ERROR")
            return False
        
        position = self.positions[trade_id]
        
        # 🛡️ ALERT COOLDOWN CHECK: Prevent spam and premature exit recommendations
        # Only check for HOLD, as HOLD should never be suppressed
        if decision.action != PositionAction.HOLD.value:
            should_suppress, suppress_reason = self._should_suppress_alert(position, decision.action)
            if should_suppress:
                position.alert_suppressed_count += 1
                log_and_print(f"🔇 Alert suppressed for {position.pair}: {suppress_reason}", "DEBUG")
                log_and_print(f"   Total suppressed: {position.alert_suppressed_count} | Intent: {position.position_intent}")
                return False  # Don't send alert, but return False to indicate no action taken
        
        # 🚨 SAFETY CHECK: Require manual approval unless explicitly skipped
        if self.require_manual_approval and not skip_approval:
            # Check for existing pending approval for this trade_id and action
            for pid, pending in self.pending_approvals.items():
                if pending['trade_id'] == trade_id and pending['decision'].action == decision.action:
                    # Already pending, don't create a duplicate
                    return False

            # Add to pending approvals queue instead of executing
            approval_id = f"{trade_id}_{int(time.time())}"
            current_price = position.current_price or position.entry_price
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            
            # Record that we're sending this alert (for cooldown tracking)
            self._record_alert_sent(position, decision.action)
            
            self.pending_approvals[approval_id] = {
                'trade_id': trade_id,
                'decision': decision,
                'timestamp': datetime.now(),
                'pair': position.pair
            }
            
            action_emoji = {
                'CLOSE_NOW': '🚪',
                'HOLD': '✋',
                'TIGHTEN_STOP': '🎯',
                'EXTEND_TARGET': '📈',
                'TAKE_PARTIAL': '💰',
                'MOVE_TO_BREAKEVEN': '🛡️'
            }
            
            log_and_print(f"🚨 APPROVAL REQUIRED: {decision.action} for {position.pair}", "WARNING")
            log_and_print(f"   Reasoning: {decision.reasoning}")
            log_and_print(f"   Approval ID: {approval_id}")
            
            # 🔔 SEND DISCORD APPROVAL REQUEST
            if self.discord_approval_manager:
                try:
                    # Build detailed reasoning
                    reasoning = (
                        f"**AI Analysis:**\n{decision.reasoning}\n\n"
                        f"**Technical Score:** {decision.technical_score:.0f}/100\n"
                        f"**Trend Score:** {decision.trend_score:.0f}/100\n"
                        f"**Risk Score:** {decision.risk_score:.0f}/100\n\n"
                        f"**Current P&L:** {pnl_pct:+.2f}%\n"
                        f"**Entry:** ${position.entry_price:,.6f}\n"
                        f"**Current:** ${current_price:,.6f}"
                    )
                    
                    # Determine position size in USD
                    position_size_usd = position.volume * current_price
                    
                    sent = self.discord_approval_manager.send_approval_request(
                        approval_id=approval_id,
                        pair=position.pair,
                        side=decision.action,  # Using action as "side" for exit decisions
                        entry_price=current_price,
                        position_size=position_size_usd,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        strategy=f"AI Position Manager - {decision.action}",
                        confidence=decision.confidence,
                        reasoning=reasoning,
                        additional_info=(
                            f"{action_emoji.get(decision.action, '🤖')} **Recommended Action:** {decision.action}\n"
                            f"**Urgency:** {decision.urgency}\n"
                            f"**Position Volume:** {position.volume:.6f}\n"
                            f"**Hold Time:** {(datetime.now() - position.entry_time).total_seconds() / 3600:.1f} hours"
                        )
                    )
                    
                    if sent:
                        log_and_print(f"📨 Discord approval request sent for {position.pair}")
                        log_and_print(f"   Reply APPROVE or REJECT in Discord to execute")
                    else:
                        log_and_print(f"⚠️ Could not send Discord approval - check Discord bot status", "WARNING")
                        
                except Exception as e:
                    log_and_print(f"Error sending Discord approval: {e}", "ERROR")
            else:
                # Fallback: send simple notification
                self._send_notification(
                    f"{action_emoji.get(decision.action, '🤖')} AI Recommendation: {position.pair}",
                    f"**Action:** {decision.action}\n"
                    f"**Confidence:** {decision.confidence:.1f}%\n"
                    f"**Current P&L:** {pnl_pct:+.2f}%\n"
                    f"**Reasoning:** {decision.reasoning[:200]}\n\n"
                    f"⚠️ **APPROVAL REQUIRED** - Discord bot not configured, approve in app",
                    color=15844367  # Orange for pending approval
                )
            
            return False  # Not executed, awaiting approval
        
        position = self.positions[trade_id]
        
        try:
            log_and_print("=" * 80)
            log_and_print(f"🤖 EXECUTING AI DECISION: {position.pair}")
            log_and_print("=" * 80)
            log_and_print(f"   Action: {decision.action}")
            log_and_print(f"   Confidence: {decision.confidence:.1f}%")
            log_and_print(f"   Reasoning: {decision.reasoning}")
            log_and_print("=" * 80)
            
            if decision.action == PositionAction.CLOSE_NOW.value:
                return self._execute_close_position(position, decision)
            
            elif decision.action == PositionAction.TIGHTEN_STOP.value:
                return self._execute_tighten_stop(position, decision)
            
            elif decision.action == PositionAction.EXTEND_TARGET.value:
                return self._execute_extend_target(position, decision)
            
            elif decision.action == PositionAction.TAKE_PARTIAL.value:
                return self._execute_partial_exit(position, decision)
            
            elif decision.action == PositionAction.MOVE_TO_BREAKEVEN.value:
                return self._execute_move_to_breakeven(position, decision)
            
            elif decision.action == PositionAction.HOLD.value:
                log_and_print(f"✓ HOLD - No action needed for {position.pair}")
                return True
            
            else:
                log_and_print(f"Unknown action: {decision.action}", "WARNING")
                return False
                
        except Exception as e:
            log_and_print(f"Error executing decision for {trade_id}: {str(e)}", "ERROR")
            return False
    
    def _execute_close_position(
        self,
        position: MonitoredCryptoPosition,
        decision: AITradeDecision
    ) -> bool:
        """Close entire position"""
        try:
            from clients.kraken_client import OrderSide, OrderType
            
            # Determine order side (opposite of entry)
            order_side = OrderSide.SELL if position.side == 'BUY' else OrderSide.BUY
            
            # Place market order to close
            result = self.kraken_client.place_order(
                pair=position.pair,
                side=order_side,
                order_type=OrderType.MARKET,
                volume=position.volume
            )
            
            if result:
                logger.info(f"✅ Position closed: {position.pair} - Order ID: {result.order_id}")
                
                # Calculate final P&L
                if position.side == 'BUY':
                    pnl_pct = ((position.current_price - position.entry_price) / position.entry_price) * 100
                else:
                    pnl_pct = ((position.entry_price - position.current_price) / position.entry_price) * 100
                
                # Update statistics
                self.ai_exit_signals += 1
                
                # Send notification
                self._send_notification(
                    f"🤖 AI Exit: {position.pair}",
                    f"**Action:** {decision.action}\n"
                    f"**P&L:** {pnl_pct:+.2f}%\n"
                    f"**Reason:** {decision.reasoning}\n"
                    f"**Confidence:** {decision.confidence:.1f}%",
                    color=65280 if pnl_pct > 0 else 16711680  # Green/Red
                )
                
                # Update journal with exit
                try:
                    from services.unified_trade_journal import get_unified_journal
                    journal = get_unified_journal()
                    
                    # Get market conditions
                    technical_data = self._get_technical_indicators(position.pair)
                    market_conditions = {
                        'rsi': technical_data.get('rsi'),
                        'macd': technical_data.get('macd'),
                        'volume_change': technical_data.get('volume_change_pct'),
                        'trend': technical_data.get('trend')
                    }
                    
                    journal.update_trade_exit(
                        trade_id=position.trade_id,
                        exit_price=position.current_price,
                        exit_time=datetime.now(),
                        exit_reason=f"AI: {decision.reasoning}",
                        market_conditions=market_conditions
                    )
                    logger.debug(f"📝 Updated journal with trade exit: {position.trade_id}")
                except Exception as e:
                    logger.debug(f"Could not update journal with exit: {e}")
                
                # Update Signal Memory with outcome (RAG learning)
                try:
                    from services.signal_memory import update_crypto_signal_outcome
                    
                    # Calculate holding time
                    holding_hours = int((datetime.now() - position.entry_time).total_seconds() / 3600)
                    outcome = "WIN" if pnl_pct > 0 else "LOSS"
                    
                    updated = update_crypto_signal_outcome(
                        trade_id=position.trade_id,
                        outcome=outcome,
                        pnl_pct=pnl_pct,
                        holding_hours=holding_hours
                    )
                    
                    if updated:
                        emoji = "✅" if outcome == "WIN" else "❌"
                        logger.info(f"🧠 {emoji} Updated signal memory: {position.pair} {outcome} ({pnl_pct:+.2f}%)")
                except Exception as mem_err:
                    logger.debug(f"Could not update signal memory outcome (non-critical): {mem_err}")
                
                # Remove from monitoring
                position.status = PositionStatus.CLOSED.value
                self.remove_position(position.trade_id, reason=f"AI Exit: {decision.reasoning}")
                
                return True
            else:
                logger.error(f"Failed to close position {position.pair}")
                return False
                
        except Exception as e:
            logger.error("Error closing position: {}", str(e), exc_info=True)
            return False
    
    def _execute_tighten_stop(
        self,
        position: MonitoredCryptoPosition,
        decision: AITradeDecision
    ) -> bool:
        """Tighten stop loss"""
        if not decision.new_stop:
            logger.warning("No new_stop provided for TIGHTEN_STOP")
            return False
        
        # Update stop loss
        old_stop = position.stop_loss
        position.stop_loss = decision.new_stop
        position.ai_adjustment_count += 1
        self.total_ai_adjustments += 1
        
        logger.info(f"📈 Stop tightened: {position.pair} - ${old_stop:,.2f} → ${decision.new_stop:,.2f}")
        
        # Note: Kraken doesn't support modifying stop orders easily
        # We update our internal tracking and will exit at market if stop hit on next check
        
        self._save_state()
        return True
    
    def _execute_extend_target(
        self,
        position: MonitoredCryptoPosition,
        decision: AITradeDecision
    ) -> bool:
        """Extend take profit target"""
        if not decision.new_target:
            logger.warning("No new_target provided for EXTEND_TARGET")
            return False
        
        # Update take profit
        old_target = position.take_profit
        position.take_profit = decision.new_target
        position.ai_adjustment_count += 1
        self.total_ai_adjustments += 1
        
        logger.info(f"🎯 Target extended: {position.pair} - ${old_target:,.2f} → ${decision.new_target:,.2f}")
        
        self._save_state()
        return True
    
    def _execute_partial_exit(
        self,
        position: MonitoredCryptoPosition,
        decision: AITradeDecision
    ) -> bool:
        """Take partial profits"""
        if not decision.partial_pct or position.partial_exit_taken:
            logger.warning(f"Cannot take partial exit for {position.pair}")
            return False
        
        try:
            from clients.kraken_client import OrderSide, OrderType
            
            # Calculate partial volume
            partial_volume = position.volume * (decision.partial_pct / 100)
            
            # Determine order side
            order_side = OrderSide.SELL if position.side == 'BUY' else OrderSide.BUY
            
            # Place market order for partial exit
            result = self.kraken_client.place_order(
                pair=position.pair,
                side=order_side,
                order_type=OrderType.MARKET,
                volume=partial_volume
            )
            
            if result:
                logger.info(f"💰 Partial exit executed: {position.pair} - {decision.partial_pct}% ({partial_volume:.6f})")
                
                # Update position
                position.volume -= partial_volume
                position.partial_exit_taken = True
                position.partial_exit_pct = decision.partial_pct
                self.partial_exits_taken += 1
                
                # Send notification
                self._send_notification(
                    f"💰 Partial Profit: {position.pair}",
                    f"**Closed:** {decision.partial_pct}% of position\n"
                    f"**Remaining:** {position.volume:.6f}\n"
                    f"**Reason:** {decision.reasoning}",
                    color=16776960  # Yellow
                )
                
                self._save_state()
                return True
            else:
                logger.error(f"Failed to execute partial exit for {position.pair}")
                return False
                
        except Exception as e:
            logger.error("Error executing partial exit: {}", str(e), exc_info=True)
            return False
    
    def _execute_move_to_breakeven(
        self,
        position: MonitoredCryptoPosition,
        decision: AITradeDecision
    ) -> bool:
        """Move stop to breakeven"""
        if not decision.new_stop:
            return False
        
        old_stop = position.stop_loss
        position.stop_loss = decision.new_stop
        position.moved_to_breakeven = True
        
        logger.info(f"🔒 Breakeven move: {position.pair} - Stop ${old_stop:,.2f} → ${decision.new_stop:,.2f}")
        
        self._save_state()
        return True
    
    def _send_notification(self, title: str, message: str, color: int = 3447003):
        """Send Discord notification"""
        try:
            import os
            import requests
            
            webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
            if not webhook_url:
                return
            
            embed = {
                'title': title,
                'description': message,
                'color': color,
                'timestamp': datetime.now().isoformat()
            }
            
            payload = {'embeds': [embed]}
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            
        except Exception as e:
            logger.debug(f"Failed to send Discord notification: {e}")
    
    def exclude_pair(self, pair: str) -> bool:
        """Exclude a pair from monitoring and close if active"""
        try:
            # Always ensure it's in the set
            if pair not in self.excluded_pairs:
                self.excluded_pairs.add(pair)
                log_and_print(f"🚫 Added {pair} to excluded list")
            
            # ALWAYS clean up active positions, even if already excluded
            positions_to_remove = []
            for trade_id, pos in self.positions.items():
                if pos.pair == pair:
                    positions_to_remove.append(trade_id)
            
            if positions_to_remove:
                log_and_print(f"   Cleaning up {len(positions_to_remove)} lingering positions for excluded pair {pair}")
                for trade_id in positions_to_remove:
                    self.remove_position(trade_id, reason="Pair excluded by user")
            
            self._save_state()
            return True
            
        except Exception as e:
            logger.error(f"Error excluding pair {pair}: {e}")
            return False

    def include_pair(self, pair: str) -> bool:
        """Re-include a pair for monitoring"""
        try:
            if pair in self.excluded_pairs:
                self.excluded_pairs.remove(pair)
                log_and_print(f"✅ Removed {pair} from excluded list")
                self._save_state()
                return True
            return False
        except Exception as e:
            logger.error(f"Error including pair {pair}: {e}")
            return False
    
    def set_position_intent(self, pair: str, intent: str) -> bool:
        """
        Set the holding intent for a position.
        
        Args:
            pair: Trading pair (e.g., 'BILLY/USD')
            intent: 'HODL', 'SWING', or 'SCALP'
                - HODL: Long-term hold, very tolerant of losses, minimal sell alerts
                - SWING: Standard swing trade rules (default)
                - SCALP: Quick trade, tight stops, aggressive exit suggestions
        
        Returns:
            True if intent was set successfully
        """
        # Validate intent
        valid_intents = [i.value for i in PositionIntent]
        intent = intent.upper()
        if intent not in valid_intents:
            log_and_print(f"❌ Invalid intent '{intent}'. Must be one of: {valid_intents}", "ERROR")
            return False
        
        # Find positions for this pair
        updated_count = 0
        for trade_id, position in self.positions.items():
            if position.pair == pair:
                old_intent = position.position_intent
                position.position_intent = intent
                updated_count += 1
                
                intent_emoji = {"HODL": "💎", "SWING": "🔄", "SCALP": "⚡"}.get(intent, "📊")
                log_and_print(f"{intent_emoji} Set {pair} intent: {old_intent} → {intent}")
                log_and_print(f"   {PositionIntent.get_description(intent)}")
        
        if updated_count == 0:
            log_and_print(f"⚠️ No active positions found for {pair}", "WARNING")
            return False
        
        self._save_state()
        return True
    
    def set_all_positions_intent(self, intent: str) -> int:
        """
        Set the holding intent for ALL active positions.
        
        Args:
            intent: 'HODL', 'SWING', or 'SCALP'
        
        Returns:
            Number of positions updated
        """
        valid_intents = [i.value for i in PositionIntent]
        intent = intent.upper()
        if intent not in valid_intents:
            log_and_print(f"❌ Invalid intent '{intent}'. Must be one of: {valid_intents}", "ERROR")
            return 0
        
        updated_count = 0
        for trade_id, position in self.positions.items():
            if position.status == PositionStatus.ACTIVE.value:
                position.position_intent = intent
                updated_count += 1
        
        if updated_count > 0:
            intent_emoji = {"HODL": "💎", "SWING": "🔄", "SCALP": "⚡"}.get(intent, "📊")
            log_and_print(f"{intent_emoji} Set ALL {updated_count} positions to {intent} intent")
            log_and_print(f"   {PositionIntent.get_description(intent)}")
            self._save_state()
        
        return updated_count
    
    def get_position_intent(self, pair: str) -> Optional[str]:
        """Get the current intent for a position"""
        for position in self.positions.values():
            if position.pair == pair:
                return position.position_intent
        return None
    
    def _save_state(self):
        """Save position state to file"""
        try:
            state = {
                'positions': {
                    trade_id: asdict(pos) for trade_id, pos in self.positions.items()
                },
                'excluded_pairs': list(self.excluded_pairs),
                'statistics': {
                    'total_ai_adjustments': self.total_ai_adjustments,
                    'trailing_stop_activations': self.trailing_stop_activations,
                    'breakeven_moves': self.breakeven_moves,
                    'partial_exits_taken': self.partial_exits_taken,
                    'ai_exit_signals': self.ai_exit_signals
                },
                'last_updated': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings
            for pos in state['positions'].values():
                if isinstance(pos.get('entry_time'), datetime):
                    pos['entry_time'] = pos['entry_time'].isoformat()
                if isinstance(pos.get('last_check_time'), datetime):
                    pos['last_check_time'] = pos['last_check_time'].isoformat()
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error("Error saving state: {}", str(e), exc_info=True)
    
    def _load_state(self):
        """Load position state from file"""
        try:
            if not os.path.exists(self.state_file):
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore excluded pairs
            self.excluded_pairs = set(state.get('excluded_pairs', []))
            
            # Restore positions, skipping those in excluded list
            self.positions = {}
            skipped_count = 0
            for trade_id, pos_data in state.get('positions', {}).items():
                pair = pos_data.get('pair')
                if pair in self.excluded_pairs:
                    skipped_count += 1
                    continue
                    
                # Convert ISO strings back to datetime
                if 'entry_time' in pos_data:
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                if 'last_check_time' in pos_data:
                    pos_data['last_check_time'] = datetime.fromisoformat(pos_data['last_check_time'])
                
                self.positions[trade_id] = MonitoredCryptoPosition(**pos_data)
            
            if skipped_count > 0:
                logger.info(f"   Skipped {skipped_count} positions that are in excluded list")
            
            # Restore statistics
            stats = state.get('statistics', {})
            self.total_ai_adjustments = stats.get('total_ai_adjustments', 0)
            self.trailing_stop_activations = stats.get('trailing_stop_activations', 0)
            self.breakeven_moves = stats.get('breakeven_moves', 0)
            self.partial_exits_taken = stats.get('partial_exits_taken', 0)
            self.ai_exit_signals = stats.get('ai_exit_signals', 0)
            
            logger.info(f"📂 Loaded {len(self.positions)} positions from state file")
            
        except Exception as e:
            logger.error("Error loading state: {}", str(e), exc_info=True)
    
    def start_monitoring_loop(self):
        """
        Start the monitoring loop in a background thread
        Checks positions every check_interval_seconds
        """
        import threading
        import sys
        
        if self.is_running:
            logger.warning("Monitoring loop already running")
            return
        
        def monitoring_loop():
            self.is_running = True
            cycle_count = 0
            log_and_print("=" * 60)
            log_and_print("🚀 AI CRYPTO POSITION MANAGER MONITORING STARTED")
            log_and_print("=" * 60)
            
            while self.is_running:
                try:
                    cycle_count += 1
                    active_positions = [p for p in self.positions.values() if p.status == PositionStatus.ACTIVE.value]
                    active_count = len(active_positions)
                    
                    # Detailed heartbeat log every cycle
                    log_and_print("")
                    log_and_print("=" * 60)
                    log_and_print(f"💓 CYCLE #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    log_and_print("=" * 60)
                    log_and_print(f"   Active positions: {active_count}")
                    log_and_print(f"   Pending approvals: {len(self.pending_approvals)}")
                    
                    # Show each position status
                    if active_positions:
                        log_and_print("")
                        log_and_print("📊 POSITION STATUS:")
                        for pos in active_positions:
                            pnl_pct = 0
                            if pos.current_price and pos.entry_price:
                                pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
                            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                            intent_emoji = {"HODL": "💎", "SWING": "🔄", "SCALP": "⚡"}.get(pos.position_intent, "📊")
                            hold_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600 if pos.entry_time else 0
                            log_and_print(f"   {pnl_emoji} {pos.pair} {intent_emoji}: ${pos.current_price:,.4f} | P&L: {pnl_pct:+.2f}% | Hold: {hold_hours:.1f}h")
                            log_and_print(f"      Entry: ${pos.entry_price:,.4f} | Stop: ${pos.stop_loss:,.4f} | Target: ${pos.take_profit:,.4f} | Intent: {pos.position_intent}")
                    
                    # Monitor all positions
                    log_and_print("")
                    log_and_print("🔍 ANALYZING POSITIONS...")
                    decisions = self.monitor_positions()
                    
                    # Execute AI decisions
                    if decisions:
                        log_and_print("")
                        log_and_print(f"⚡ ACTIONS REQUIRED: {len(decisions)} decision(s)")
                        for i, (decision, decision_trade_id) in enumerate(decisions, 1):
                            position = self.positions.get(decision_trade_id)
                            pair_name = position.pair if position else 'Unknown'
                            log_and_print(f"   [{i}] {decision.action} - Confidence: {decision.confidence:.0f}%")
                            log_and_print(f"       Reason: {decision.reasoning[:100]}...")
                            log_and_print(f"       Executing for: {pair_name}")
                            
                            if position:
                                self.execute_decision(decision_trade_id, decision)
                    else:
                        if active_count == 0:
                            log_and_print("   ⏳ No positions to manage - waiting for trades...")
                        else:
                            log_and_print("   ✅ All positions healthy - no action needed")
                    
                    # Show pending approvals if any
                    if self.pending_approvals:
                        log_and_print("")
                        log_and_print(f"⏳ PENDING APPROVALS ({len(self.pending_approvals)}):")
                        for approval_id, approval in self.pending_approvals.items():
                            age_mins = (datetime.now() - approval['timestamp']).total_seconds() / 60
                            log_and_print(f"   • {approval['pair']}: {approval['decision'].action} (waiting {age_mins:.0f}m)")
                    
                    log_and_print("")
                    log_and_print(f"💤 Sleeping {self.check_interval_seconds}s until next cycle...")
                    log_and_print("=" * 60)
                    
                    time.sleep(self.check_interval_seconds)
                    
                except Exception as e:
                    log_and_print(f"❌ Error in monitoring loop: {str(e)}", "ERROR")
                    time.sleep(10)  # Short sleep on error
            
            log_and_print("🛑 AI Crypto Position Manager stopped")
        
        self.thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.thread.start()
        log_and_print(f"✅ Monitoring loop started (checking every {self.check_interval_seconds}s)")
    
    def stop(self):
        """Stop the monitoring loop"""
        logger.info("⏸️ Stopping AI Crypto Position Manager...")
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=5)
    
    def _generate_portfolio_recommendations(self, position_details: List[Dict]) -> List[Dict]:
        """Generate high-level portfolio recommendations"""
        recommendations = []
        total_value = sum(p['current_value'] for p in position_details)

        # 1. Concentration risk
        for pos in position_details:
            if pos['allocation_pct'] > 30:  # Over 30% in one asset
                recommendations.append({
                    'pair': pos['pair'],
                    'priority': 'MEDIUM',
                    'action': 'REBALANCE',
                    'reason': f"High concentration at {pos['allocation_pct']:.0f}% of portfolio. Consider diversifying."
                })

        # 2. Significant loss
        for pos in position_details:
            if pos['pnl_pct'] < -15:  # Over 15% loss
                recommendations.append({
                    'pair': pos['pair'],
                    'priority': 'HIGH',
                    'action': 'CUT_LOSS',
                    'reason': f"Significant loss of {pos['pnl_pct']:.1f}%. Consider cutting losses or dollar-cost averaging."
                })

        # 3. Significant gain
        for pos in position_details:
            if pos['pnl_pct'] > 30:  # Over 30% gain
                recommendations.append({
                    'pair': pos['pair'],
                    'priority': 'MEDIUM',
                    'action': 'TAKE_PROFIT',
                    'reason': f"Significant unrealized gain of {pos['pnl_pct']:.1f}%. Consider taking partial profits."
                })
        
        return recommendations

    def analyze_portfolio(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a comprehensive analysis of the provided portfolio positions.

        Args:
            positions: A list of position dictionaries from the broker client.

        Returns:
            A dictionary with portfolio summary, aggregated positions, and recommendations.
        """
        if not positions:
            return {
                'summary': {
                    'total_value': 0,
                    'total_pnl': 0,
                    'total_pnl_pct': 0,
                    'realized_gains': 0,
                    'win_rate': 0,
                    'top_gainers': [],
                    'top_losers': []
                },
                'positions': [],
                'winners': [],  # Add top-level winners key
                'losers': [],   # Add top-level losers key
                'recommendations': []
            }

        # 1. Aggregate positions by pair
        aggregated = {}
        for pos in positions:
            pair = pos.get('pair')
            if not pair: continue

            volume = float(pos.get('volume', 0))
            entry_price = float(pos.get('entry_price', 0))
            current_price = float(pos.get('current_price', 0))
            
            cost_basis = entry_price * volume
            current_value = current_price * volume

            if pair not in aggregated:
                aggregated[pair] = {
                    'pair': pair,
                    'volume': 0,
                    'cost_basis': 0,
                    'current_value': 0,
                    'entry_price': 0 # This will be an aggregated average
                }

            aggregated[pair]['volume'] += volume
            aggregated[pair]['cost_basis'] += cost_basis
            aggregated[pair]['current_value'] += current_value

        # 2. Calculate final metrics for aggregated positions
        position_details = []
        total_value = 0
        total_cost_basis = 0

        for pair, data in aggregated.items():
            if data['volume'] > 0:
                avg_entry_price = data['cost_basis'] / data['volume']
                current_price = data['current_value'] / data['volume'] # Assumes uniform current price
                pnl = data['current_value'] - data['cost_basis']
                pnl_pct = (pnl / data['cost_basis']) * 100 if data['cost_basis'] > 0 else 0

                position_details.append({
                    'pair': pair,
                    'volume': data['volume'],
                    'entry_price': avg_entry_price,  # Changed from 'entry'
                    'current_price': current_price,   # Changed from 'current'
                    'current_value': data['current_value'],  # Changed from 'value'
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                total_value += data['current_value']
                total_cost_basis += data['cost_basis']

        # 3. Calculate portfolio summary
        total_pnl = total_value - total_cost_basis
        total_pnl_pct = (total_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else 0

        # Sort by value and add allocation
        position_details = sorted(position_details, key=lambda x: x['current_value'], reverse=True)
        for pos in position_details:
            pos['allocation_pct'] = (pos['current_value'] / total_value) * 100 if total_value > 0 else 0

        # Top gainers/losers
        top_gainers = sorted([p for p in position_details if p['pnl'] > 0], key=lambda x: x['pnl'], reverse=True)[:3]
        top_losers = sorted([p for p in position_details if p['pnl'] < 0], key=lambda x: x['pnl'])[:3]

        # AI Recommendations
        recommendations = self._generate_portfolio_recommendations(position_details)

        return {
            'summary': {
                'total_value': total_value,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'realized_gains': 0,  # Placeholder
                'win_rate': 0,  # Placeholder
                'top_gainers': top_gainers,
                'top_losers': top_losers
            },
            'positions': position_details,
            'winners': top_gainers,  # Add top-level winners key
            'losers': top_losers,    # Add top-level losers key
            'recommendations': recommendations
        }

    def sync_with_kraken(self) -> Dict[str, int]:
        """
        Sync AI monitor positions with current Kraken positions
        Removes positions not in Kraken, adds missing positions
        
        Returns:
            Dict with 'removed', 'added', 'kept' counts
        """
        try:
            log_and_print("🔄 Syncing with Kraken portfolio...")
            
            # Fetch current Kraken positions
            kraken_positions = self.kraken_client.get_open_positions(calculate_real_cost=True)
            log_and_print(f"   Found {len(kraken_positions)} position(s) on Kraken")
            
            # Create a map of Kraken positions by pair
            kraken_pairs = {}
            for pos in kraken_positions:
                if pos.pair not in kraken_pairs:
                    kraken_pairs[pos.pair] = []
                kraken_pairs[pos.pair].append(pos)
            
            removed_count = 0
            kept_count = 0
            
            # Remove AI positions that no longer exist in Kraken
            positions_to_remove = []
            for trade_id, ai_pos in self.positions.items():
                # Check if this pair still exists in Kraken
                if ai_pos.pair not in kraken_pairs:
                    positions_to_remove.append(trade_id)
                    log_and_print(f"   📤 Removing {ai_pos.pair} - No longer in Kraken portfolio")
                else:
                    kept_count += 1
            
            # Actually remove the positions
            for trade_id in positions_to_remove:
                self.remove_position(trade_id, reason="Not in Kraken portfolio")
                removed_count += 1
            
            # Add new positions from Kraken that aren't being monitored
            added_count = 0
            import time
            for pair, kraken_pos_list in kraken_pairs.items():
                # Check if pair is excluded
                if pair in self.excluded_pairs:
                    # log_and_print(f"   🚫 Skipping {pair} (in exclusion list)")
                    continue

                # Check if we're already monitoring this pair
                already_monitored = any(p.pair == pair for p in self.positions.values())
                
                if not already_monitored and kraken_pos_list:
                    # Add the first position for this pair
                    pos = kraken_pos_list[0]
                    trade_id = f"{pos.pair}_synced_{int(time.time())}_{added_count}"
                    
                    # Calculate stop loss and take profit (5% and 10% from current)
                    stop_loss = pos.current_price * 0.95
                    take_profit = pos.current_price * 1.10
                    entry_price = pos.entry_price if pos.entry_price > 0 else pos.current_price
                    
                    log_and_print(f"   📥 Adding {pos.pair} - Volume: {pos.volume:.6f}, Entry: ${entry_price:,.4f}")
                    
                    success = self.add_position(
                        trade_id=trade_id,
                        pair=pos.pair,
                        side='BUY',
                        volume=pos.volume,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy='SYNCED_FROM_KRAKEN',
                        entry_order_id=f"synced_{added_count}"
                    )
                    
                    if success:
                        added_count += 1
                        log_and_print(f"   ✅ Added {pos.pair} to AI monitoring")
                        
                        # Calculate P&L for notification
                        pnl_pct = ((pos.current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                        pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                        
                        # Send Discord notification for synced position
                        self._send_notification(
                            f"📥 POSITION SYNCED: {pos.pair}",
                            f"**Imported from Kraken!**\n"
                            f"**Volume:** {pos.volume:.6f}\n"
                            f"**Entry:** ${entry_price:,.4f}\n"
                            f"**Current:** ${pos.current_price:,.4f} ({pnl_emoji} {pnl_pct:+.1f}%)\n"
                            f"**Stop:** ${stop_loss:,.4f} | **Target:** ${take_profit:,.4f}\n\n"
                            f"_AI monitoring now active for this position_",
                            color=3447003  # Blue
                        )
            
            self._save_state()
            
            # Send summary notification if positions were synced
            if added_count > 0:
                self._send_notification(
                    f"✅ KRAKEN SYNC COMPLETE",
                    f"**Added:** {added_count} position(s)\n"
                    f"**Removed:** {removed_count}\n"
                    f"**Kept:** {kept_count}\n\n"
                    f"_All positions now being monitored by AI_",
                    color=3066993  # Green
                )
            
            log_and_print(f"✅ Sync complete: {added_count} added, {removed_count} removed, {kept_count} kept")
            
            return {
                'added': added_count,
                'removed': removed_count,
                'kept': kept_count
            }
            
        except Exception as e:
            log_and_print(f"Error syncing with Kraken: {e}", "ERROR")
            return {'added': 0, 'removed': 0, 'kept': 0}
    
    def get_status(self) -> Dict:
        """Get current manager status"""
        return {
            'is_running': self.is_running,
            'active_positions': len([p for p in self.positions.values() if p.status == PositionStatus.ACTIVE.value]),
            'total_positions': len(self.positions),
            'excluded_pairs': list(self.excluded_pairs),
            'positions': {
                trade_id: {
                    'pair': pos.pair,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl_pct': (
                        ((pos.current_price - pos.entry_price) / pos.entry_price) * 100 
                        if pos.side == 'BUY' and pos.entry_price > 0
                        else ((pos.entry_price - pos.current_price) / pos.entry_price) * 100 
                        if pos.entry_price > 0
                        else 0.0
                    ),
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'hold_time_minutes': (datetime.now() - pos.entry_time).total_seconds() / 60,
                    'last_ai_action': pos.last_ai_action,
                    'last_ai_confidence': pos.last_ai_confidence,
                    'moved_to_breakeven': pos.moved_to_breakeven,
                    'partial_exit_taken': pos.partial_exit_taken,
                    'status': pos.status,
                    'position_intent': pos.position_intent  # HODL, SWING, or SCALP
                }
                for trade_id, pos in self.positions.items()
            },
            'statistics': {
                'total_ai_adjustments': self.total_ai_adjustments,
                'trailing_stop_activations': self.trailing_stop_activations,
                'breakeven_moves': self.breakeven_moves,
                'partial_exits_taken': self.partial_exits_taken,
                'ai_exit_signals': self.ai_exit_signals
            },
            'config': {
                'check_interval': self.check_interval_seconds,
                'ai_enabled': self.enable_ai_decisions,
                'trailing_stops': self.enable_trailing_stops,
                'breakeven_moves': self.enable_breakeven_moves,
                'partial_exits': self.enable_partial_exits,
                'min_ai_confidence': self.min_ai_confidence
            }
        }


# Singleton instance
_ai_position_manager: Optional[AICryptoPositionManager] = None


def _create_kraken_client():
    """Create Kraken client from environment configuration"""
    try:
        from clients.kraken_client import KrakenClient
        
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_API_SECRET')
        
        if not api_key or not api_secret:
            logger.warning("Kraken API credentials not found in environment")
            return None
        
        client = KrakenClient(
            api_key=api_key,
            api_secret=api_secret
        )
        logger.info("✅ Kraken client created for crypto position manager")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create Kraken client: {e}")
        return None


def get_ai_position_manager(
    kraken_client=None,
    llm_analyzer=None,
    **kwargs
) -> AICryptoPositionManager:
    """Get or create singleton AI position manager"""
    global _ai_position_manager
    
    if _ai_position_manager is None:
        if kraken_client is None:
            raise ValueError("kraken_client required for first initialization")
        
        _ai_position_manager = AICryptoPositionManager(
            kraken_client=kraken_client,
            llm_analyzer=llm_analyzer,
            **kwargs
        )
    
    return _ai_position_manager


def get_ai_crypto_position_manager(
    kraken_client=None,
    **kwargs
) -> Optional[AICryptoPositionManager]:
    """
    Get or create singleton AI Crypto Position Manager instance
    
    This function auto-initializes the Kraken client from environment variables
    if not provided, making it easy to use from the Service Control Panel.
    
    Args:
        kraken_client: Optional KrakenClient instance
        **kwargs: Additional configuration options
        
    Returns:
        AICryptoPositionManager instance or None if Kraken not configured
    """
    global _ai_position_manager
    
    # Auto-create Kraken client if not provided and not already initialized
    if _ai_position_manager is None:
        if kraken_client is None:
            kraken_client = _create_kraken_client()
        
        if kraken_client is None:
            logger.warning("Cannot initialize AI Crypto Position Manager - Kraken client not available")
            return None
        
        try:
            _ai_position_manager = AICryptoPositionManager(
                kraken_client=kraken_client,
                **kwargs
            )
            logger.info("✅ AI Crypto Position Manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AI Crypto Position Manager: {e}")
            return None
    
    return _ai_position_manager

