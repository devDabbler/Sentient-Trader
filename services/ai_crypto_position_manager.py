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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

from services.trading_agents import TradingAgentOrchestrator, CoordinatorDecisionContext


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
    last_check_time: datetime = None
    
    # AI management
    trailing_stop_pct: float = 2.0  # 2% trailing stop
    breakeven_trigger_pct: float = 3.0  # Move to BE after 3% profit
    moved_to_breakeven: bool = False
    partial_exit_taken: bool = False
    partial_exit_pct: float = 0.0
    
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


class AICryptoPositionManager:
    """
    AI-Enhanced Position Manager for Crypto Trading
    Monitors positions 24/7 and makes intelligent exit decisions
    """
    
    def __init__(
        self,
        kraken_client,
        llm_analyzer=None,
        check_interval_seconds: int = 60,
        enable_ai_decisions: bool = True,
        enable_trailing_stops: bool = True,
        enable_breakeven_moves: bool = True,
        enable_partial_exits: bool = True,
        min_ai_confidence: float = 65.0,
        state_file: str = "data/ai_crypto_positions.json",
        require_manual_approval: bool = True  # SAFETY: Require manual approval for trades
    ):
        """
        Initialize AI Crypto Position Manager
        
        Args:
            kraken_client: KrakenClient instance
            llm_analyzer: LLM analyzer for AI decisions (LLMStrategyAnalyzer)
            check_interval_seconds: How often to check positions (default: 60s)
            enable_ai_decisions: Enable AI-powered exit decisions
            enable_trailing_stops: Enable trailing stop feature
            enable_breakeven_moves: Enable break-even stop moves
            enable_partial_exits: Enable partial profit taking
            min_ai_confidence: Minimum AI confidence to act (0-100)
            state_file: File to persist position state
            require_manual_approval: SAFETY - Require user approval before executing trades (default: True)
        """
        self.kraken_client = kraken_client
        self.llm_analyzer = llm_analyzer
        self.check_interval_seconds = check_interval_seconds
        self.enable_ai_decisions = enable_ai_decisions
        self.enable_trailing_stops = enable_trailing_stops
        self.enable_breakeven_moves = enable_breakeven_moves
        self.enable_partial_exits = enable_partial_exits
        self.min_ai_confidence = min_ai_confidence
        self.state_file = state_file
        self.require_manual_approval = require_manual_approval
        
        # Monitored positions
        self.positions: Dict[str, MonitoredCryptoPosition] = {}
        
        # SAFETY: Pending approvals queue
        self.pending_approvals: Dict[str, Dict] = {}  # {approval_id: {trade_id, decision, timestamp}}
        
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
        self.max_positions = 10
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
        logger.info("   SAFETY - MANUAL APPROVAL: {}", str(' REQUIRED' if require_manual_approval else ' AUTO-EXECUTE (DANGEROUS!)'))
        logger.info("=" * 80)
    
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
            logger.info(f"   Risk/Reward: {}:1 {rr_ratio:.2f}")
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
            
            logger.info(f" Removed {position.pair} from AI monitoring (Reason: {reason})")
            self._save_state()
            return True
        return False
    
    def monitor_positions(self) -> List[AITradeDecision]:
        """
        Check all monitored positions and get AI recommendations
        
        Returns:
            List of AI decisions for positions requiring action
        """
        decisions = []
        
        if not self.positions:
            logger.debug("No positions to monitor")
            return decisions
        
        logger.debug(f" Checking {len(self.positions)} positions...")
        
        for trade_id, position in list(self.positions.items()):
            try:
                # Skip if not active
                if position.status != PositionStatus.ACTIVE.value:
                    continue
                
                # Skip futures/staking positions (not supported for spot trading)
                # Check for .F, .S, .M, .P suffixes in the base currency
                pair_parts = position.pair.split('/')
                if len(pair_parts) > 0:
                    base_currency = pair_parts[0]
                    if any(base_currency.endswith(suffix) for suffix in ['.F', '.S', '.M', '.P']):
                        logger.warning(f" Closing futures/staking position: {position.pair} (not supported for spot trading)")
                        # Mark as closed to stop monitoring
                        position.status = PositionStatus.CLOSED.value
                        position.exit_time = datetime.now()
                        position.exit_reason = "Futures/staking contract not supported"
                        continue
                
                # Get current price
                ticker_info = self.kraken_client.get_ticker_info(position.pair)
                if not ticker_info:
                    logger.warning(f" Failed to get ticker for {position.pair}")
                    continue
                
                current_price = float(ticker_info.get('c', [0])[0])
                if current_price <= 0:
                    logger.warning(f" Invalid price for {position.pair}: {current_price}")
                    continue
                
                # Update position tracking
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
                
                # 1. Check basic exit conditions (stop/target)
                basic_action = self._check_basic_exit_conditions(position, current_price, pnl_pct)
                if basic_action:
                    decisions.append(basic_action)
                    logger.info(f" BASIC EXIT SIGNAL: {position.pair} - {basic_action.action} ({basic_action.reasoning})")
                    continue
                
                # 2. Check breakeven move
                if self.enable_breakeven_moves:
                    be_action = self._check_breakeven_move(position, pnl_pct)
                    if be_action:
                        decisions.append(be_action)
                        logger.info(f" BREAKEVEN MOVE: {position.pair} - Moving stop to breakeven")
                        continue
                
                # 3. Check trailing stop
                if self.enable_trailing_stops and pnl_pct > 0:
                    trail_action = self._check_trailing_stop(position, current_price, pnl_pct)
                    if trail_action:
                        decisions.append(trail_action)
                        logger.info(f" TRAILING STOP: {position.pair} - Tightening stop to ${trail_action.new_stop:,.2f}")
                        continue
                
                # 4. Get AI recommendation (every check for ACTIVE positions)
                if self.enable_ai_decisions and self.llm_analyzer:
                    ai_decision = self._get_ai_recommendation(position, current_price, pnl_pct)
                    
                    if ai_decision and ai_decision.confidence >= self.min_ai_confidence:
                        logger.info(f" AI RECOMMENDATION: {position.pair} - {ai_decision.action} (confidence: {ai_decision.confidence:.0f}%)")
                        logger.info(f"   Reasoning: {ai_decision.reasoning}")
                        
                        # Execute AI decision (will require approval if enabled)
                        result = self.execute_decision(position.trade_id, ai_decision)
                        if not result and self.require_manual_approval:
                            logger.info(f"   Awaiting manual approval in UI...")
                        
                        # Update position with AI feedback
                        position.last_ai_action = ai_decision.action
                        position.last_ai_reasoning = ai_decision.reasoning
                        position.last_ai_confidence = ai_decision.confidence
                        
                        # Log AI decision to journal
                        try:
                            from services.unified_trade_journal import get_unified_journal, AIDecisionLog
                            journal = get_unified_journal()
                            
                            decision_log = AIDecisionLog(
                                timestamp=datetime.now(),
                                action=ai_decision.action,
                                confidence=ai_decision.confidence,
                                reasoning=ai_decision.reasoning,
                                technical_score=ai_decision.technical_score,
                                trend_score=ai_decision.trend_score,
                                risk_score=ai_decision.risk_score,
                                new_stop=ai_decision.new_stop,
                                new_target=ai_decision.new_target,
                                partial_pct=ai_decision.partial_pct
                            )
                            
                            journal.log_ai_decision(trade_id, decision_log)
                        except Exception as e:
                            logger.debug(f"Could not log AI decision to journal: {e}")
                
                # Log status periodically
                hold_time = (datetime.now() - position.entry_time).total_seconds() / 60
                logger.debug("✓ {}: ${current_price:,.2f} (P&L: {pnl_pct:+.2f}%), ", str(position.pair)
                           f"Hold: {hold_time:.0f}min, AI: {position.last_ai_action}")
                
            except Exception as e:
                logger.error("Error monitoring position {trade_id}: {}", str(e), exc_info=True)
                continue
        
        # Save state after check
        self._save_state()
        
        return decisions
    
    def _check_basic_exit_conditions(
        self,
        position: MonitoredCryptoPosition,
        current_price: float,
        pnl_pct: float
    ) -> Optional[AITradeDecision]:
        """Check if stop loss or take profit hit"""
        
        # Stop loss check
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
                        f"📰 Fetched {len(recent_news))} recent news articles for {base_asset} "
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
            df = pd.DataFrame(ohlc_data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
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
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
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
        recent_news: List[Dict] = None,
        sentiment_score: float = None
    ) -> str:
        """
        Build prompt for AI analysis (ENHANCED with real-time news/sentiment)
        
        Args:
            recent_news: List of recent news articles with sentiment
            sentiment_score: Aggregate sentiment score (0-100)
        """
        
        prompt = f"""
Analyze this active crypto position with REAL-TIME MARKET CONTEXT and recommend the BEST action.

**Position Details:**
- Asset: {position.pair} ({position.side} position)
- Entry: ${position.entry_price:,.2f} at {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}
- Current: ${current_price:,.2f} (P&L: {pnl_pct:+.2f}%)
- Stop Loss: ${position.stop_loss:,.2f} | Take Profit: ${position.take_profit:,.2f}
- Hold Time: {hold_time:.1f} minutes
- Strategy: {position.strategy}
- Max Profit Seen: {position.max_favorable_pct:+.2f}%
- Max Drawdown: {position.max_adverse_pct:+.2f}%
- Breakeven Moved: {position.moved_to_breakeven}
- Partial Exit Taken: {position.partial_exit_taken}

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
        
        prompt += """
**Decision Framework:**
Analyze the position using these factors:
1. **Trend Strength**: Is momentum building or weakening?
2. **News Impact**: How does recent sentiment affect this position?
3. **Risk/Reward**: Is current R:R still favorable given market context?
4. **Technical Signals**: What do indicators suggest?
5. **Position Progress**: Are we near entry or near target?

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

**Respond ONLY with valid JSON (no other text):**
{
    "action": "HOLD|TIGHTEN_STOP|EXTEND_TARGET|TAKE_PARTIAL|CLOSE_NOW",
    "confidence": 0-100,
    "reasoning": "Brief 1-2 sentence explanation citing news/technical factors",
    "urgency": "LOW|MEDIUM|HIGH",
    "new_stop": price_value_or_null,
    "new_target": price_value_or_null,
    "partial_pct": percentage_or_null,
    "technical_score": 0-100,
    "trend_score": 0-100,
    "risk_score": 0-100
}
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
            logger.error(f"Position {trade_id} not found")
            return False
        
        # 🚨 SAFETY CHECK: Require manual approval unless explicitly skipped
        if self.require_manual_approval and not skip_approval:
            # Add to pending approvals queue instead of executing
            approval_id = f"{trade_id}_{int(time.time())}"
            self.pending_approvals[approval_id] = {
                'trade_id': trade_id,
                'decision': decision,
                'timestamp': datetime.now(),
                'pair': self.positions[trade_id].pair
            }
            logger.warning(f"🚨 APPROVAL REQUIRED: {decision.action} for {self.positions[trade_id].pair}")
            logger.warning(f"   Reasoning: {decision.reasoning}")
            logger.warning(f"   Approval ID: {approval_id}")
            logger.warning(f"   ⚠️ Trade will NOT execute until you approve it in the UI")
            
            # Send Discord notification for pending approval
            position = self.positions[trade_id]
            current_price = position.last_price or position.entry_price
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            
            action_emoji = {
                'CLOSE_NOW': '🚪',
                'HOLD': '✋',
                'TIGHTEN_STOP': '🎯',
                'EXTEND_TARGET': '📈',
                'TAKE_PARTIAL': '💰',
                'MOVE_TO_BREAKEVEN': '🛡️'
            }
            
            self._send_notification(
                f"{action_emoji.get(decision.action, '🤖')} AI Recommendation: {position.pair}",
                f"**Action:** {decision.action}\n"
                f"**Confidence:** {decision.confidence:.1f}%\n"
                f"**Current P&L:** {pnl_pct:+.2f}%\n"
                f"**Reasoning:** {decision.reasoning[:200]}\n\n"
                f"⚠️ **APPROVAL REQUIRED** - Check the app to approve/reject",
                color=15844367  # Orange for pending approval
            )
            
            return False  # Not executed, awaiting approval
        
        position = self.positions[trade_id]
        
        try:
            logger.info("=" * 80)
            logger.info(f"🤖 EXECUTING AI DECISION: {position.pair}")
            logger.info("=" * 80)
            logger.info(f"   Action: {decision.action}")
            logger.info(f"   Confidence: {}% {decision.confidence:.1f}")
            logger.info(f"   Reasoning: {decision.reasoning}")
            logger.info("=" * 80)
            
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
                logger.info(f"✓ HOLD - No action needed for {position.pair}")
                return True
            
            else:
                logger.warning(f"Unknown action: {decision.action}")
                return False
                
        except Exception as e:
            logger.error("Error executing decision for {trade_id}: {}", str(e), exc_info=True)
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
    
    def _save_state(self):
        """Save position state to file"""
        try:
            state = {
                'positions': {
                    trade_id: asdict(pos) for trade_id, pos in self.positions.items()
                },
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
            
            # Restore positions
            for trade_id, pos_data in state.get('positions', {}).items():
                # Convert ISO strings back to datetime
                if 'entry_time' in pos_data:
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                if 'last_check_time' in pos_data:
                    pos_data['last_check_time'] = datetime.fromisoformat(pos_data['last_check_time'])
                
                self.positions[trade_id] = MonitoredCryptoPosition(**pos_data)
            
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
        
        if self.is_running:
            logger.warning("Monitoring loop already running")
            return
        
        def monitoring_loop():
            self.is_running = True
            logger.info("=" * 80)
            logger.info("🚀 AI CRYPTO POSITION MANAGER STARTED")
            logger.info("=" * 80)
            
            while self.is_running:
                try:
                    # Monitor all positions
                    decisions = self.monitor_positions()
                    
                    # Execute AI decisions
                    for decision in decisions:
                        # Find the position for this decision
                        for trade_id, position in self.positions.items():
                            if position.status == PositionStatus.ACTIVE.value:
                                self.execute_decision(trade_id, decision)
                                break
                    
                    # Sleep until next check
                    time.sleep(self.check_interval_seconds)
                    
                except Exception as e:
                    logger.error("Error in monitoring loop: {}", str(e), exc_info=True)
                    time.sleep(10)  # Short sleep on error
            
            logger.info("🛑 AI Crypto Position Manager stopped")
        
        self.thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.thread.start()
        logger.info(f"✅ Monitoring loop started (checking every {self.check_interval_seconds}s)")
    
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

    def get_status(self) -> Dict:
        """Get current manager status"""
        return {
            'is_running': self.is_running,
            'active_positions': len([p for p in self.positions.values() if p.status == PositionStatus.ACTIVE.value]),
            'total_positions': len(self.positions),
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
                    'status': pos.status
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

