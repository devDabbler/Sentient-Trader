"""RiskAgent - validates and sizes trades based on risk parameters"""

from __future__ import annotations

import asyncio
from loguru import logger
from datetime import datetime, timedelta
from typing import List

from services.event_bus import EventBus
from services.cash_manager import CashManager
from services.risk_limits import RiskManager
from services.agents.messages import TradeCandidate, ApprovedOrder, TradingMode



class RiskAgent:
    """
    Subscribes to TradeCandidate, applies risk checks and position sizing.
    Publishes ApprovedOrder or logs rejection.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        cash_manager: CashManager,
        risk_manager: RiskManager,
        account_equity: float
    ):
        """
        Args:
            event_bus: Event bus for pub/sub
            cash_manager: Cash manager for settled funds
            risk_manager: Risk manager for limits
            account_equity: Total account equity
        """
        self.event_bus = event_bus
        self.cash_manager = cash_manager
        self.risk_manager = risk_manager
        self.account_equity = account_equity
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start risk agent"""
        if self._running:
            logger.warning("RiskAgent already running")
            return
        
        self._running = True
        logger.info("Starting RiskAgent")
        
        task = asyncio.create_task(self._listen_candidates())
        self._tasks.append(task)
    
    async def stop(self):
        """Stop risk agent"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("RiskAgent stopped")
    
    async def _listen_candidates(self):
        """Listen for trade candidates"""
        async for candidate in self.event_bus.subscribe("trade_candidates"):
            if not self._running:
                break
            
            try:
                await self._process_candidate(candidate)
            except Exception as e:
                logger.error("Error processing candidate: {}", str(e), exc_info=True)
    
    async def _process_candidate(self, candidate: TradeCandidate):
        """
        Process trade candidate: validate, size, and approve/reject.
        """
        symbol = candidate.symbol
        mode = candidate.mode
        
        # Check if we can enter trade
        can_enter, reason = self.risk_manager.can_enter_trade(symbol, mode)
        if not can_enter:
            logger.warning(f"Trade rejected: {symbol} - {reason}")
            self.event_bus.publish("rejected_trades", {
                'candidate': candidate,
                'reason': reason
            })
            return
        
        # Get settled cash
        settled_cash = self.cash_manager.get_settled_cash()
        
        # Select bucket
        bucket_idx = self.cash_manager.select_active_bucket()
        bucket_cash = self.cash_manager.bucket_target_cash(settled_cash, bucket_idx)
        
        # Get risk-adjusted sizing multiplier
        sizing_multiplier = self.risk_manager.get_risk_adjusted_sizing_multiplier()
        
        # Calculate position size by risk
        risk_perc = 0.02 * sizing_multiplier  # Base 2% risk, adjusted
        shares = self.cash_manager.compute_position_size_by_risk(
            account_equity=self.account_equity,
            risk_perc=risk_perc,
            entry_price=candidate.entry_price,
            stop_price=candidate.stop_price
        )
        
        # Clamp to settled cash available in bucket
        shares = self.cash_manager.clamp_to_settled_cash(
            shares=shares,
            entry_price=candidate.entry_price,
            settled_cash=bucket_cash,
            reserve_pct=0.05  # Reserve 5%
        )
        
        # Additional max position size check (e.g., 20% of equity)
        max_shares_by_pct = int((self.account_equity * 0.20) / candidate.entry_price)
        shares = min(shares, max_shares_by_pct)
        
        if shares <= 0:
            logger.warning(f"Trade rejected: {symbol} - No settled cash available")
            self.event_bus.publish("rejected_trades", {
                'candidate': candidate,
                'reason': 'Insufficient settled cash'
            })
            return
        
        # Calculate estimated settlement
        settlement_days = self.cash_manager.config.t_plus_days
        estimated_settlement = datetime.now() + timedelta(days=settlement_days)
        
        # Determine order duration based on mode
        duration = 'day' if mode == TradingMode.SLOW_SCALPER else 'gtc'
        
        # Determine side
        side = 'BUY'  # Default to BUY for long setups
        if candidate.metadata and candidate.metadata.get('direction') == 'short':
            side = 'SELL'
        
        # Create tag
        tag = f"AUTO_{mode.value}_{candidate.setup_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create approved order
        approved = ApprovedOrder(
            symbol=symbol,
            setup_type=candidate.setup_type,
            mode=mode,
            timestamp=datetime.now(),
            side=side,
            quantity=shares,
            entry_price=candidate.entry_price,
            stop_price=candidate.stop_price,
            target_price=candidate.target_price,
            bucket_index=bucket_idx,
            estimated_settlement=estimated_settlement,
            tag=tag,
            duration=duration,
            metadata=candidate.metadata
        )
        
        logger.info(
            f"Trade approved: {symbol} {side} {shares} shares @ ${candidate.entry_price:.2f}, "
            f"stop=${candidate.stop_price:.2f}, target=${candidate.target_price:.2f}, "
            f"bucket={bucket_idx}, risk={risk_perc*100:.2f}%"
        )
        
        # Record entry with risk manager
        self.risk_manager.record_entry(symbol, mode)
        
        # Publish approved order
        self.event_bus.publish("approved_orders", approved)

