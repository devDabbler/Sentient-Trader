"""JournalAgent - records trades and updates cash manager"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List

from services.event_bus import EventBus
from services.cash_manager import CashManager
from services.journal_service import JournalService
from services.risk_limits import RiskManager
from services.agents.messages import ApprovedOrder, OrderUpdate, JournalEntry

logger = logging.getLogger(__name__)


class JournalAgent:
    """
    Subscribes to OrderUpdate events.
    Records fills in journal and updates CashManager.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        journal_service: JournalService,
        cash_manager: CashManager,
        risk_manager: RiskManager
    ):
        """
        Args:
            event_bus: Event bus for pub/sub
            journal_service: Journal service for persistence
            cash_manager: Cash manager for settlement tracking
            risk_manager: Risk manager for P&L tracking
        """
        self.event_bus = event_bus
        self.journal_service = journal_service
        self.cash_manager = cash_manager
        self.risk_manager = risk_manager
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Track open orders for fill matching
        self._pending_orders: dict = {}  # order_id -> ApprovedOrder
    
    async def start(self):
        """Start journal agent"""
        if self._running:
            logger.warning("JournalAgent already running")
            return
        
        self._running = True
        logger.info("Starting JournalAgent")
        
        # Subscribe to approved orders to track pending
        task1 = asyncio.create_task(self._listen_approved_orders())
        task2 = asyncio.create_task(self._listen_order_updates())
        
        self._tasks.extend([task1, task2])
    
    async def stop(self):
        """Stop journal agent"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("JournalAgent stopped")
    
    async def _listen_approved_orders(self):
        """Track approved orders"""
        async for order in self.event_bus.subscribe("approved_orders"):
            if not self._running:
                break
            
            # Store for later matching with fills
            # In production, use actual order ID from broker
            order_key = f"{order.symbol}_{order.timestamp.timestamp()}"
            self._pending_orders[order_key] = order
    
    async def _listen_order_updates(self):
        """Listen for order updates (fills, rejections)"""
        async for update in self.event_bus.subscribe("order_updates"):
            if not self._running:
                break
            
            try:
                await self._process_order_update(update)
            except Exception as e:
                logger.error(f"Error processing order update: {e}", exc_info=True)
    
    async def _process_order_update(self, update: OrderUpdate):
        """
        Process order update: record fills, update cash and risk state.
        """
        if update.status == 'rejected':
            logger.warning(f"Order rejected: {update.symbol} - {update.message}")
            return
        
        if update.status == 'filled' and update.filled_qty > 0:
            # Find matching approved order
            order = self._find_pending_order(update.symbol, update.timestamp)
            
            if not order:
                logger.warning(f"No pending order found for fill: {update.symbol}")
                return
            
            # Record fill with cash manager
            fill_record = self.cash_manager.record_fill(
                symbol=update.symbol,
                side=order.side,
                quantity=update.filled_qty,
                price=update.filled_price,
                fees=0.0,  # TODO: include fees if available
                filled_at=update.timestamp
            )
            
            # Create journal entry
            journal_entry = JournalEntry(
                symbol=update.symbol,
                setup_type=order.setup_type,
                mode=order.mode,
                side=order.side,
                entry_time=update.timestamp,
                entry_price=update.filled_price,
                stop_price=order.stop_price,
                target_price=order.target_price,
                shares=update.filled_qty,
                bucket_index=order.bucket_index,
                settlement_date=fill_record.settlement_date,
                settled_cash_after=self.cash_manager.get_settled_cash()
            )
            
            # Persist to journal
            trade_id = self.journal_service.record_trade(journal_entry)
            
            logger.info(
                f"Trade journaled: #{trade_id} {update.symbol} {order.side} {update.filled_qty} "
                f"@ ${update.filled_price:.2f}, settles {fill_record.settlement_date}"
            )
            
            # Remove from pending
            order_key = self._get_order_key(update.symbol, update.timestamp)
            self._pending_orders.pop(order_key, None)
    
    def _find_pending_order(self, symbol: str, timestamp: datetime) -> ApprovedOrder:
        """Find pending order matching symbol and timestamp"""
        # Try exact match first
        order_key = self._get_order_key(symbol, timestamp)
        if order_key in self._pending_orders:
            return self._pending_orders[order_key]
        
        # Try finding by symbol within time window (5 minutes)
        for key, order in self._pending_orders.items():
            if order.symbol == symbol:
                time_diff = abs((order.timestamp - timestamp).total_seconds())
                if time_diff < 300:  # Within 5 minutes
                    return order
        
        return None
    
    def _get_order_key(self, symbol: str, timestamp: datetime) -> str:
        """Generate order key"""
        return f"{symbol}_{timestamp.timestamp()}"

