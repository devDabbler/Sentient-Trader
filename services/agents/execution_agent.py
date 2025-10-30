"""ExecutionAgent - places broker orders with retry logic"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import List, Optional

from services.event_bus import EventBus
from services.agents.messages import ApprovedOrder, OrderUpdate

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    Subscribes to ApprovedOrder, places broker orders with retry.
    Publishes OrderUpdate on fills/rejections.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        tradier_client,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0
    ):
        """
        Args:
            event_bus: Event bus for pub/sub
            tradier_client: Broker client (e.g., TradierClient)
            max_retries: Max retry attempts per order
            retry_delay_seconds: Base delay between retries (with jitter)
        """
        self.event_bus = event_bus
        self.tradier_client = tradier_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start execution agent"""
        if self._running:
            logger.warning("ExecutionAgent already running")
            return
        
        self._running = True
        logger.info("Starting ExecutionAgent")
        
        task = asyncio.create_task(self._listen_approved_orders())
        self._tasks.append(task)
    
    async def stop(self):
        """Stop execution agent"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("ExecutionAgent stopped")
    
    async def _listen_approved_orders(self):
        """Listen for approved orders"""
        async for order in self.event_bus.subscribe("approved_orders"):
            if not self._running:
                break
            
            try:
                await self._execute_order(order)
            except Exception as e:
                logger.error(f"Error executing order: {e}", exc_info=True)
    
    async def _execute_order(self, order: ApprovedOrder):
        """
        Execute order with retry logic and jitter.
        """
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            attempt += 1
            
            try:
                # Place bracket order
                success, result = await self._place_bracket_order(order)
                
                if success:
                    # Publish success update
                    update = OrderUpdate(
                        symbol=order.symbol,
                        order_id=str(result.get('id', 'unknown')),
                        status='pending',
                        timestamp=order.timestamp,
                        message=f"Order placed successfully (attempt {attempt})"
                    )
                    self.event_bus.publish("order_updates", update)
                    logger.info(f"Order placed: {order.symbol} {order.side} {order.quantity} @ ${order.entry_price:.2f}")
                    return
                else:
                    last_error = result
                    logger.warning(f"Order placement failed (attempt {attempt}/{self.max_retries}): {result}")
            
            except Exception as e:
                last_error = str(e)
                logger.error(f"Exception placing order (attempt {attempt}/{self.max_retries}): {e}")
            
            # Retry with jitter
            if attempt < self.max_retries:
                jitter = random.uniform(0, 0.5)
                delay = self.retry_delay * attempt + jitter
                await asyncio.sleep(delay)
        
        # All retries exhausted
        update = OrderUpdate(
            symbol=order.symbol,
            order_id=None,
            status='rejected',
            timestamp=order.timestamp,
            message=f"Order rejected after {self.max_retries} attempts: {last_error}"
        )
        self.event_bus.publish("order_updates", update)
        logger.error(f"Order rejected: {order.symbol} - {last_error}")
    
    async def _place_bracket_order(self, order: ApprovedOrder) -> tuple[bool, dict]:
        """
        Place bracket order with broker.
        Returns (success, result_dict)
        """
        try:
            # Use asyncio to run synchronous broker call
            loop = asyncio.get_event_loop()
            success, result = await loop.run_in_executor(
                None,
                self._sync_place_bracket_order,
                order
            )
            return success, result
        except Exception as e:
            logger.error(f"Error in place_bracket_order: {e}", exc_info=True)
            return False, {'error': str(e)}
    
    def _sync_place_bracket_order(self, order: ApprovedOrder) -> tuple[bool, dict]:
        """
        Synchronous broker call (runs in executor).
        """
        try:
            success, result = self.tradier_client.place_bracket_order(
                symbol=order.symbol,
                side=order.side.lower(),
                quantity=order.quantity,
                entry_price=order.entry_price,
                take_profit_price=order.target_price,
                stop_loss_price=order.stop_price,
                duration=order.duration,
                tag=order.tag
            )
            return success, result
        except Exception as e:
            return False, {'error': str(e)}

