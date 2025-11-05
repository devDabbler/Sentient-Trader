"""In-process asyncio event bus for agent communication"""

from __future__ import annotations

import asyncio
from loguru import logger
from collections import defaultdict
from typing import Any, AsyncIterator, Dict, Set
from dataclasses import dataclass



@dataclass
class Subscription:
    """Subscription to a topic"""
    topic: str
    queue: asyncio.Queue


class EventBus:
    """Asyncio-based in-process event bus"""
    
    def __init__(self, max_queue_size: int = 1000):
        self._subscriptions: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._max_queue_size = max_queue_size
        self._stats: Dict[str, int] = defaultdict(int)
    
    def publish(self, topic: str, message: Any) -> int:
        """
        Publish message to topic. Returns number of subscribers notified.
        """
        if topic not in self._subscriptions:
            logger.debug(f"No subscribers for topic '{topic}'")
            return 0
        
        count = 0
        for queue in self._subscriptions[topic]:
            try:
                queue.put_nowait(message)
                count += 1
            except asyncio.QueueFull:
                logger.warning(f"Queue full for topic '{topic}', dropping message")
        
        self._stats[f"pub_{topic}"] += 1
        return count
    
    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        """
        Subscribe to topic. Returns async iterator of messages.
        
        Usage:
            async for message in event_bus.subscribe('bar_events'):
                process(message)
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscriptions[topic].add(queue)
        logger.info(f"Subscribed to topic '{topic}'")
        
        try:
            while True:
                message = await queue.get()
                self._stats[f"sub_{topic}"] += 1
                yield message
        finally:
            self._subscriptions[topic].discard(queue)
            logger.info(f"Unsubscribed from topic '{topic}'")
    
    def get_stats(self) -> Dict[str, int]:
        """Get event statistics"""
        return dict(self._stats)
    
    def clear_stats(self):
        """Reset statistics"""
        self._stats.clear()


# Singleton instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get or create singleton event bus"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus():
    """Reset event bus (mainly for testing)"""
    global _event_bus
    _event_bus = None

