"""SetupAgent - evaluates indicators and detects trade setups"""

from __future__ import annotations

import asyncio
from loguru import logger
from datetime import datetime
from typing import Dict, List

from services.event_bus import EventBus
from services.intraday_data_service import IntradayDataService
from services.strategy_detectors import SetupScanner, MarketContext
from services.agents.messages import IndicatorEvent, TradeCandidate, Timeframe



class SetupAgent:
    """
    Subscribes to IndicatorEvent, runs setup detectors, publishes TradeCandidate.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        data_service: IntradayDataService
    ):
        """
        Args:
            event_bus: Event bus for pub/sub
            data_service: Data service for market context
        """
        self.event_bus = event_bus
        self.data_service = data_service
        self.scanner = SetupScanner()
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start setup agent"""
        if self._running:
            logger.warning("SetupAgent already running")
            return
        
        self._running = True
        logger.info("Starting SetupAgent")
        
        # Subscribe to 5-minute indicators (primary timeframe for setups)
        task = asyncio.create_task(self._listen_indicators())
        self._tasks.append(task)
    
    async def stop(self):
        """Stop setup agent"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("SetupAgent stopped")
    
    async def _listen_indicators(self):
        """Listen for indicator events and evaluate setups"""
        async for event in self.event_bus.subscribe("indicators_5min"):
            if not self._running:
                break
            
            try:
                await self._process_indicator_event(event)
            except Exception as e:
                logger.error("Error processing indicator event: {}", str(e), exc_info=True)
    
    async def _process_indicator_event(self, event: IndicatorEvent):
        """
        Process indicator event and check for trade setups.
        """
        symbol = event.symbol
        indicators = event.indicators
        
        # Build market context
        context = self._build_market_context(symbol, event.timestamp, indicators)
        
        if not context:
            return
        
        # Scan for setups
        candidates = self.scanner.scan_all(context)
        
        # Publish trade candidates
        for candidate in candidates:
            logger.info(
                f"Setup detected: {candidate.symbol} {candidate.setup_type.value} "
                f"@ ${candidate.entry_price:.2f}, confidence={candidate.confidence:.1f}%"
            )
            self.event_bus.publish("trade_candidates", candidate)
    
    def _build_market_context(
        self,
        symbol: str,
        timestamp: datetime,
        indicators_5m: Dict
    ) -> MarketContext:
        """
        Build comprehensive market context from multi-timeframe data.
        """
        # Get current price
        current_price = self.data_service.get_current_price(symbol, Timeframe.MIN_5.value)
        if not current_price:
            return None
        
        # Get recent bars
        recent_bars_5m = self.data_service.get_recent_bars(symbol, Timeframe.MIN_5.value, 20)
        recent_bars_dict = [
            {
                'open': b.open,
                'high': b.high,
                'low': b.low,
                'close': b.close,
                'volume': b.volume,
                'timestamp': b.timestamp
            }
            for b in recent_bars_5m
        ]
        
        # Get indicators for other timeframes
        indicators_15m = self.data_service.get_indicators(symbol, Timeframe.MIN_15.value)
        indicators_60m = self.data_service.get_indicators(symbol, Timeframe.MIN_60.value)
        
        # Build context
        context = MarketContext(
            symbol=symbol,
            current_price=current_price,
            timestamp=timestamp,
            
            # 5-minute data
            ema_9_5m=indicators_5m.get('ema_9'),
            ema_20_5m=indicators_5m.get('ema_20'),
            rsi_5m=indicators_5m.get('rsi'),
            volume_5m=indicators_5m.get('current_volume'),
            avg_volume_5m=indicators_5m.get('avg_volume'),
            vwap=indicators_5m.get('vwap'),
            
            # 15-minute data
            ema_9_15m=indicators_15m.get('ema_9'),
            ema_20_15m=indicators_15m.get('ema_20'),
            rsi_15m=indicators_15m.get('rsi'),
            
            # 60-minute data
            ema_9_60m=indicators_60m.get('ema_9'),
            ema_20_60m=indicators_60m.get('ema_20'),
            rsi_60m=indicators_60m.get('rsi'),
            
            # Recent bars
            recent_bars_5m=recent_bars_dict,
            
            # Key levels (TODO: implement pre-market level detection)
            key_levels={}
        )
        
        return context

