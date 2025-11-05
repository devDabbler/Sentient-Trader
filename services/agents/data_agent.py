"""DataAgent - fetches and publishes bar and indicator data"""

from __future__ import annotations

import asyncio
from loguru import logger
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional

from services.event_bus import EventBus
from services.intraday_data_service import IntradayDataService, OHLCVBar
from services.agents.messages import BarEvent, IndicatorEvent, Timeframe



class DataAgent:
    """
    Fetches intraday bar data and publishes BarEvent and IndicatorEvent.
    Runs as async task per timeframe.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        data_service: IntradayDataService,
        symbols: List[str],
        timeframes: List[Timeframe] = None
    ):
        """
        Args:
            event_bus: Event bus for publishing
            data_service: Data service for bars and indicators
            symbols: List of symbols to monitor
            timeframes: List of timeframes to track (default: 5min, 15min, 60min)
        """
        self.event_bus = event_bus
        self.data_service = data_service
        self.symbols = symbols
        self.timeframes = timeframes or [Timeframe.MIN_5, Timeframe.MIN_15, Timeframe.MIN_60]
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start data agent tasks"""
        if self._running:
            logger.warning("DataAgent already running")
            return
        
        self._running = True
        logger.info(f"Starting DataAgent for {len(self.symbols)} symbols, timeframes: {self.timeframes}")
        
        # Create task per timeframe
        for tf in self.timeframes:
            task = asyncio.create_task(self._fetch_loop(tf))
            self._tasks.append(task)
    
    async def stop(self):
        """Stop data agent"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("DataAgent stopped")
    
    async def _fetch_loop(self, timeframe: Timeframe):
        """
        Fetch loop for a specific timeframe.
        Polls broker/data source at appropriate intervals.
        """
        interval_seconds = self._get_interval_seconds(timeframe)
        
        while self._running:
            try:
                await self._fetch_and_publish(timeframe)
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in fetch loop for {timeframe}: {e}", exc_info=True)
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _fetch_and_publish(self, timeframe: Timeframe):
        """
        Fetch latest bars for all symbols and publish events.
        In production, this would fetch from broker API.
        For now, we simulate with mock data.
        """
        now = datetime.now()
        
        # Skip if outside market hours
        if not self._is_market_hours(now):
            return
        
        for symbol in self.symbols:
            try:
                # In production: fetch real data from broker
                # For now: simulate or use cached data
                bar = await self._fetch_bar(symbol, timeframe, now)
                
                if bar:
                    # Add to data service
                    self.data_service.add_bar(symbol, timeframe.value, bar)
                    
                    # Publish BarEvent
                    bar_event = BarEvent(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume
                    )
                    self.event_bus.publish(f"bars_{timeframe.value}", bar_event)
                    
                    # Calculate and publish indicators
                    indicators = self.data_service.get_indicators(symbol, timeframe.value, force_recalc=True)
                    if indicators:
                        ind_event = IndicatorEvent(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=now,
                            indicators=indicators
                        )
                        self.event_bus.publish(f"indicators_{timeframe.value}", ind_event)
            
            except Exception as e:
                logger.error(f"Error fetching {symbol} {timeframe}: {e}")
    
    async def _fetch_bar(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> Optional[OHLCVBar]:
        """
        Fetch a single bar from data source.
        
        TODO: Integrate with real broker API (IBKR, Tradier, etc.)
        For now, returns None (can be extended with mock data for testing).
        """
        # Placeholder - in production this would fetch from broker
        # Example:
        # bars = await broker_client.get_bars(symbol, timeframe, count=1)
        # return bars[-1] if bars else None
        return None
    
    def _get_interval_seconds(self, timeframe: Timeframe) -> int:
        """Get polling interval for timeframe"""
        intervals = {
            Timeframe.MIN_1: 60,
            Timeframe.MIN_5: 60,  # Poll every minute to catch bar closes
            Timeframe.MIN_15: 60,
            Timeframe.MIN_60: 60,
            Timeframe.DAILY: 300  # Poll every 5 minutes
        }
        return intervals.get(timeframe, 60)
    
    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if during market hours"""
        if dt.weekday() >= 5:  # Weekend
            return False
        
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = dt.time()
        
        return market_open <= current_time <= market_close

