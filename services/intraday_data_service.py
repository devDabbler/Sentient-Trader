"""Intraday data fetching and indicator calculation service"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OHLCVBar:
    """OHLCV bar data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class RingBuffer:
    """Ring buffer for efficient bar storage"""
    maxlen: int
    bars: Deque[OHLCVBar] = field(default_factory=deque)
    
    def __post_init__(self):
        self.bars = deque(maxlen=self.maxlen)
    
    def append(self, bar: OHLCVBar):
        """Add bar to buffer"""
        self.bars.append(bar)
    
    def get_recent(self, n: int) -> List[OHLCVBar]:
        """Get last N bars"""
        return list(self.bars)[-n:]
    
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays for indicator calculation"""
        if not self.bars:
            return {
                'open': np.array([]),
                'high': np.array([]),
                'low': np.array([]),
                'close': np.array([]),
                'volume': np.array([])
            }
        
        return {
            'open': np.array([b.open for b in self.bars]),
            'high': np.array([b.high for b in self.bars]),
            'low': np.array([b.low for b in self.bars]),
            'close': np.array([b.close for b in self.bars]),
            'volume': np.array([b.volume for b in self.bars])
        }


class IndicatorCalculator:
    """Incremental indicator calculations"""
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> Optional[float]:
        """
        Calculate EMA for given prices.
        Returns None if insufficient data.
        """
        if len(prices) < period:
            return None
        
        alpha = 2 / (period + 1)
        ema_val = prices[0]
        for price in prices[1:]:
            ema_val = alpha * price + (1 - alpha) * ema_val
        return float(ema_val)
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    @staticmethod
    def vwap(bars: List[OHLCVBar]) -> Optional[float]:
        """Calculate VWAP from bars"""
        if not bars:
            return None
        
        total_pv = 0.0
        total_v = 0.0
        
        for bar in bars:
            typical_price = (bar.high + bar.low + bar.close) / 3
            total_pv += typical_price * bar.volume
            total_v += bar.volume
        
        if total_v == 0:
            return None
        
        return total_pv / total_v
    
    @staticmethod
    def average_volume(volumes: np.ndarray, period: int = 20) -> Optional[float]:
        """Calculate average volume"""
        if len(volumes) < period:
            return None
        return float(np.mean(volumes[-period:]))


class IntradayDataService:
    """
    Manages intraday bar data and indicator calculation.
    Uses ring buffers for memory efficiency.
    """
    
    def __init__(self, max_bars_per_timeframe: int = 390):
        """
        Args:
            max_bars_per_timeframe: Max bars to keep (390 = full day of 1-min bars)
        """
        self.max_bars = max_bars_per_timeframe
        
        # symbol -> timeframe -> RingBuffer
        self._data: Dict[str, Dict[str, RingBuffer]] = {}
        
        # Cached indicators (symbol -> timeframe -> indicators dict)
        self._cached_indicators: Dict[str, Dict[str, Dict]] = {}
        self._cache_ttl_seconds = 60  # Re-calculate every minute
        self._last_calc_time: Dict[str, datetime] = {}
    
    def add_bar(self, symbol: str, timeframe: str, bar: OHLCVBar):
        """Add a new bar for symbol/timeframe"""
        if symbol not in self._data:
            self._data[symbol] = {}
        
        if timeframe not in self._data[symbol]:
            self._data[symbol][timeframe] = RingBuffer(maxlen=self.max_bars)
        
        self._data[symbol][timeframe].append(bar)
        
        # Invalidate cache for this symbol/timeframe
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._last_calc_time:
            del self._last_calc_time[cache_key]
    
    def get_recent_bars(self, symbol: str, timeframe: str, n: int) -> List[OHLCVBar]:
        """Get last N bars"""
        if symbol not in self._data or timeframe not in self._data[symbol]:
            return []
        return self._data[symbol][timeframe].get_recent(n)
    
    def get_indicators(self, symbol: str, timeframe: str, force_recalc: bool = False) -> Dict:
        """
        Get calculated indicators for symbol/timeframe.
        Uses caching to avoid redundant calculations.
        
        Returns dict with keys: ema_9, ema_20, rsi, vwap, avg_volume
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache
        if not force_recalc and cache_key in self._last_calc_time:
            elapsed = (datetime.now() - self._last_calc_time[cache_key]).total_seconds()
            if elapsed < self._cache_ttl_seconds:
                return self._cached_indicators.get(symbol, {}).get(timeframe, {})
        
        # Calculate indicators
        if symbol not in self._data or timeframe not in self._data[symbol]:
            return {}
        
        buffer = self._data[symbol][timeframe]
        arrays = buffer.to_arrays()
        
        if len(arrays['close']) == 0:
            return {}
        
        indicators = {
            'ema_9': IndicatorCalculator.ema(arrays['close'], 9),
            'ema_20': IndicatorCalculator.ema(arrays['close'], 20),
            'rsi': IndicatorCalculator.rsi(arrays['close'], 14),
            'vwap': IndicatorCalculator.vwap(list(buffer.bars)),
            'avg_volume': IndicatorCalculator.average_volume(arrays['volume'], 20),
            'current_volume': float(arrays['volume'][-1]) if len(arrays['volume']) > 0 else None
        }
        
        # Update cache
        if symbol not in self._cached_indicators:
            self._cached_indicators[symbol] = {}
        self._cached_indicators[symbol][timeframe] = indicators
        self._last_calc_time[cache_key] = datetime.now()
        
        return indicators
    
    def get_current_price(self, symbol: str, timeframe: str) -> Optional[float]:
        """Get latest close price"""
        bars = self.get_recent_bars(symbol, timeframe, 1)
        if not bars:
            return None
        return bars[0].close
    
    def clear_symbol(self, symbol: str):
        """Clear all data for a symbol"""
        if symbol in self._data:
            del self._data[symbol]
        if symbol in self._cached_indicators:
            del self._cached_indicators[symbol]
    
    def clear_all(self):
        """Clear all data"""
        self._data.clear()
        self._cached_indicators.clear()
        self._last_calc_time.clear()


# Singleton instance
_data_service: Optional[IntradayDataService] = None


def get_data_service() -> IntradayDataService:
    """Get or create singleton data service"""
    global _data_service
    if _data_service is None:
        _data_service = IntradayDataService()
    return _data_service

