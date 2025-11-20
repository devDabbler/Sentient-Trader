"""
Fair Value Gap (FVG) Detection Service

A Fair Value Gap (FVG) is an inefficient price zone where there is no overlap 
between three consecutive candles. These gaps represent imbalances in supply/demand
and often act as strong support/resistance zones.

Bullish FVG: Gap up with no overlap (candle 1 high < candle 3 low)
Bearish FVG: Gap down with no overlap (candle 1 low > candle 3 high)
"""

from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap"""
    gap_type: str  # 'bullish' or 'bearish'
    top: float  # Upper boundary of the gap
    bottom: float  # Lower boundary of the gap
    timestamp: datetime  # When the gap was created
    strength: float  # 0-100 score based on gap size and volume
    mitigated: bool = False  # Whether gap has been filled
    
    def size(self) -> float:
        """Return gap size in price units"""
        return self.top - self.bottom
    
    def size_pct(self) -> float:
        """Return gap size as percentage"""
        return (self.size() / self.bottom) * 100
    
    def contains_price(self, price: float) -> bool:
        """Check if price is within the gap"""
        return self.bottom <= price <= self.top
    
    def is_filled(self, price: float) -> bool:
        """Check if gap has been filled by current price"""
        if self.gap_type == 'bullish':
            # Bullish FVG filled if price comes back down into gap
            return price <= (self.bottom + (self.size() * 0.5))
        else:
            # Bearish FVG filled if price comes back up into gap
            return price >= (self.top - (self.size() * 0.5))


class FVGDetector:
    """Detects Fair Value Gaps in price action"""
    
    def __init__(self, min_gap_pct: float = 0.2, max_lookback: int = 50):
        """
        Initialize FVG detector
        
        Args:
            min_gap_pct: Minimum gap size as % of price (default 0.2%)
            max_lookback: Maximum number of historical FVGs to track
        """
        self.min_gap_pct = min_gap_pct
        self.max_lookback = max_lookback
        self._active_fvgs: Dict[str, List[FairValueGap]] = {}
    
    def detect_fvgs(self, 
                    symbol: str,
                    candles: List[Dict],
                    current_price: float) -> List[FairValueGap]:
        """
        Detect FVGs in recent candle data
        
        Args:
            symbol: Stock ticker
            candles: List of candle dicts with keys: open, high, low, close, volume, timestamp
            current_price: Current market price
            
        Returns:
            List of active (unfilled) FVGs
        """
        if len(candles) < 3:
            return []
        
        detected_fvgs = []
        
        # Scan through candles looking for 3-candle FVG patterns
        for i in range(len(candles) - 2):
            candle_1 = candles[i]
            candle_2 = candles[i + 1]
            candle_3 = candles[i + 2]
            
            # Bullish FVG: Candle 1 high < Candle 3 low (gap up)
            if candle_1['high'] < candle_3['low']:
                gap_bottom = candle_1['high']
                gap_top = candle_3['low']
                gap_size_pct = ((gap_top - gap_bottom) / gap_bottom) * 100
                
                if gap_size_pct >= self.min_gap_pct:
                    # Calculate strength based on gap size and volume
                    volume_ratio = (candle_2['volume'] + candle_3['volume']) / (candle_1['volume'] + 1)
                    strength = min(100, (gap_size_pct * 10) + (volume_ratio * 5))
                    
                    fvg = FairValueGap(
                        gap_type='bullish',
                        top=gap_top,
                        bottom=gap_bottom,
                        timestamp=candle_3.get('timestamp', datetime.now()),
                        strength=strength
                    )
                    
                    # Check if gap is still active (not filled)
                    if not fvg.is_filled(current_price):
                        detected_fvgs.append(fvg)
            
            # Bearish FVG: Candle 1 low > Candle 3 high (gap down)
            elif candle_1['low'] > candle_3['high']:
                gap_top = candle_1['low']
                gap_bottom = candle_3['high']
                gap_size_pct = ((gap_top - gap_bottom) / gap_bottom) * 100
                
                if gap_size_pct >= self.min_gap_pct:
                    volume_ratio = (candle_2['volume'] + candle_3['volume']) / (candle_1['volume'] + 1)
                    strength = min(100, (gap_size_pct * 10) + (volume_ratio * 5))
                    
                    fvg = FairValueGap(
                        gap_type='bearish',
                        top=gap_top,
                        bottom=gap_bottom,
                        timestamp=candle_3.get('timestamp', datetime.now()),
                        strength=strength
                    )
                    
                    if not fvg.is_filled(current_price):
                        detected_fvgs.append(fvg)
        
        # Store active FVGs for this symbol
        self._active_fvgs[symbol] = detected_fvgs[-self.max_lookback:]
        
        return detected_fvgs
    
    def get_nearest_fvg(self, 
                       symbol: str,
                       current_price: float,
                       gap_type: Optional[str] = None) -> Optional[FairValueGap]:
        """
        Get the nearest FVG to current price
        
        Args:
            symbol: Stock ticker
            current_price: Current price
            gap_type: Filter by 'bullish' or 'bearish' (None = any)
            
        Returns:
            Nearest FVG or None
        """
        if symbol not in self._active_fvgs:
            return None
        
        active_fvgs = self._active_fvgs[symbol]
        
        # Filter by type if specified
        if gap_type:
            active_fvgs = [fvg for fvg in active_fvgs if fvg.gap_type == gap_type]
        
        if not active_fvgs:
            return None
        
        # Find nearest FVG by distance to current price
        def distance_to_gap(fvg: FairValueGap) -> float:
            gap_mid = (fvg.top + fvg.bottom) / 2
            return abs(current_price - gap_mid)
        
        return min(active_fvgs, key=distance_to_gap)
    
    def is_price_in_fvg(self, 
                       symbol: str,
                       current_price: float,
                       gap_type: Optional[str] = None) -> Optional[FairValueGap]:
        """
        Check if current price is within any FVG
        
        Args:
            symbol: Stock ticker
            current_price: Current price
            gap_type: Filter by type (None = any)
            
        Returns:
            FVG containing price, or None
        """
        if symbol not in self._active_fvgs:
            return None
        
        active_fvgs = self._active_fvgs[symbol]
        
        # Filter by type
        if gap_type:
            active_fvgs = [fvg for fvg in active_fvgs if fvg.gap_type == gap_type]
        
        # Return first FVG that contains the price
        for fvg in active_fvgs:
            if fvg.contains_price(current_price):
                return fvg
        
        return None
    
    def get_fvg_support_resistance(self,
                                   symbol: str,
                                   current_price: float) -> Dict[str, List[float]]:
        """
        Get FVG-based support and resistance levels
        
        Args:
            symbol: Stock ticker
            current_price: Current price
            
        Returns:
            Dict with 'support' and 'resistance' lists
        """
        if symbol not in self._active_fvgs:
            return {'support': [], 'resistance': []}
        
        support_levels = []
        resistance_levels = []
        
        for fvg in self._active_fvgs[symbol]:
            gap_mid = (fvg.top + fvg.bottom) / 2
            
            # FVG below current price = support
            if gap_mid < current_price:
                support_levels.append(gap_mid)
            # FVG above current price = resistance
            elif gap_mid > current_price:
                resistance_levels.append(gap_mid)
        
        return {
            'support': sorted(support_levels, reverse=True),  # Nearest first
            'resistance': sorted(resistance_levels)  # Nearest first
        }
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear FVG cache"""
        if symbol:
            self._active_fvgs.pop(symbol, None)
        else:
            self._active_fvgs.clear()
