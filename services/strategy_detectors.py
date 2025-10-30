"""Strategy detection algorithms for PDT-safe trading"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Tuple
import numpy as np

from services.agents.messages import SetupType, TradingMode, TradeCandidate

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Multi-timeframe market context for a symbol"""
    symbol: str
    current_price: float
    timestamp: datetime
    
    # 5-minute data
    ema_9_5m: Optional[float] = None
    ema_20_5m: Optional[float] = None
    rsi_5m: Optional[float] = None
    volume_5m: Optional[float] = None
    avg_volume_5m: Optional[float] = None
    vwap: Optional[float] = None
    
    # 15-minute data
    ema_9_15m: Optional[float] = None
    ema_20_15m: Optional[float] = None
    rsi_15m: Optional[float] = None
    
    # 60-minute data
    ema_9_60m: Optional[float] = None
    ema_20_60m: Optional[float] = None
    rsi_60m: Optional[float] = None
    
    # Recent bars (last N candles)
    recent_bars_5m: List[Dict] = None  # List of {open, high, low, close, volume, timestamp}
    
    # Key levels
    key_levels: Optional[Dict[str, float]] = None  # {'prev_day_high', 'prev_day_low', 'overnight_high', 'overnight_low'}


class ORBDetector:
    """Opening Range Breakout detector"""
    
    def __init__(self, orb_minutes: int = 15):
        """
        Args:
            orb_minutes: Opening range period in minutes (default 15)
        """
        self.orb_minutes = orb_minutes
        self._orb_cache: Dict[str, Dict] = {}  # symbol -> {high, low, timestamp}
    
    def detect(self, context: MarketContext) -> Optional[TradeCandidate]:
        """
        Detect ORB setup:
        - First 15 minutes after open: establish high/low
        - 5-min candle closes through high/low with volume > 1.5x avg
        - Stop at range opposite
        - Target: 1.5-2R
        """
        now = context.timestamp.time()
        market_open = dt_time(9, 30)
        orb_end = dt_time(9, 30 + self.orb_minutes)
        
        # Build opening range during first N minutes
        if market_open <= now < orb_end:
            if context.symbol not in self._orb_cache:
                self._orb_cache[context.symbol] = {
                    'high': context.current_price,
                    'low': context.current_price,
                    'timestamp': context.timestamp
                }
            else:
                orb = self._orb_cache[context.symbol]
                orb['high'] = max(orb['high'], context.current_price)
                orb['low'] = min(orb['low'], context.current_price)
            return None
        
        # After opening range, look for breakout
        if context.symbol not in self._orb_cache:
            return None
        
        orb = self._orb_cache[context.symbol]
        orb_high = orb['high']
        orb_low = orb['low']
        orb_range = orb_high - orb_low
        
        if orb_range < 0.01:  # Range too small
            return None
        
        # Volume confirmation
        if not context.recent_bars_5m or len(context.recent_bars_5m) < 1:
            return None
        
        last_bar = context.recent_bars_5m[-1]
        if context.avg_volume_5m and context.avg_volume_5m > 0:
            volume_ratio = last_bar['volume'] / context.avg_volume_5m
            if volume_ratio < 1.5:
                return None
        
        # Bullish breakout: close above orb_high
        if last_bar['close'] > orb_high and context.current_price > orb_high:
            entry = context.current_price
            stop = orb_low
            risk = entry - stop
            target = entry + (1.5 * risk)  # 1.5R target
            
            confidence = min(95.0, 60.0 + (volume_ratio * 10) + (risk / entry * 100))
            
            return TradeCandidate(
                symbol=context.symbol,
                setup_type=SetupType.ORB,
                mode=TradingMode.SLOW_SCALPER,
                timestamp=context.timestamp,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                confidence=confidence,
                metadata={
                    'orb_high': orb_high,
                    'orb_low': orb_low,
                    'volume_ratio': volume_ratio,
                    'r_multiple': 1.5
                }
            )
        
        # Bearish breakout: close below orb_low
        elif last_bar['close'] < orb_low and context.current_price < orb_low:
            entry = context.current_price
            stop = orb_high
            risk = stop - entry
            target = entry - (1.5 * risk)  # 1.5R target
            
            confidence = min(95.0, 60.0 + (volume_ratio * 10) + (risk / entry * 100))
            
            return TradeCandidate(
                symbol=context.symbol,
                setup_type=SetupType.ORB,
                mode=TradingMode.SLOW_SCALPER,
                timestamp=context.timestamp,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                confidence=confidence,
                metadata={
                    'orb_high': orb_high,
                    'orb_low': orb_low,
                    'volume_ratio': volume_ratio,
                    'r_multiple': 1.5,
                    'direction': 'short'
                }
            )
        
        return None
    
    def clear_cache(self):
        """Clear ORB cache (call at end of day)"""
        self._orb_cache.clear()


class VWAPBounceDetector:
    """VWAP bounce detector"""
    
    def detect(self, context: MarketContext) -> Optional[TradeCandidate]:
        """
        Detect VWAP bounce:
        - Trend direction confirmed (9 EMA > 20 EMA for uptrend)
        - Price touches VWAP with rejection candle
        - Volume > 1.3x average
        - Invalidate if close beyond VWAP by 0.5%
        """
        if not context.vwap:
            return None
        
        # Check trend alignment
        if not (context.ema_9_5m and context.ema_20_5m):
            return None
        
        is_uptrend = context.ema_9_5m > context.ema_20_5m
        is_downtrend = context.ema_9_5m < context.ema_20_5m
        
        if not (is_uptrend or is_downtrend):
            return None
        
        # Volume check
        if context.avg_volume_5m and context.volume_5m:
            volume_ratio = context.volume_5m / context.avg_volume_5m
            if volume_ratio < 1.3:
                return None
        else:
            volume_ratio = 1.5  # Default if no data
        
        # Check for bounce in recent bars
        if not context.recent_bars_5m or len(context.recent_bars_5m) < 2:
            return None
        
        last_bar = context.recent_bars_5m[-1]
        prev_bar = context.recent_bars_5m[-2]
        
        vwap = context.vwap
        threshold = vwap * 0.005  # 0.5% threshold
        
        # Bullish bounce in uptrend
        if is_uptrend:
            # Price touched/came close to VWAP and bounced
            touched_vwap = prev_bar['low'] <= vwap * 1.01
            bounced = last_bar['close'] > vwap and last_bar['close'] > last_bar['open']
            not_broken = last_bar['close'] > (vwap - threshold)
            
            if touched_vwap and bounced and not_broken:
                entry = context.current_price
                stop = vwap - (vwap * 0.01)  # 1% below VWAP
                risk = entry - stop
                target = entry + (2.0 * risk)  # 2R target
                
                confidence = min(95.0, 65.0 + (volume_ratio * 8))
                
                return TradeCandidate(
                    symbol=context.symbol,
                    setup_type=SetupType.VWAP_BOUNCE,
                    mode=TradingMode.SLOW_SCALPER,
                    timestamp=context.timestamp,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    confidence=confidence,
                    metadata={
                        'vwap': vwap,
                        'volume_ratio': volume_ratio,
                        'direction': 'long',
                        'r_multiple': 2.0
                    }
                )
        
        # Bearish bounce in downtrend
        elif is_downtrend:
            touched_vwap = prev_bar['high'] >= vwap * 0.99
            bounced = last_bar['close'] < vwap and last_bar['close'] < last_bar['open']
            not_broken = last_bar['close'] < (vwap + threshold)
            
            if touched_vwap and bounced and not_broken:
                entry = context.current_price
                stop = vwap + (vwap * 0.01)  # 1% above VWAP
                risk = stop - entry
                target = entry - (2.0 * risk)  # 2R target
                
                confidence = min(95.0, 65.0 + (volume_ratio * 8))
                
                return TradeCandidate(
                    symbol=context.symbol,
                    setup_type=SetupType.VWAP_BOUNCE,
                    mode=TradingMode.SLOW_SCALPER,
                    timestamp=context.timestamp,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    confidence=confidence,
                    metadata={
                        'vwap': vwap,
                        'volume_ratio': volume_ratio,
                        'direction': 'short',
                        'r_multiple': 2.0
                    }
                )
        
        return None


class KeyLevelRejectionDetector:
    """Key level rejection detector"""
    
    def detect(self, context: MarketContext) -> Optional[TradeCandidate]:
        """
        Detect key level rejection:
        - Pre-market derive 2-3 levels (prev day high/low, overnight high/low)
        - Price approaches level with decreasing volume
        - Reversal confirmation candle
        """
        if not context.key_levels:
            return None
        
        if not context.recent_bars_5m or len(context.recent_bars_5m) < 3:
            return None
        
        # Analyze volume trend
        volumes = [bar['volume'] for bar in context.recent_bars_5m[-3:]]
        volume_decreasing = volumes[-1] < volumes[-2] < volumes[-3]
        
        last_bar = context.recent_bars_5m[-1]
        current_price = context.current_price
        
        # Check proximity to key levels
        for level_name, level_price in context.key_levels.items():
            if not level_price:
                continue
            
            distance_pct = abs(current_price - level_price) / level_price
            
            # Within 1% of level
            if distance_pct > 0.01:
                continue
            
            # Resistance rejection (bearish)
            if level_name in ['prev_day_high', 'overnight_high']:
                rejection = (last_bar['high'] >= level_price * 0.995 and 
                           last_bar['close'] < last_bar['open'] and
                           current_price < level_price)
                
                if rejection and volume_decreasing:
                    entry = current_price
                    stop = level_price * 1.005
                    risk = stop - entry
                    target = entry - (1.5 * risk)
                    
                    confidence = 70.0
                    
                    return TradeCandidate(
                        symbol=context.symbol,
                        setup_type=SetupType.KEY_LEVEL_REJECTION,
                        mode=TradingMode.MICRO_SWING,
                        timestamp=context.timestamp,
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        confidence=confidence,
                        metadata={
                            'level_name': level_name,
                            'level_price': level_price,
                            'direction': 'short',
                            'r_multiple': 1.5
                        }
                    )
            
            # Support bounce (bullish)
            elif level_name in ['prev_day_low', 'overnight_low']:
                bounce = (last_bar['low'] <= level_price * 1.005 and 
                         last_bar['close'] > last_bar['open'] and
                         current_price > level_price)
                
                if bounce and volume_decreasing:
                    entry = current_price
                    stop = level_price * 0.995
                    risk = entry - stop
                    target = entry + (1.5 * risk)
                    
                    confidence = 70.0
                    
                    return TradeCandidate(
                        symbol=context.symbol,
                        setup_type=SetupType.KEY_LEVEL_REJECTION,
                        mode=TradingMode.MICRO_SWING,
                        timestamp=context.timestamp,
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        confidence=confidence,
                        metadata={
                            'level_name': level_name,
                            'level_price': level_price,
                            'direction': 'long',
                            'r_multiple': 1.5
                        }
                    )
        
        return None


class SetupScanner:
    """Orchestrates all setup detectors"""
    
    def __init__(self):
        self.orb_detector = ORBDetector()
        self.vwap_bounce_detector = VWAPBounceDetector()
        self.key_level_detector = KeyLevelRejectionDetector()
        
        # Debouncing: track last signal per symbol per setup
        self._last_signals: Dict[Tuple[str, SetupType], datetime] = {}
        self._signal_cooldown_minutes = 15  # Min time between signals
    
    def scan_all(self, context: MarketContext) -> List[TradeCandidate]:
        """
        Scan for all setup types.
        Returns list of candidates (may be empty).
        """
        candidates = []
        
        # Try each detector
        orb_candidate = self.orb_detector.detect(context)
        if orb_candidate and self._should_emit_signal(orb_candidate):
            candidates.append(orb_candidate)
            self._record_signal(orb_candidate)
        
        vwap_candidate = self.vwap_bounce_detector.detect(context)
        if vwap_candidate and self._should_emit_signal(vwap_candidate):
            candidates.append(vwap_candidate)
            self._record_signal(vwap_candidate)
        
        key_level_candidate = self.key_level_detector.detect(context)
        if key_level_candidate and self._should_emit_signal(key_level_candidate):
            candidates.append(key_level_candidate)
            self._record_signal(key_level_candidate)
        
        return candidates
    
    def _should_emit_signal(self, candidate: TradeCandidate) -> bool:
        """Check if we should emit this signal (debouncing)"""
        key = (candidate.symbol, candidate.setup_type)
        if key not in self._last_signals:
            return True
        
        last_time = self._last_signals[key]
        minutes_since = (candidate.timestamp - last_time).total_seconds() / 60
        return minutes_since >= self._signal_cooldown_minutes
    
    def _record_signal(self, candidate: TradeCandidate):
        """Record signal for debouncing"""
        key = (candidate.symbol, candidate.setup_type)
        self._last_signals[key] = candidate.timestamp
    
    def clear_daily_state(self):
        """Clear state at end of day"""
        self.orb_detector.clear_cache()
        self._last_signals.clear()

