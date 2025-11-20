"""
15-Minute ORB + FVG Strategy (Reddit-Inspired)

Based on the profitable strategy from r/tradingmillionaires:
- 15-minute Opening Range Breakout
- Fair Value Gap confirmation
- Once per day trading
- Clean structure, tight risk, zero guessing

Strategy Rules:
1. Mark 15-min opening range (9:30-9:45 AM)
2. Wait for breakout above/below range
3. Confirm with FVG presence in breakout direction
4. Enter on pullback to FVG or immediate breakout
5. Stop: Opposite side of ORB
6. Target: 1.5-2R (risk/reward)
"""

from loguru import logger
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import List, Optional, Dict
import yfinance as yf
import pandas as pd
import numpy as np

from services.fvg_detector import FVGDetector, FairValueGap


@dataclass
class ORBLevel:
    """Opening Range Breakout levels"""
    high: float
    low: float
    established_time: datetime
    
    def range_size(self) -> float:
        return self.high - self.low
    
    def range_pct(self) -> float:
        return (self.range_size() / self.low) * 100


@dataclass
class ORBFVGSignal:
    """Trading signal combining ORB and FVG"""
    symbol: str
    timestamp: datetime
    
    # Signal details
    signal_type: str  # 'LONG' or 'SHORT'
    confidence: float  # 0-100
    
    # ORB data
    orb_high: float
    orb_low: float
    orb_range_pct: float
    
    # FVG data
    fvg: Optional[FairValueGap]
    fvg_alignment: bool  # True if FVG aligns with breakout direction
    
    # Trade parameters
    entry_price: float
    stop_loss: float
    target_price: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    
    # Context
    current_price: float
    volume_ratio: float  # Current volume vs average
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display"""
        return {
            'symbol': self.symbol,
            'signal': self.signal_type,
            'confidence': f"{self.confidence:.1f}%",
            'entry': f"${self.entry_price:.2f}",
            'stop': f"${self.stop_loss:.2f}",
            'target': f"${self.target_price:.2f}",
            'risk_reward': f"{self.risk_reward_ratio:.1f}R",
            'orb_range': f"{self.orb_range_pct:.2f}%",
            'fvg_present': 'Yes' if self.fvg else 'No',
            'fvg_aligned': 'Yes' if self.fvg_alignment else 'No',
            'volume': f"{self.volume_ratio:.1f}x"
        }


class ORBFVGStrategy:
    """15-Minute ORB + FVG Strategy Implementation"""
    
    def __init__(self,
                 orb_minutes: int = 15,
                 min_gap_pct: float = 0.2,
                 min_volume_ratio: float = 1.5,
                 target_rr: float = 2.0):
        """
        Initialize strategy
        
        Args:
            orb_minutes: Opening range period (default 15 minutes)
            min_gap_pct: Minimum FVG size (default 0.2%)
            min_volume_ratio: Minimum volume ratio for entry (default 1.5x)
            target_rr: Target risk/reward ratio (default 2.0)
        """
        self.orb_minutes = orb_minutes
        self.min_volume_ratio = min_volume_ratio
        self.target_rr = target_rr
        
        # Initialize FVG detector
        self.fvg_detector = FVGDetector(min_gap_pct=min_gap_pct)
        
        # Cache for ORB levels (symbol -> ORBLevel)
        self._orb_cache: Dict[str, ORBLevel] = {}
        
        # Track signals to prevent duplicates
        self._signals_today: Dict[str, datetime] = {}
    
    def analyze_ticker(self,
                      symbol: str,
                      current_time: Optional[datetime] = None) -> Optional[ORBFVGSignal]:
        """
        Analyze ticker for ORB+FVG setup
        
        Args:
            symbol: Stock ticker
            current_time: Current time (default: now)
            
        Returns:
            ORBFVGSignal if valid setup found, else None
        """
        if current_time is None:
            current_time = datetime.now()
        
        current_time_only = current_time.time()
        market_open = dt_time(9, 30)
        orb_end = dt_time(9, 45)  # 15 minutes after open
        trading_window_end = dt_time(11, 0)  # Trade within 90 min of open
        
        # Only trade during market hours
        if not (dt_time(9, 30) <= current_time_only <= dt_time(16, 0)):
            return None
        
        # Check if we already have a signal today
        if symbol in self._signals_today:
            last_signal_time = self._signals_today[symbol]
            if last_signal_time.date() == current_time.date():
                logger.debug(f"{symbol}: Already have signal today, skipping")
                return None
        
        try:
            # Fetch intraday data
            ticker = yf.Ticker(symbol)
            
            # Get 5-minute bars for detailed analysis
            hist_5m = ticker.history(period='1d', interval='5m')
            
            if hist_5m.empty or len(hist_5m) < 3:
                logger.warning(f"{symbol}: Insufficient 5-minute data")
                return None
            
            # Calculate ORB levels if not cached or if new day
            orb_level = self._get_or_calculate_orb(symbol, hist_5m, current_time)
            
            if not orb_level:
                logger.debug(f"{symbol}: ORB not yet established")
                return None
            
            # Must be after ORB establishment but before trading window closes
            if not (orb_end < current_time_only < trading_window_end):
                return None
            
            # Get current price and volume data
            current_bar = hist_5m.iloc[-1]
            current_price = float(current_bar['Close'])
            current_volume = float(current_bar['Volume'])
            
            # Calculate average volume
            avg_volume = float(hist_5m['Volume'].mean())
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Volume filter
            if volume_ratio < self.min_volume_ratio:
                logger.debug(f"{symbol}: Volume too low ({volume_ratio:.1f}x)")
                return None
            
            # Convert bars to list of dicts for FVG detector
            candles = []
            for idx, row in hist_5m.iterrows():
                candles.append({
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'timestamp': idx
                })
            
            # Detect FVGs
            fvgs = self.fvg_detector.detect_fvgs(symbol, candles, current_price)
            
            # Check for bullish breakout
            if current_price > orb_level.high:
                # Look for bullish FVG to confirm
                bullish_fvg = None
                for fvg in fvgs:
                    if fvg.gap_type == 'bullish' and fvg.bottom <= orb_level.high:
                        bullish_fvg = fvg
                        break
                
                # Calculate trade parameters
                entry = current_price
                stop = orb_level.low
                risk = entry - stop
                target = entry + (risk * self.target_rr)
                
                # Confidence scoring
                confidence = self._calculate_confidence(
                    volume_ratio=volume_ratio,
                    orb_range_pct=orb_level.range_pct(),
                    has_fvg=bullish_fvg is not None,
                    fvg_strength=bullish_fvg.strength if bullish_fvg else 0
                )
                
                signal = ORBFVGSignal(
                    symbol=symbol,
                    timestamp=current_time,
                    signal_type='LONG',
                    confidence=confidence,
                    orb_high=orb_level.high,
                    orb_low=orb_level.low,
                    orb_range_pct=orb_level.range_pct(),
                    fvg=bullish_fvg,
                    fvg_alignment=bullish_fvg is not None,
                    entry_price=entry,
                    stop_loss=stop,
                    target_price=target,
                    risk_amount=risk,
                    reward_amount=target - entry,
                    risk_reward_ratio=self.target_rr,
                    current_price=current_price,
                    volume_ratio=volume_ratio
                )
                
                # Record signal
                self._signals_today[symbol] = current_time
                
                logger.info(f"{symbol}: LONG signal at ${entry:.2f}, target ${target:.2f}, stop ${stop:.2f}")
                return signal
            
            # Check for bearish breakdown
            elif current_price < orb_level.low:
                # Look for bearish FVG to confirm
                bearish_fvg = None
                for fvg in fvgs:
                    if fvg.gap_type == 'bearish' and fvg.top >= orb_level.low:
                        bearish_fvg = fvg
                        break
                
                entry = current_price
                stop = orb_level.high
                risk = stop - entry
                target = entry - (risk * self.target_rr)
                
                confidence = self._calculate_confidence(
                    volume_ratio=volume_ratio,
                    orb_range_pct=orb_level.range_pct(),
                    has_fvg=bearish_fvg is not None,
                    fvg_strength=bearish_fvg.strength if bearish_fvg else 0
                )
                
                signal = ORBFVGSignal(
                    symbol=symbol,
                    timestamp=current_time,
                    signal_type='SHORT',
                    confidence=confidence,
                    orb_high=orb_level.high,
                    orb_low=orb_level.low,
                    orb_range_pct=orb_level.range_pct(),
                    fvg=bearish_fvg,
                    fvg_alignment=bearish_fvg is not None,
                    entry_price=entry,
                    stop_loss=stop,
                    target_price=target,
                    risk_amount=risk,
                    reward_amount=entry - target,
                    risk_reward_ratio=self.target_rr,
                    current_price=current_price,
                    volume_ratio=volume_ratio
                )
                
                self._signals_today[symbol] = current_time
                
                logger.info(f"{symbol}: SHORT signal at ${entry:.2f}, target ${target:.2f}, stop ${stop:.2f}")
                return signal
        
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            return None
        
        return None
    
    def _get_or_calculate_orb(self,
                             symbol: str,
                             hist_5m: pd.DataFrame,
                             current_time: datetime) -> Optional[ORBLevel]:
        """Get or calculate ORB levels"""
        
        # Check cache
        if symbol in self._orb_cache:
            cached_orb = self._orb_cache[symbol]
            # Verify it's from today
            if cached_orb.established_time.date() == current_time.date():
                return cached_orb
        
        # Calculate ORB from first 15 minutes
        market_open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        orb_end_time = current_time.replace(hour=9, minute=45, second=0, microsecond=0)
        
        # Filter bars in ORB window
        orb_bars = hist_5m[
            (hist_5m.index >= market_open_time) & 
            (hist_5m.index < orb_end_time)
        ]
        
        if orb_bars.empty or len(orb_bars) < 2:
            return None
        
        # Calculate high and low of opening range
        orb_high = float(orb_bars['High'].max())
        orb_low = float(orb_bars['Low'].min())
        
        orb_level = ORBLevel(
            high=orb_high,
            low=orb_low,
            established_time=orb_end_time
        )
        
        # Cache it
        self._orb_cache[symbol] = orb_level
        
        logger.info(f"{symbol}: ORB established - High: ${orb_high:.2f}, Low: ${orb_low:.2f}")
        
        return orb_level
    
    def _calculate_confidence(self,
                             volume_ratio: float,
                             orb_range_pct: float,
                             has_fvg: bool,
                             fvg_strength: float) -> float:
        """Calculate signal confidence score"""
        
        # Base confidence
        confidence = 50.0
        
        # Volume bonus (up to +20)
        if volume_ratio >= 2.0:
            confidence += 20
        elif volume_ratio >= 1.5:
            confidence += 10
        
        # ORB range bonus (up to +15)
        # Ideal range: 0.5-2.0%
        if 0.5 <= orb_range_pct <= 2.0:
            confidence += 15
        elif orb_range_pct < 0.5:
            confidence += 5  # Too tight
        
        # FVG bonus (up to +20)
        if has_fvg:
            confidence += min(20, fvg_strength / 5)
        
        return min(95.0, confidence)
    
    def scan_multiple_tickers(self,
                             tickers: List[str],
                             current_time: Optional[datetime] = None) -> List[ORBFVGSignal]:
        """
        Scan multiple tickers for ORB+FVG setups
        
        Args:
            tickers: List of ticker symbols
            current_time: Current time (default: now)
            
        Returns:
            List of valid signals
        """
        signals = []
        
        for ticker in tickers:
            try:
                signal = self.analyze_ticker(ticker, current_time)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
        
        return signals
    
    def clear_cache(self):
        """Clear all caches (call at end of trading day)"""
        self._orb_cache.clear()
        self._signals_today.clear()
        self.fvg_detector.clear_cache()
        logger.info("ORB+FVG strategy cache cleared")
