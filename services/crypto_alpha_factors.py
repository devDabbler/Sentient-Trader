"""
Crypto Alpha Factors

Crypto-specific technical indicators and alpha factors for ML model training.
Adapted from stock alpha factors but tailored for cryptocurrency markets:
- 24/7 market (no gaps)
- Higher volatility tolerance
- Different liquidity patterns
- Social sentiment importance
- On-chain metrics potential

Note: On-chain metrics would require additional APIs (e.g., Glassnode, IntoTheBlock)
This module focuses on price/volume-based factors that work with Kraken data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from dataclasses import dataclass



@dataclass
class CryptoAlphaFactors:
    """Container for crypto alpha factors"""
    symbol: str
    
    # Price-based factors
    returns_1h: float = 0.0
    returns_4h: float = 0.0
    returns_24h: float = 0.0
    returns_7d: float = 0.0
    
    # Volatility factors
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    volatility_7d: float = 0.0
    volatility_ratio_short_long: float = 0.0  # Short-term vs long-term vol
    
    # Volume factors
    volume_24h: float = 0.0
    volume_ratio_1h_24h: float = 0.0
    volume_surge_score: float = 0.0
    volume_trend_7d: float = 0.0
    
    # Momentum factors
    rsi_5m: float = 50.0
    rsi_1h: float = 50.0
    rsi_4h: float = 50.0
    rsi_trend: float = 0.0  # RSI slope
    
    # Trend factors
    ema_8_1h: float = 0.0
    ema_20_1h: float = 0.0
    ema_50_1h: float = 0.0
    ema_alignment_score: float = 0.0  # Bullish alignment = positive
    price_vs_ema8_pct: float = 0.0
    price_vs_ema20_pct: float = 0.0
    price_vs_vwap_pct: float = 0.0
    
    # Breakout/Support factors
    distance_from_24h_high_pct: float = 0.0
    distance_from_24h_low_pct: float = 0.0
    is_near_24h_high: bool = False
    is_near_24h_low: bool = False
    consolidation_score: float = 0.0
    
    # Market regime factors
    trend_strength: float = 0.0
    market_phase: str = "UNKNOWN"  # ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN
    
    # Liquidity factors
    bid_ask_spread_pct: float = 0.0
    depth_imbalance: float = 0.0  # Bid vs ask size
    
    # Advanced momentum
    macd_1h: float = 0.0
    macd_signal_1h: float = 0.0
    macd_histogram_1h: float = 0.0
    adx_1h: float = 0.0  # Trend strength indicator
    
    # Statistical factors
    skewness_24h: float = 0.0
    kurtosis_24h: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for ML models"""
        return {k: v for k, v in self.__dict__.items() if k != 'symbol'}
    
    def get_feature_vector(self) -> List[float]:
        """Get feature vector for ML models"""
        data = self.to_dict()
        # Convert booleans to int
        return [
            float(v) if not isinstance(v, (bool, str)) else (1.0 if v is True else 0.0)
            for v in data.values()
            if not isinstance(v, str)
        ]


class CryptoAlphaCalculator:
    """Calculate crypto-specific alpha factors from market data"""
    
    def __init__(self):
        """Initialize calculator"""
        logger.info("Crypto Alpha Calculator initialized")
    
    def calculate_factors(
        self,
        symbol: str,
        ohlcv_5m: List[Dict],
        ohlcv_1h: List[Dict],
        ticker: Dict
    ) -> CryptoAlphaFactors:
        """
        Calculate all alpha factors for a crypto asset
        
        Args:
            symbol: Crypto pair (e.g., 'BTC/USD')
            ohlcv_5m: 5-minute OHLCV candles
            ohlcv_1h: 1-hour OHLCV candles
            ticker: Current ticker data
            
        Returns:
            CryptoAlphaFactors object
        """
        try:
            # Convert to DataFrames for easier manipulation
            df_5m = pd.DataFrame(ohlcv_5m)
            df_1h = pd.DataFrame(ohlcv_1h)
            
            current_price = ticker['last_price']
            vwap = ticker.get('vwap_24h', current_price)
            
            factors = CryptoAlphaFactors(symbol=symbol)
            
            # Calculate returns
            factors.returns_1h = self._calculate_return(df_1h, periods=1)
            factors.returns_4h = self._calculate_return(df_1h, periods=4)
            factors.returns_24h = self._calculate_return(df_1h, periods=24)
            factors.returns_7d = self._calculate_return(df_1h, periods=168)  # 7 days
            
            # Calculate volatility
            factors.volatility_1h = self._calculate_volatility(df_5m, window=12)  # 1 hour of 5m candles
            factors.volatility_24h = self._calculate_volatility(df_1h, window=24)
            factors.volatility_7d = self._calculate_volatility(df_1h, window=168)
            factors.volatility_ratio_short_long = (
                factors.volatility_1h / factors.volatility_24h
                if factors.volatility_24h > 0 else 1.0
            )
            
            # Volume factors
            factors.volume_24h = ticker['volume_24h']
            factors.volume_ratio_1h_24h = self._calculate_volume_ratio(df_1h, short_window=1, long_window=24)
            factors.volume_surge_score = self._calculate_volume_surge_score(df_1h)
            factors.volume_trend_7d = self._calculate_volume_trend(df_1h, window=168)
            
            # RSI calculations
            factors.rsi_5m = self._calculate_rsi(df_5m['close'].values, period=14)
            factors.rsi_1h = self._calculate_rsi(df_1h['close'].values, period=14)
            factors.rsi_4h = self._calculate_rsi(df_1h['close'].values[-96:], period=14)  # Last 96 hours
            factors.rsi_trend = factors.rsi_1h - factors.rsi_4h
            
            # EMA calculations
            factors.ema_8_1h = self._calculate_ema(df_1h['close'].values, period=8)
            factors.ema_20_1h = self._calculate_ema(df_1h['close'].values, period=20)
            factors.ema_50_1h = self._calculate_ema(df_1h['close'].values, period=50)
            
            # EMA alignment
            factors.ema_alignment_score = self._calculate_ema_alignment(
                current_price,
                factors.ema_8_1h,
                factors.ema_20_1h,
                factors.ema_50_1h
            )
            
            factors.price_vs_ema8_pct = ((current_price - factors.ema_8_1h) / factors.ema_8_1h) * 100
            factors.price_vs_ema20_pct = ((current_price - factors.ema_20_1h) / factors.ema_20_1h) * 100
            factors.price_vs_vwap_pct = ((current_price - vwap) / vwap) * 100
            
            # Breakout/Support factors
            high_24h = ticker['high_24h']
            low_24h = ticker['low_24h']
            
            factors.distance_from_24h_high_pct = ((current_price - high_24h) / high_24h) * 100
            factors.distance_from_24h_low_pct = ((current_price - low_24h) / low_24h) * 100
            factors.is_near_24h_high = factors.distance_from_24h_high_pct > -2.0
            factors.is_near_24h_low = factors.distance_from_24h_low_pct < 2.0
            factors.consolidation_score = self._calculate_consolidation_score(df_1h)
            
            # Market regime
            factors.trend_strength = self._calculate_trend_strength(df_1h)
            factors.market_phase = self._determine_market_phase(
                factors.trend_strength,
                factors.volatility_24h,
                factors.volume_surge_score
            )
            
            # MACD calculations
            macd_data = self._calculate_macd(df_1h['close'].values)
            factors.macd_1h = macd_data['macd']
            factors.macd_signal_1h = macd_data['signal']
            factors.macd_histogram_1h = macd_data['histogram']
            
            # ADX (trend strength)
            factors.adx_1h = self._calculate_adx(df_1h, period=14)
            
            # Statistical factors
            factors.skewness_24h = self._calculate_skewness(df_1h['close'].values[-24:])
            factors.kurtosis_24h = self._calculate_kurtosis(df_1h['close'].values[-24:])
            
            # Liquidity factors (if available in ticker data)
            factors.bid_ask_spread_pct = self._calculate_spread(ticker)
            
            pass  # logger.debug(f"Calculated {len(factors.get_feature_vector()} alpha factors for {symbol}"))
            
            return factors
            
        except Exception as e:
            logger.error(f"Error calculating alpha factors for {symbol}: {e}")
            return CryptoAlphaFactors(symbol=symbol)
    
    def _calculate_return(self, df: pd.DataFrame, periods: int) -> float:
        """Calculate return over N periods"""
        if len(df) < periods + 1:
            return 0.0
        
        current = df['close'].iloc[-1]
        previous = df['close'].iloc[-periods-1]
        
        return ((current - previous) / previous) * 100
    
    def _calculate_volatility(self, df: pd.DataFrame, window: int) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if len(df) < window:
            return 0.0
        
        closes = df['close'].iloc[-window:]
        returns = closes.pct_change().dropna()
        
        return returns.std() * 100 * np.sqrt(window)  # Annualized-style
    
    def _calculate_volume_ratio(self, df: pd.DataFrame, short_window: int, long_window: int) -> float:
        """Calculate volume ratio (short vs long window)"""
        if len(df) < long_window:
            return 1.0
        
        short_vol = df['volume'].iloc[-short_window:].mean()
        long_vol = df['volume'].iloc[-long_window:].mean()
        
        return short_vol / long_vol if long_vol > 0 else 1.0
    
    def _calculate_volume_surge_score(self, df: pd.DataFrame) -> float:
        """Calculate volume surge score (0-100)"""
        if len(df) < 24:
            return 0.0
        
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[-24:].mean()
        
        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Convert to 0-100 score
        score = min((ratio - 1.0) * 50, 100)
        
        return max(0.0, score)
    
    def _calculate_volume_trend(self, df: pd.DataFrame, window: int) -> float:
        """Calculate volume trend (positive = increasing)"""
        if len(df) < window:
            return 0.0
        
        volumes = df['volume'].iloc[-window:]
        
        # Linear regression slope
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes.values, 1)[0]
        
        # Normalize by mean volume
        return slope / volumes.mean() if volumes.mean() > 0 else 0.0
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = gains[-period:].mean()
        avg_loss = losses[-period:].mean()
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices.mean() if len(prices) > 0 else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[:period].mean()
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_ema_alignment(self, price: float, ema8: float, ema20: float, ema50: float) -> float:
        """
        Calculate EMA alignment score
        Positive = bullish alignment, Negative = bearish
        """
        if ema8 == 0 or ema20 == 0:
            return 0.0
        
        # Perfect bullish: price > ema8 > ema20 > ema50
        # Perfect bearish: price < ema8 < ema20 < ema50
        
        score = 0.0
        
        if price > ema8:
            score += 25
        else:
            score -= 25
        
        if ema8 > ema20:
            score += 25
        else:
            score -= 25
        
        if ema20 > ema50:
            score += 25
        else:
            score -= 25
        
        # Bonus for strong alignment
        if price > ema8 > ema20 > ema50:
            score += 25
        elif price < ema8 < ema20 < ema50:
            score -= 25
        
        return score
    
    def _calculate_consolidation_score(self, df: pd.DataFrame, window: int = 24) -> float:
        """
        Calculate consolidation score (0-100)
        High score = tight consolidation
        """
        if len(df) < window:
            return 0.0
        
        closes = df['close'].iloc[-window:]
        
        # Calculate trading range
        price_range = (closes.max() - closes.min()) / closes.mean() * 100
        
        # Low range = high consolidation
        consolidation = max(0, 100 - (price_range * 10))
        
        return min(100.0, consolidation)
    
    def _calculate_trend_strength(self, df: pd.DataFrame, window: int = 50) -> float:
        """
        Calculate trend strength (-100 to +100)
        Positive = uptrend, Negative = downtrend
        """
        if len(df) < window:
            return 0.0
        
        closes = df['close'].iloc[-window:]
        
        # Linear regression
        x = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes.values, 1)
        
        # Calculate R-squared (trend consistency)
        y_pred = slope * x + intercept
        ss_res = np.sum((closes.values - y_pred) ** 2)
        ss_tot = np.sum((closes.values - closes.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Normalize slope by price
        normalized_slope = (slope / closes.mean()) * 100 if closes.mean() > 0 else 0
        
        # Combine slope and R-squared
        strength = normalized_slope * r_squared * 100
        
        return np.clip(strength, -100, 100)
    
    def _determine_market_phase(self, trend_strength: float, volatility: float, volume_surge: float) -> str:
        """Determine current market phase"""
        
        if trend_strength > 30 and volume_surge > 30:
            return "MARKUP"  # Strong uptrend with volume
        elif trend_strength < -30 and volume_surge > 30:
            return "MARKDOWN"  # Strong downtrend with volume
        elif abs(trend_strength) < 20 and volatility < 5:
            return "ACCUMULATION"  # Low volatility consolidation
        elif abs(trend_strength) < 20 and volatility > 10:
            return "DISTRIBUTION"  # High volatility chop
        else:
            return "UNKNOWN"
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        # For simplicity, use a fraction of MACD
        signal_line = macd * 0.9  # Simplified
        
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index) - simplified version"""
        if len(df) < period + 1:
            return 0.0
        
        # Simplified ADX calculation
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate true range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr = tr[-period:].mean() if len(tr) >= period else 0.0
        
        # Normalize
        adx = (atr / close[-1]) * 100 if close[-1] > 0 else 0.0
        
        return min(100.0, adx)
    
    def _calculate_skewness(self, prices: np.ndarray) -> float:
        """Calculate skewness of price distribution"""
        if len(prices) < 3:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        skewness = ((returns - mean_return) ** 3).mean() / (std_return ** 3)
        
        return skewness
    
    def _calculate_kurtosis(self, prices: np.ndarray) -> float:
        """Calculate kurtosis of price distribution"""
        if len(prices) < 4:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        kurtosis = ((returns - mean_return) ** 4).mean() / (std_return ** 4) - 3.0
        
        return kurtosis
    
    def _calculate_spread(self, ticker: Dict) -> float:
        """Calculate bid-ask spread percentage"""
        bid = ticker.get('bid', 0.0)
        ask = ticker.get('ask', 0.0)
        
        if bid == 0 or ask == 0:
            return 0.0
        
        mid = (bid + ask) / 2
        
        return ((ask - bid) / mid) * 100 if mid > 0 else 0.0


# Convenience function
def calculate_crypto_alpha_factors(
    symbol: str,
    ohlcv_5m: List[Dict],
    ohlcv_1h: List[Dict],
    ticker: Dict
) -> CryptoAlphaFactors:
    """Calculate alpha factors for a crypto asset"""
    calculator = CryptoAlphaCalculator()
    return calculator.calculate_factors(symbol, ohlcv_5m, ohlcv_1h, ticker)
