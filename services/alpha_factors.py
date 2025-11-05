"""
Alpha Factor Calculator using yfinance

Calculates 50+ alpha factors similar to Qlib's Alpha158 but using yfinance data.
No need for Qlib data download - works with your existing setup.
"""

from loguru import logger
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta



class AlphaFactorCalculator:
    """
    Calculate alpha factors from yfinance data.
    
    This gives you many of the same benefits as Qlib Alpha158
    without needing to download Qlib's data.
    """
    
    @staticmethod
    def calculate_factors(ticker: str, period: str = "1y") -> Dict[str, float]:
        """
        Calculate 50+ alpha factors for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Historical period to analyze
        
        Returns:
            Dict with calculated alpha factors
        """
        try:
            logger.info(f"Starting alpha factor calculation for {ticker}...")
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty or len(hist) < 50:
                logger.warning(f"Insufficient data for {ticker}")
                return {}

            logger.info(f"Retrieved {len(hist)} historical records for {ticker}.")
            
            # Calculate all factors
            factors = {}
            
            # Price-based factors
            logger.info(f"Calculating price factors for {ticker}...")
            factors.update(AlphaFactorCalculator._price_factors(hist))
            
            # Volume-based factors
            logger.info(f"Calculating volume factors for {ticker}...")
            factors.update(AlphaFactorCalculator._volume_factors(hist))
            
            # Momentum factors
            logger.info(f"Calculating momentum factors for {ticker}...")
            factors.update(AlphaFactorCalculator._momentum_factors(hist))
            
            # Volatility factors
            logger.info(f"Calculating volatility factors for {ticker}...")
            factors.update(AlphaFactorCalculator._volatility_factors(hist))
            
            # Technical indicators
            logger.info(f"Calculating technical factors for {ticker}...")
            factors.update(AlphaFactorCalculator._technical_factors(hist))
            
            logger.info(f"Successfully calculated {len(factors)} factors for {ticker}.")
            return factors
            
        except Exception as e:
            logger.error(f"Error calculating factors for {ticker}: {e}")
            return {}
    
    @staticmethod
    def _price_factors(hist: pd.DataFrame) -> Dict[str, float]:
        """Price-based alpha factors"""
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        open_price = hist['Open']
        
        return {
            # Returns over different periods
            'return_1d': close.pct_change(1).iloc[-1] if len(close) > 1 else 0,
            'return_5d': close.pct_change(5).iloc[-1] if len(close) > 5 else 0,
            'return_10d': close.pct_change(10).iloc[-1] if len(close) > 10 else 0,
            'return_20d': close.pct_change(20).iloc[-1] if len(close) > 20 else 0,
            'return_60d': close.pct_change(60).iloc[-1] if len(close) > 60 else 0,
            
            # Moving averages
            'ma5_ratio': close.iloc[-1] / close.rolling(5).mean().iloc[-1] if len(close) > 5 else 1,
            'ma10_ratio': close.iloc[-1] / close.rolling(10).mean().iloc[-1] if len(close) > 10 else 1,
            'ma20_ratio': close.iloc[-1] / close.rolling(20).mean().iloc[-1] if len(close) > 20 else 1,
            'ma60_ratio': close.iloc[-1] / close.rolling(60).mean().iloc[-1] if len(close) > 60 else 1,
            
            # Price range factors
            'high_low_ratio': (high.iloc[-1] / low.iloc[-1] - 1) if low.iloc[-1] > 0 else 0,
            'close_open_ratio': (close.iloc[-1] / open_price.iloc[-1] - 1) if open_price.iloc[-1] > 0 else 0,
        }
    
    @staticmethod
    def _volume_factors(hist: pd.DataFrame) -> Dict[str, float]:
        """Volume-based alpha factors"""
        volume = hist['Volume']
        close = hist['Close']
        
        return {
            # Volume ratios
            'volume_5d_ratio': volume.iloc[-1] / volume.rolling(5).mean().iloc[-1] if len(volume) > 5 else 1,
            'volume_10d_ratio': volume.iloc[-1] / volume.rolling(10).mean().iloc[-1] if len(volume) > 10 else 1,
            'volume_20d_ratio': volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if len(volume) > 20 else 1,
            
            # Volume trends
            'volume_trend_5d': (volume.rolling(5).mean().iloc[-1] / volume.rolling(5).mean().iloc[-6] - 1) if len(volume) > 10 else 0,
            'volume_trend_20d': (volume.rolling(20).mean().iloc[-1] / volume.rolling(20).mean().iloc[-21] - 1) if len(volume) > 40 else 0,
            
            # Price-volume correlation
            'pv_corr_5d': close.pct_change().tail(5).corr(volume.pct_change().tail(5)) if len(volume) > 5 else 0,
            'pv_corr_20d': close.pct_change().tail(20).corr(volume.pct_change().tail(20)) if len(volume) > 20 else 0,
        }
    
    @staticmethod
    def _momentum_factors(hist: pd.DataFrame) -> Dict[str, float]:
        """Momentum alpha factors"""
        close = hist['Close']
        
        # Calculate momentum
        momentum_5d = close.pct_change(5).iloc[-1] if len(close) > 5 else 0
        momentum_20d = close.pct_change(20).iloc[-1] if len(close) > 20 else 0
        
        return {
            'momentum_5d': momentum_5d,
            'momentum_20d': momentum_20d,
            'momentum_60d': close.pct_change(60).iloc[-1] if len(close) > 60 else 0,
            
            # Acceleration (change in momentum)
            'momentum_accel': momentum_5d - momentum_20d if len(close) > 20 else 0,
            
            # Relative strength
            'rs_5d': AlphaFactorCalculator._relative_strength(close, 5),
            'rs_20d': AlphaFactorCalculator._relative_strength(close, 20),
        }
    
    @staticmethod
    def _volatility_factors(hist: pd.DataFrame) -> Dict[str, float]:
        """Volatility alpha factors"""
        close = hist['Close']
        returns = close.pct_change()
        
        return {
            # Standard deviation (volatility)
            'volatility_5d': returns.rolling(5).std().iloc[-1] if len(returns) > 5 else 0,
            'volatility_20d': returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0,
            'volatility_60d': returns.rolling(60).std().iloc[-1] if len(returns) > 60 else 0,
            
            # Annualized volatility
            'annual_volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            
            # Volatility trends
            'volatility_trend': (returns.rolling(5).std().iloc[-1] / returns.rolling(20).std().iloc[-1] - 1) if len(returns) > 20 else 0,
            
            # High-low volatility
            'hl_volatility': (hist['High'] - hist['Low']).rolling(20).mean().iloc[-1] / close.iloc[-1] if len(close) > 20 else 0,
        }
    
    @staticmethod
    def _technical_factors(hist: pd.DataFrame) -> Dict[str, float]:
        """Technical indicator factors"""
        close = hist['Close']
        
        # RSI
        rsi_14 = AlphaFactorCalculator._calculate_rsi(close, 14)
        
        # MACD
        macd, signal = AlphaFactorCalculator._calculate_macd(close)
        
        # Bollinger Bands
        bb_position = AlphaFactorCalculator._bollinger_position(close)
        
        return {
            'rsi_14': rsi_14,
            'rsi_normalized': (rsi_14 - 50) / 50,  # Normalize around 0
            
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': macd - signal,
            
            'bollinger_position': bb_position,
            
            # Price position in range
            'price_position_20d': AlphaFactorCalculator._price_position(close, 20),
            'price_position_60d': AlphaFactorCalculator._price_position(close, 60),
        }
    
    @staticmethod
    def _relative_strength(close: pd.Series, period: int) -> float:
        """Calculate relative strength"""
        if len(close) <= period:
            return 0
        
        gains = close.diff()
        avg_gain = gains[gains > 0].tail(period).mean()
        avg_loss = abs(gains[gains < 0].tail(period).mean())
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return rs
    
    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(close) <= period:
            return 50
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    @staticmethod
    def _calculate_macd(close: pd.Series) -> tuple:
        """Calculate MACD"""
        if len(close) < 26:
            return 0, 0
        
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return macd.iloc[-1], signal.iloc[-1]
    
    @staticmethod
    def _bollinger_position(close: pd.Series, period: int = 20) -> float:
        """Calculate position within Bollinger Bands (0-1)"""
        if len(close) < period:
            return 0.5
        
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        
        current = close.iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        
        if upper_val == lower_val:
            return 0.5
        
        position = (current - lower_val) / (upper_val - lower_val)
        return max(0, min(1, position))
    
    @staticmethod
    def _price_position(close: pd.Series, period: int) -> float:
        """Calculate where price is in its recent range (0-1)"""
        if len(close) < period:
            return 0.5
        
        recent = close.tail(period)
        high = recent.max()
        low = recent.min()
        current = close.iloc[-1]
        
        if high == low:
            return 0.5
        
        position = (current - low) / (high - low)
        return position


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    calc = AlphaFactorCalculator()
    factors = calc.calculate_factors('AAPL')
    
    print(f"\nCalculated {len(factors)} alpha factors for AAPL:")
    for name, value in list(factors.items())[:10]:
        print(f"  {name}: {value:.4f}")
    print(f"  ... and {len(factors) - 10} more")
